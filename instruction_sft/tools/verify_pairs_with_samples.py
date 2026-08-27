#!/usr/bin/env python3
"""Compile all Instruction pairs and run cached official samples offline."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INSTRUCTION_ROOT = PROJECT_ROOT / "instruction_sft"
DEFAULT_PAIRS = INSTRUCTION_ROOT / "data" / "processed" / "dataset.jsonl"
RAW_ROOT = INSTRUCTION_ROOT / "data" / "statements" / "raw"
DEFAULT_JSON = INSTRUCTION_ROOT / "reports" / "sample_verification.json"
DEFAULT_MD = INSTRUCTION_ROOT / "reports" / "sample_verification.md"
DEFAULT_MANUAL = INSTRUCTION_ROOT / "reports" / "sample_failure_manual_review.json"
OUTPUT_LIMIT = 1024 * 1024


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_case(executable: Path, work: Path, index: int, sample: dict[str, str], timeout: float) -> dict[str, Any]:
    stdout_path, stderr_path = work / f"sample_{index}.out", work / f"sample_{index}.err"
    command = [
        "prlimit",
        f"--as={512 * 1024 * 1024}",
        "--cpu=3",
        f"--fsize={OUTPUT_LIMIT}",
        "--",
        str(executable),
    ]
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=stdout, stderr=stderr)
        try:
            process.communicate(sample["input"].encode("utf-8"), timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            return {"index": index, "status": "time_limit_exceeded", "expected": sample["output"][:500], "actual": "", "error": f"exceeded {timeout:g}s"}
    actual_bytes = stdout_path.read_bytes()
    error = stderr_path.read_text(encoding="utf-8", errors="replace")[-1000:]
    actual = actual_bytes.decode("utf-8", errors="replace")
    if len(actual_bytes) >= OUTPUT_LIMIT:
        status = "output_limit_exceeded"
    elif process.returncode != 0:
        status = "runtime_error"
    elif actual.split() == sample["output"].split():
        status = "accepted"
    else:
        status = "wrong_answer"
    return {
        "index": index,
        "status": status,
        "expected": sample["output"][:500],
        "actual": actual[:500],
        "error": error,
        "returncode": process.returncode,
    }


def verify_one(record: dict[str, Any], temp_root: Path, timeout: float) -> dict[str, Any]:
    work = temp_root / record["code_sha256"]
    work.mkdir()
    source, executable = work / "main.cpp", work / "main"
    source.write_text(record["response"], encoding="utf-8")
    try:
        compile_result = subprocess.run(
            ["g++", "-std=c++17", "-O2", "-pipe", str(source), "-o", str(executable)],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"id": record["id"], "source": record["source"], "problem_id": record["problem_id"], "compile": False, "status": "compile_timeout", "samples": 0, "passed": 0, "cases": []}
    if compile_result.returncode:
        return {
            "id": record["id"], "source": record["source"], "problem_id": record["problem_id"],
            "compile": False, "status": "compile_error", "compile_error": compile_result.stderr[-2000:],
            "samples": 0, "passed": 0, "cases": [],
        }
    raw_path = RAW_ROOT / record["source"] / f"{record['problem_id']}.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    samples = raw.get("samples") or []
    if not samples:
        return {"id": record["id"], "source": record["source"], "problem_id": record["problem_id"], "compile": True, "status": "no_executable_sample", "samples": 0, "passed": 0, "cases": []}
    parsed, cases = [], []
    for index, sample in enumerate(samples, 1):
        if not isinstance(sample, dict) or not isinstance(sample.get("input"), str) or not isinstance(sample.get("output"), str) or not sample["input"].strip() or not sample["output"].strip():
            cases.append({"index": index, "status": "unparseable_sample"})
        else:
            parsed.append((index, sample))
    for index, sample in parsed:
        cases.append(run_case(executable, work, index, sample, timeout))
    cases.sort(key=lambda item: item["index"])
    passed = sum(case["status"] == "accepted" for case in cases)
    if passed == len(cases):
        status = "all_samples_passed"
    elif passed:
        status = "partial_sample_failure"
    else:
        status = "all_samples_failed"
    return {
        "id": record["id"], "source": record["source"], "problem_id": record["problem_id"],
        "compile": True, "status": status, "samples": len(cases), "passed": passed, "cases": cases,
    }


def attach_manual_reviews(records: list[dict[str, Any]], manual_path: Path) -> list[dict[str, Any]]:
    if not manual_path.is_file():
        return []
    reviews = json.loads(manual_path.read_text(encoding="utf-8"))
    by_id = {record["id"]: record for record in records}
    for review in reviews:
        if review["id"] not in by_id:
            raise ValueError(f"manual review does not match a verification failure: {review['id']}")
        by_id[review["id"]]["manual_review"] = review
    return reviews


def summarize(records: list[dict[str, Any]], manual_reviews: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    with_samples = [record for record in records if record["samples"]]
    failures = [record for record in with_samples if record["status"] != "all_samples_passed"]
    rng = random.Random(seed)
    candidates = sorted(rng.sample(failures, min(20, len(failures))), key=lambda item: item["id"])
    pair_case_status = Counter()
    for record in records:
        pair_case_status.update({case["status"] for case in record["cases"]})
    return {
        "total_pairs": len(records),
        "compiled": sum(record["compile"] for record in records),
        "pairs_with_executable_samples": len(with_samples),
        "pairs_without_executable_samples": sum(record["status"] == "no_executable_sample" for record in records),
        "all_samples_passed": sum(record["status"] == "all_samples_passed" for record in records),
        "all_samples_passed_rate_among_verifiable": round(sum(record["status"] == "all_samples_passed" for record in records) / max(1, len(with_samples)), 6),
        "partial_sample_failure": sum(record["status"] == "partial_sample_failure" for record in records),
        "all_samples_failed": sum(record["status"] == "all_samples_failed" for record in records),
        "pairs_with_runtime_error": sum(any(case["status"] == "runtime_error" for case in record["cases"]) for record in records),
        "pairs_with_time_limit": sum(any(case["status"] == "time_limit_exceeded" for case in record["cases"]) for record in records),
        "pairs_with_output_limit": sum(any(case["status"] == "output_limit_exceeded" for case in record["cases"]) for record in records),
        "pairs_with_unparseable_sample": sum(any(case["status"] == "unparseable_sample" for case in record["cases"]) for record in records),
        "manual_review_seed": seed,
        "manual_review_completed": len(manual_reviews),
        "manual_review_category_distribution": dict(sorted(Counter(review["category"] for review in manual_reviews).items())),
        "manual_review_candidates": [
            {"id": record["id"], "status": record["status"], "samples": record["samples"], "passed": record["passed"], "first_failure": next(case for case in record["cases"] if case["status"] != "accepted")}
            for record in candidates
        ],
    }


def render_markdown(summary: dict[str, Any], manual_reviews: list[dict[str, Any]]) -> str:
    lines = [
        "# Phase 3.5 Official Sample Verification",
        "",
        "Every response is recompiled unchanged and run against cached official samples. Output comparison is whitespace-token based.",
        "",
        "| Metric | Count |",
        "| --- | ---: |",
    ]
    for key in (
        "total_pairs", "compiled", "pairs_with_executable_samples", "pairs_without_executable_samples",
        "all_samples_passed", "partial_sample_failure", "all_samples_failed", "pairs_with_runtime_error",
        "pairs_with_time_limit", "pairs_with_output_limit", "pairs_with_unparseable_sample",
    ):
        lines.append(f"| {key} | {summary[key]} |")
    lines += ["", f"All-sample pass rate among verifiable pairs: **{summary['all_samples_passed_rate_among_verifiable']:.2%}**.", "", "## Deterministic random manual review", ""]
    if manual_reviews:
        lines.append(f"Reviewed {len(manual_reviews)} failures; categories: `{json.dumps(summary['manual_review_category_distribution'], ensure_ascii=False)}`.")
        lines.extend(f"- **{review['id']}** — `{review['category']}`: {review['reason']}" for review in manual_reviews)
    else:
        lines.append("Manual review file is not present yet; the JSON report contains 20 deterministic candidates.")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MD)
    parser.add_argument("--manual-review", type=Path, default=DEFAULT_MANUAL)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reuse-results", action="store_true", help="Reuse records already stored in --json and only refresh reviews/summary")
    args = parser.parse_args()
    if shutil.which("g++") is None or shutil.which("prlimit") is None:
        parser.error("g++ and prlimit are required")
    if args.reuse_results:
        if not args.json.is_file():
            parser.error("--reuse-results requires an existing --json report")
        records = json.loads(args.json.read_text(encoding="utf-8"))["records"]
        for record in records:
            record.pop("manual_review", None)
    else:
        pairs = read_jsonl(args.pairs.resolve())
        with tempfile.TemporaryDirectory(prefix="instruction_sample_verify_") as temporary:
            temp_root = Path(temporary)
            records = []
            with ThreadPoolExecutor(max_workers=args.jobs) as executor:
                futures = {executor.submit(verify_one, pair, temp_root, args.timeout): pair["id"] for pair in pairs}
                for completed, future in enumerate(as_completed(futures), 1):
                    records.append(future.result())
                    if completed % 200 == 0 or completed == len(futures):
                        print(f"verified {completed}/{len(futures)}", flush=True)
    records.sort(key=lambda item: item["id"])
    reviews = attach_manual_reviews(records, args.manual_review.resolve())
    summary = summarize(records, reviews, args.seed)
    report = {"summary": summary, "records": records}
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(render_markdown(summary, reviews), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

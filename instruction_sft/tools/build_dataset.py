#!/usr/bin/env python3
"""Build Instruction SFT Dataset v1 from local, auditable problem/code pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from common import (
    DEFAULT_SOURCE_ROOT,
    INSTRUCTION_ROOT,
    PROJECT_ROOT,
    DatasetError,
    code_tokens,
    jaccard,
    load_style_records,
    normalize_code,
    read_jsonl,
    shingles,
    write_json,
    write_jsonl,
)


INDEX_PATH = INSTRUCTION_ROOT / "data" / "raw" / "code_index.jsonl"
STATEMENT_ROOT = INSTRUCTION_ROOT / "data" / "raw" / "statements"
PROCESSED_PATH = INSTRUCTION_ROOT / "data" / "processed" / "dataset.jsonl"
SELECTED_PATH = INSTRUCTION_ROOT / "data" / "processed" / "selected_codes.jsonl"
TRAIN_PATH = INSTRUCTION_ROOT / "data" / "splits" / "train.jsonl"
VAL_PATH = INSTRUCTION_ROOT / "data" / "splits" / "val.jsonl"
REPORT_ROOT = INSTRUCTION_ROOT / "reports"
BENCHMARK_DIR = PROJECT_ROOT / "benchmark"
MODEL_PATH = Path("/data/PersonalCoder/model")
RANDOM_SEED = 42
SIMILARITY_THRESHOLD = 0.8
STYLE_PATTERNS = (
    re.compile(r"\busing\s+namespace\s+std\s*;"),
    re.compile(r"#\s*include\s*<bits/stdc\+\+\.h>"),
    re.compile(r"\b(?:MAXN|MAXM|MAX_[A-Za-z0-9_]+)\b"),
    re.compile(r"\bios\s*::\s*sync_with_stdio\s*\("),
    re.compile(r"\bcin\s*\.\s*tie\s*\("),
    re.compile(r"(?m)^\s*(?:static\s+)?(?:const\s+)?(?:long\s+long|int|char|bool|double)\s+\w+\s*\[[^\]\n]+\]"),
)


def load_index(path: Path) -> list[dict[str, Any]]:
    records = list(read_jsonl(path))
    if len(records) != 3261:
        raise DatasetError(f"expected 3261 discovery records, found {len(records)}")
    return records


def load_benchmark() -> list[dict[str, Any]]:
    result = []
    manifest_path = BENCHMARK_DIR / "manifest.jsonl"
    for item in read_jsonl(manifest_path):
        problem_dir = BENCHMARK_DIR / str(item["path"])
        reference_path = problem_dir / "reference.cpp"
        statement_path = problem_dir / "statement.md"
        reference = reference_path.read_text(encoding="utf-8")
        reference_tokens = code_tokens(reference)
        statement = statement_path.read_text(encoding="utf-8")
        result.append(
            {
                "benchmark_id": item["id"],
                "source": str(item["oj"]).lower(),
                "problem_id": str(item["problem_id"]).upper(),
                "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
                "normalized_sha256": hashlib.sha256(normalize_code(reference).encode()).hexdigest(),
                "token_count": len(reference_tokens),
                "shingles": shingles(reference_tokens),
                "statement_shingles": shingles(code_tokens(statement)),
                "statement_token_count": len(code_tokens(statement)),
            }
        )
    if len(result) != 30:
        raise DatasetError(f"expected 30 benchmark problems, found {len(result)}")
    return result


def _valid_statement(text: str) -> bool:
    normalized = text.strip()
    if len(normalized) < 100 or "#include" in normalized:
        return False
    lowered = normalized.lower()
    has_input = "输入" in normalized or "input" in lowered
    has_output = "输出" in normalized or "output" in lowered
    return has_input and has_output


def statement_index(statement_root: Path) -> dict[tuple[str, str], tuple[str, Path]]:
    result: dict[tuple[str, str], tuple[str, Path]] = {}
    if not statement_root.is_dir():
        return result
    for path in sorted(statement_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".md", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if not _valid_statement(text):
            continue
        stem = path.stem.upper()
        source = path.parent.name.lower() if path.parent != statement_root else "unknown"
        result[(source, stem)] = (text.strip(), path)
        result.setdefault(("unknown", stem), (text.strip(), path))
    return result


def find_statement(
    source_root: Path,
    curated: dict[tuple[str, str], tuple[str, Path]],
    record: dict[str, Any],
) -> tuple[str, str] | None:
    source, problem_id = record["source"], record["problem_id"]
    for key in ((source, problem_id.upper()), ("unknown", problem_id.upper())):
        if key in curated:
            text, path = curated[key]
            return text, str(path)
    origin = source_root / record["path"]
    candidates = [origin.with_suffix(".md"), origin.with_suffix(".txt")]
    try:
        cpp_count = sum(1 for path in origin.parent.iterdir() if path.suffix.lower() == ".cpp")
    except OSError:
        cpp_count = 0
    if cpp_count == 1:
        candidates.extend((origin.parent / "statement.md", origin.parent / "statement.txt"))
    for path in candidates:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if _valid_statement(text):
            return text.strip(), str(path)
    return None


def contamination(
    record: dict[str, Any],
    code: str,
    statement: str | None,
    benchmark: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    normalized = normalize_code(code)
    normalized_sha = hashlib.sha256(normalized.encode()).hexdigest()
    tokens = code_tokens(normalized)
    code_shingles = shingles(tokens)
    statement_tokens = code_tokens(statement) if statement else []
    statement_shingles = shingles(statement_tokens) if statement else frozenset()
    matches = []
    for target in benchmark:
        reasons = []
        problem_id = str(record.get("problem_id") or "").upper()
        if problem_id and record["source"] == target["source"] and problem_id == target["problem_id"]:
            reasons.append("problem_id")
        if record["sha256"] in {target["reference_sha256"], target["normalized_sha256"]} or normalized_sha == target["normalized_sha256"]:
            reasons.append("sha256")
        size_ratio = min(len(tokens), target["token_count"]) / max(len(tokens), target["token_count"], 1)
        code_similarity = jaccard(code_shingles, target["shingles"]) if size_ratio >= 0.35 else 0.0
        if code_similarity >= SIMILARITY_THRESHOLD:
            reasons.append("code_similarity")
        statement_similarity = 0.0
        if statement:
            statement_size_ratio = min(len(statement_tokens), target["statement_token_count"]) / max(
                len(statement_tokens), target["statement_token_count"], 1
            )
            if statement_size_ratio >= 0.35:
                statement_similarity = jaccard(statement_shingles, target["statement_shingles"])
            if statement_similarity >= SIMILARITY_THRESHOLD:
                reasons.append("statement_similarity")
        if reasons:
            matches.append(
                {
                    "benchmark_problem": target["benchmark_id"],
                    "reasons": reasons,
                    "code_similarity": round(code_similarity, 6),
                    "statement_similarity": round(statement_similarity, 6),
                }
            )
    return matches


def malformed_reason(code: str) -> str | None:
    if not code.strip():
        return "empty"
    if not re.search(r"\b(?:int|signed|auto)\s+main\s*\(", code):
        return "missing_main"
    if code.count("{") != code.count("}"):
        return "unbalanced_braces"
    return None


def compile_one(temp_root: Path, record: dict[str, Any], timeout: float) -> dict[str, Any]:
    source_path = temp_root / f"{record['sha256']}.cpp"
    source_path.write_text(record["text"], encoding="utf-8")
    try:
        completed = subprocess.run(
            ["g++", "-std=c++17", "-O2", "-pipe", "-fsyntax-only", str(source_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "error": f"compile exceeded {timeout:g}s"}
    return {
        "status": "pass" if completed.returncode == 0 else "fail",
        "error": completed.stderr[-2000:] if completed.returncode else "",
    }


def style_score(code: str) -> int:
    return sum(bool(pattern.search(code)) for pattern in STYLE_PATTERNS)


def timestamp_value(value: str | None) -> float:
    if not value:
        return 0.0
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return 0.0


def token_length_distribution(lengths: list[int]) -> dict[str, Any]:
    if not lengths:
        return {"count": 0, "min": 0, "median": 0, "p90": 0, "max": 0, "mean": 0.0}
    ordered = sorted(lengths)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "median": ordered[(len(ordered) - 1) // 2],
        "p90": ordered[int((len(ordered) - 1) * 0.9)],
        "max": ordered[-1],
        "mean": round(sum(ordered) / len(ordered), 2),
    }


def load_tokenizer():
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise DatasetError("transformers is required in the project venv") from exc
    if not MODEL_PATH.is_dir():
        raise DatasetError(f"local tokenizer path not found: {MODEL_PATH}")
    return AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)


def run(source_root: Path, jobs: int, compile_timeout: float) -> dict[str, Any]:
    if not source_root.is_dir():
        raise DatasetError(f"source root not found: {source_root}")
    index = load_index(INDEX_PATH)
    text_by_sha = {record["sha256"]: record for record in load_style_records()}
    benchmark = load_benchmark()
    curated_statements = statement_index(STATEMENT_ROOT)
    candidates = []
    exclusions = []
    eligible_contamination_count = 0
    malformed = []
    statements: dict[tuple[str, str], tuple[str, str]] = {}

    for metadata in index:
        record = {**metadata, "text": text_by_sha[metadata["sha256"]]["text"]}
        statement_value = (
            find_statement(source_root, curated_statements, record)
            if record["problem_id"]
            else None
        )
        if statement_value:
            statements[(record["source"], record["problem_id"])] = statement_value
        matches = contamination(
            record, record["text"], statement_value[0] if statement_value else None, benchmark
        )
        if matches:
            exclusions.append(
                {
                    "path": record["path"],
                    "sha256": record["sha256"],
                    "source": record["source"],
                    "problem_id": record["problem_id"],
                    "matches": matches,
                }
            )
            if record["source_type"] == "solution" and record["problem_id"]:
                eligible_contamination_count += 1
            continue
        if record["source_type"] != "solution" or not record["problem_id"]:
            continue
        if reason := malformed_reason(record["text"]):
            malformed.append(
                {"path": record["path"], "sha256": record["sha256"], "reason": reason}
            )
            continue
        candidates.append(record)

    compile_results: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix="personalcoder_instruction_compile_") as temp:
        temp_root = Path(temp)
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = {
                executor.submit(compile_one, temp_root, record, compile_timeout): record
                for record in candidates
            }
            total = len(futures)
            for completed_count, future in enumerate(as_completed(futures), 1):
                record = futures[future]
                compile_results[record["sha256"]] = future.result()
                if completed_count % 250 == 0 or completed_count == total:
                    print(f"Compiled {completed_count}/{total}", flush=True)

    passing = [record for record in candidates if compile_results[record["sha256"]]["status"] == "pass"]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in passing:
        grouped[(record["source"], record["problem_id"])].append(record)
    selected = []
    alternatives = []
    for key, versions in sorted(grouped.items()):
        ranked = sorted(
            versions,
            key=lambda record: (
                timestamp_value(record.get("timestamp")),
                style_score(record["text"]),
                len(record["text"]),
                record["sha256"],
            ),
            reverse=True,
        )
        winner = ranked[0]
        selected.append(winner)
        alternatives.append(
            {
                "source": key[0],
                "problem_id": key[1],
                "selected_sha256": winner["sha256"],
                "selected_path": winner["path"],
                "other_versions": [
                    {"sha256": item["sha256"], "path": item["path"], "timestamp": item.get("timestamp")}
                    for item in ranked[1:]
                ],
            }
        )

    tokenizer = load_tokenizer()
    final_records = []
    missing_statements = []
    for record in selected:
        key = (record["source"], record["problem_id"])
        if key not in statements:
            missing_statements.append(
                {
                    "source": record["source"],
                    "problem_id": record["problem_id"],
                    "origin_code": record["path"],
                    "code_sha256": record["sha256"],
                }
            )
            continue
        statement, statement_path = statements[key]
        final_records.append(
            {
                "id": f"{record['source']}_{record['problem_id']}",
                "source": record["source"],
                "problem_id": record["problem_id"],
                "instruction": statement,
                "response": record["text"],
                "code_sha256": record["sha256"],
                "origin_path": record["path"],
                "statement_path": statement_path,
                "timestamp": record.get("timestamp"),
                "age_bucket": record.get("age_bucket", "unknown_time"),
                "verified": True,
                "verification_level": "local_cpp17_compile_only",
                "response_token_length": len(tokenizer.encode(record["text"], add_special_tokens=False)),
            }
        )

    ordered = sorted(
        final_records,
        key=lambda record: hashlib.sha256(
            f"{RANDOM_SEED}:{record['source']}:{record['problem_id']}".encode()
        ).hexdigest(),
    )
    val_count = round(len(ordered) * 0.1) if len(ordered) >= 10 else 0
    validation = ordered[:val_count]
    train = ordered[val_count:]
    train_ids = {(record["source"], record["problem_id"]) for record in train}
    val_ids = {(record["source"], record["problem_id"]) for record in validation}
    if train_ids & val_ids:
        raise DatasetError("problem ID leakage detected between train and validation")

    selected_metadata = [
        {
            "source": record["source"],
            "problem_id": record["problem_id"],
            "path": record["path"],
            "sha256": record["sha256"],
            "timestamp": record.get("timestamp"),
            "age_bucket": record.get("age_bucket", "unknown_time"),
            "compile": True,
            "style_score": style_score(record["text"]),
        }
        for record in selected
    ]
    write_jsonl(PROCESSED_PATH, ordered)
    write_jsonl(SELECTED_PATH, selected_metadata)
    write_jsonl(TRAIN_PATH, train)
    write_jsonl(VAL_PATH, validation)
    write_json(
        REPORT_ROOT / "benchmark_exclusion.json",
        {
            "policy": {
                "held_out_problems": 30,
                "similarity_threshold": SIMILARITY_THRESHOLD,
                "checks": ["problem_id", "path-derived problem_id", "SHA256", "code_similarity", "statement_similarity"],
                "action": "exclude",
            },
            "excluded_count": len(exclusions),
            "excluded_eligible_solution_count": eligible_contamination_count,
            "matched_benchmark_problems": sorted(
                {
                    match["benchmark_problem"]
                    for record in exclusions
                    for match in record["matches"]
                }
            ),
            "excluded": exclusions,
        },
    )
    write_json(REPORT_ROOT / "missing_statements.json", missing_statements)
    raw_stats = json.loads((PROJECT_ROOT / "data" / "processed" / "style" / "stats.json").read_text(encoding="utf-8"))
    compile_counts = Counter(result["status"] for result in compile_results.values())
    quality = {
        "raw_cpp_files": raw_stats.get("raw_cpp_files"),
        "deduplicated_samples": len(index),
        "duplicate_raw_files": raw_stats.get("removed_duplicate_files"),
        "empty_raw_files": raw_stats.get("filtered_empty_files"),
        "too_short_raw_files": raw_stats.get("filtered_too_short_files"),
        "non_cpp": 0,
        "non_solution_excluded": sum(record["source_type"] != "solution" for record in index),
        "unidentified_solution_excluded": sum(
            record["source_type"] == "solution" and not record["problem_id"] for record in index
        ),
        "identified_solution_candidates": sum(
            record["source_type"] == "solution" and bool(record["problem_id"]) for record in index
        ),
        "malformed": len(malformed),
        "excluded_contamination_all_samples": len(exclusions),
        "excluded_contamination_eligible_solutions": eligible_contamination_count,
        "compile_attempted": len(compile_results),
        "compile_pass": compile_counts.get("pass", 0),
        "compile_fail": compile_counts.get("fail", 0),
        "compile_timeout": compile_counts.get("timeout", 0),
        "compile_failures": [
            {
                "path": record["path"],
                "sha256": record["sha256"],
                **compile_results[record["sha256"]],
            }
            for record in candidates
            if compile_results[record["sha256"]]["status"] != "pass"
        ],
        "malformed_records": malformed,
        "selected_unique_compile_passed_problems": len(selected),
    }
    write_json(REPORT_ROOT / "code_quality.json", quality)
    write_json(
        REPORT_ROOT / "selection_metadata.json",
        {"policy": ["recent", "compile_pass", "current_style", "completeness"], "problems": alternatives},
    )

    source_distribution = Counter(record["source"] for record in ordered)
    age_distribution = Counter(record["age_bucket"] for record in ordered)
    token_distribution = token_length_distribution(
        [record["response_token_length"] for record in ordered]
    )
    summary = {
        "raw_codes": len(index),
        "identified_codes": sum(bool(record["problem_id"]) for record in index),
        "unique_identified_problems": len(
            {(record["source"], record["problem_id"]) for record in index if record["problem_id"]}
        ),
        "real_statements": len(statements),
        "benchmark_excluded": len(exclusions),
        "benchmark_excluded_eligible_solutions": eligible_contamination_count,
        "compile_pass": quality["compile_pass"],
        "compile_fail": quality["compile_fail"],
        "compile_timeout": quality["compile_timeout"],
        "final_pairs": len(ordered),
        "train": len(train),
        "validation": len(validation),
        "source_distribution": dict(source_distribution),
        "age_distribution": dict(age_distribution),
        "response_token_distribution": token_distribution,
        "missing_statement_problems": len(missing_statements),
        "training_threshold": ">=500",
        "threshold_met": len(ordered) >= 500,
        "recommendation": (
            "ready_for_instruction_sft_v1"
            if len(ordered) >= 500
            else "small_scale_only_data_is_limited"
            if len(ordered) >= 200
            else "do_not_train_collect_real_statements_and_problem_mappings"
        ),
    }
    write_json(REPORT_ROOT / "dataset_summary.json", summary)
    card = [
        "# PersonalCoder Instruction SFT Dataset v1",
        "",
        "Local-only dataset audit; no statement was generated from code and no website was crawled.",
        "",
        "| Metric | Count |",
        "| --- | ---: |",
        f"| Deduplicated historical code | {summary['raw_codes']} |",
        f"| Codes with problem ID | {summary['identified_codes']} |",
        f"| Unique identified problems | {summary['unique_identified_problems']} |",
        f"| Reliable local statements | {summary['real_statements']} |",
        f"| Benchmark-contaminated code excluded | {summary['benchmark_excluded']} |",
        f"| C++17 compile pass | {summary['compile_pass']} |",
        f"| C++17 compile fail | {summary['compile_fail']} |",
        f"| C++17 compile timeout | {summary['compile_timeout']} |",
        f"| Final instruction-response pairs | {summary['final_pairs']} |",
        f"| Train / validation | {summary['train']} / {summary['validation']} |",
        "",
        "## Distributions",
        "",
        f"- Sources: {summary['source_distribution']}",
        f"- Age buckets: {summary['age_distribution']}",
        f"- Response tokens: {summary['response_token_distribution']}",
        "",
        "## Selection and leakage policy",
        "",
        "- One compile-passed response per `(source, problem_id)`, ranked by recency, current style, then completeness.",
        "- The audited 30-problem Benchmark is excluded by ID, SHA256, path-derived ID, code similarity, and statement similarity.",
        "- Splits are deterministic by problem ID; no problem ID can cross train/validation.",
        "- `verified=true` means local C++17 compilation only; no offline tests or official OJ AC status were available.",
        "",
        "## Decision",
        "",
        f"- Threshold met: {summary['threshold_met']}",
        f"- Recommendation: `{summary['recommendation']}`",
    ]
    (REPORT_ROOT / "dataset_card.md").write_text("\n".join(card) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--compile-timeout", type=float, default=20.0)
    args = parser.parse_args()
    if args.jobs < 1 or args.compile_timeout <= 0:
        parser.error("--jobs and --compile-timeout must be positive")
    try:
        summary = run(args.source_root.resolve(), args.jobs, args.compile_timeout)
    except DatasetError as exc:
        parser.error(str(exc))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

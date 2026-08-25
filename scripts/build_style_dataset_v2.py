#!/usr/bin/env python3
"""Build an offline, time-weighted, leak-free Style LoRA v2 dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import statistics
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SOURCE_ROOT = PROJECT_ROOT / "data" / "raw" / "algorithm"
LOCAL_SOURCE_ROOT = Path("/mnt/d/algorithm")
MODEL_PATH = Path("/data/PersonalCoder/model")
V1_ROOT = PROJECT_ROOT / "data" / "processed" / "style"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "style_v2"
REPORT_PATH = PROJECT_ROOT / "outputs" / "style_v2_dataset_report.json"
SPLITS = ("train", "validation", "test")
RANDOM_SEED = 42
SIX_MONTH_DAYS = 183
TWO_YEAR_DAYS = 730
BUCKET_ORDER = (
    "recent_6_months",
    "6_to_12_months",
    "1_to_2_years",
    "older_than_2_years",
)
BUCKET_LABELS = {
    "recent_6_months": "最近半年",
    "6_to_12_months": "半年至1年",
    "1_to_2_years": "1年至2年",
    "older_than_2_years": "2年以上",
}
BUCKET_WEIGHTS = {
    "recent_6_months": 3.0,
    "6_to_12_months": 2.0,
    "1_to_2_years": 2.0,
    "older_than_2_years": 1.0,
}
BOOLEAN_STYLE_KEYS = (
    "vector",
    "static_array",
    "maxn_or_maxm",
    "bits_stdcpp",
    "using_namespace_std",
    "global_array",
    "fast_io",
    "line_comment",
    "block_comment",
    "any_comment",
)


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def choose_source_root(requested: Path | None) -> Path:
    if requested is not None:
        return requested.resolve()
    for candidate in (PROJECT_SOURCE_ROOT, LOCAL_SOURCE_ROOT):
        if candidate.is_dir():
            return candidate.resolve()
    fail(f"raw C++ source not found at {PROJECT_SOURCE_ROOT} or {LOCAL_SOURCE_ROOT}")


def normalize_content(content: bytes) -> bytes:
    return content.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def decode_source(content: bytes) -> tuple[str, str]:
    try:
        return content.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        return content.decode("gb18030"), "gb18030"


def count_lines(content: bytes) -> int:
    if not content:
        return 0
    return content.count(b"\n") + int(not content.endswith(b"\n"))


def iso_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).astimezone().isoformat()


def git_last_commit_times(source_root: Path) -> dict[str, int]:
    if not (source_root / ".git").exists():
        return {}
    result = subprocess.run(
        [
            "git",
            "-C",
            str(source_root),
            "-c",
            "core.quotepath=false",
            "log",
            "--all",
            "--format=__STYLE_V2_COMMIT__%ct",
            "--name-only",
            "--",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        return {}
    current_timestamp: int | None = None
    times: dict[str, int] = {}
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("__STYLE_V2_COMMIT__"):
            try:
                current_timestamp = int(line.removeprefix("__STYLE_V2_COMMIT__"))
            except ValueError:
                current_timestamp = None
            continue
        if current_timestamp is not None and line.endswith(".cpp"):
            times[line] = max(times.get(line, 0), current_timestamp)
    return times


def time_bucket(age_days: float) -> str:
    if age_days <= SIX_MONTH_DAYS:
        return "recent_6_months"
    if age_days <= 365:
        return "6_to_12_months"
    if age_days <= TWO_YEAR_DAYS:
        return "1_to_2_years"
    return "older_than_2_years"


def global_array_present(code: str) -> bool:
    declaration = re.compile(
        r"^\s*(?:(?:static|const|constexpr)\s+)*(?:unsigned\s+)?"
        r"(?:bool|char|short|int|long\s+long|ll|float|double)\s+"
        r"[A-Za-z_]\w*\s*\[[^\]]+\]"
    )
    depth = 0
    in_block_comment = False
    for line in code.splitlines():
        cleaned = line
        if in_block_comment:
            if "*/" not in cleaned:
                continue
            cleaned = cleaned.split("*/", 1)[1]
            in_block_comment = False
        if "/*" in cleaned:
            before, after = cleaned.split("/*", 1)
            cleaned = before
            if "*/" not in after:
                in_block_comment = True
        cleaned = cleaned.split("//", 1)[0]
        if depth == 0 and declaration.search(cleaned):
            return True
        depth += cleaned.count("{") - cleaned.count("}")
        depth = max(depth, 0)
    return False


def style_metrics(code: str) -> dict[str, Any]:
    static_array = bool(
        re.search(
            r"(?m)^\s*(?:(?:static|const|constexpr)\s+)*(?:unsigned\s+)?"
            r"(?:bool|char|short|int|long\s+long|ll|float|double)\s+"
            r"[A-Za-z_]\w*\s*\[[^\]]+\]",
            code,
        )
    )
    declarations = re.findall(
        r"\b(?:bool|char|short|int|long\s+long|ll|float|double|string|auto)\s+"
        r"([A-Za-z_]\w*)",
        code,
    )
    single_letter = sum(len(name) == 1 for name in declarations)
    snake_case = sum("_" in name.strip("_") for name in declarations)
    camel_case = sum(bool(re.search(r"[a-z][A-Z]", name)) for name in declarations)
    line_comments = len(re.findall(r"//[^\n]*", code))
    block_comments = len(re.findall(r"/\*.*?\*/", code, flags=re.DOTALL))
    sync_io = bool(re.search(r"\bios\s*::\s*sync_with_stdio\s*\(\s*(?:false|0)", code))
    untie_io = bool(re.search(r"\bcin\s*\.\s*tie\s*\(\s*(?:nullptr|NULL|0)", code))
    return {
        "vector": bool(re.search(r"\b(?:std::)?vector\s*<", code)),
        "static_array": static_array,
        "maxn_or_maxm": bool(re.search(r"\bMAX[NM]\b", code)),
        "bits_stdcpp": bool(
            re.search(r"#\s*include\s*<\s*bits/stdc\+\+\.h\s*>", code)
        ),
        "using_namespace_std": bool(re.search(r"\busing\s+namespace\s+std\s*;", code)),
        "global_array": global_array_present(code),
        "fast_io": sync_io or untie_io,
        "line_comment": line_comments > 0,
        "block_comment": block_comments > 0,
        "any_comment": line_comments + block_comments > 0,
        "line_comment_count": line_comments,
        "block_comment_count": block_comments,
        "variable_declaration_count": len(declarations),
        "single_letter_variable_count": single_letter,
        "snake_case_variable_count": snake_case,
        "camel_case_variable_count": camel_case,
        "variable_name_characters": sum(len(name) for name in declarations),
    }


def deterministic_probability(seed: int, digest: str) -> float:
    value = hashlib.sha256(f"{seed}:{digest}".encode()).digest()
    return int.from_bytes(value, "big") / (1 << (8 * len(value)))


def percentile(values: list[int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    interpolation = position - lower
    return round(ordered[lower] * (1 - interpolation) + ordered[upper] * interpolation, 2)


def token_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    values = [int(sample["token_count"]) for sample in samples]
    return {
        "total_tokens": sum(values),
        "minimum": min(values) if values else 0,
        "mean": round(statistics.mean(values), 2) if values else 0.0,
        "median": round(statistics.median(values), 2) if values else 0.0,
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "maximum": max(values) if values else 0,
    }


def distribution(samples: list[dict[str, Any]]) -> dict[str, Any]:
    total_count = len(samples)
    total_tokens = sum(int(sample["token_count"]) for sample in samples)
    result: dict[str, Any] = {}
    for bucket in BUCKET_ORDER:
        selected = [sample for sample in samples if sample["time_bucket"] == bucket]
        tokens = sum(int(sample["token_count"]) for sample in selected)
        result[bucket] = {
            "label": BUCKET_LABELS[bucket],
            "weight": BUCKET_WEIGHTS[bucket],
            "sample_probability": BUCKET_WEIGHTS[bucket] / 3.0,
            "samples": len(selected),
            "sample_share": round(len(selected) / total_count, 6) if total_count else 0.0,
            "tokens": tokens,
            "token_share": round(tokens / total_tokens, 6) if total_tokens else 0.0,
        }
    return result


def style_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(samples)
    summary: dict[str, Any] = {"sample_count": count}
    for key in BOOLEAN_STYLE_KEYS:
        hits = sum(bool(sample["style"][key]) for sample in samples)
        summary[key] = {"count": hits, "rate": round(hits / count, 6) if count else 0.0}
    variable_count = sum(sample["style"]["variable_declaration_count"] for sample in samples)
    name_characters = sum(sample["style"]["variable_name_characters"] for sample in samples)
    summary["variable_naming"] = {
        "declarations": variable_count,
        "single_letter_rate": round(
            sum(sample["style"]["single_letter_variable_count"] for sample in samples)
            / variable_count,
            6,
        ) if variable_count else 0.0,
        "snake_case_rate": round(
            sum(sample["style"]["snake_case_variable_count"] for sample in samples)
            / variable_count,
            6,
        ) if variable_count else 0.0,
        "camel_case_rate": round(
            sum(sample["style"]["camel_case_variable_count"] for sample in samples)
            / variable_count,
            6,
        ) if variable_count else 0.0,
        "average_name_length": round(name_characters / variable_count, 3) if variable_count else 0.0,
    }
    summary["comments"] = {
        "line_total": sum(sample["style"]["line_comment_count"] for sample in samples),
        "block_total": sum(sample["style"]["block_comment_count"] for sample in samples),
        "average_per_sample": round(
            sum(
                sample["style"]["line_comment_count"]
                + sample["style"]["block_comment_count"]
                for sample in samples
            ) / count,
            4,
        ) if count else 0.0,
    }
    return summary


def compare_recent_to_baseline(
    by_bucket: dict[str, dict[str, Any]], baseline_bucket: str
) -> dict[str, Any]:
    recent = by_bucket["recent_6_months"]
    baseline = by_bucket[baseline_bucket]
    rate_deltas = {
        key: round((recent[key]["rate"] - baseline[key]["rate"]) * 100, 3)
        for key in BOOLEAN_STYLE_KEYS
    }
    variable_deltas = {
        key: round(
            (recent["variable_naming"][key] - baseline["variable_naming"][key]) * 100,
            3,
        )
        for key in ("single_letter_rate", "snake_case_rate", "camel_case_rate")
    }
    material_metrics = sum(abs(rate_deltas[key]) >= 10 for key in (
        "vector", "static_array", "maxn_or_maxm", "bits_stdcpp",
        "using_namespace_std", "global_array", "fast_io", "any_comment",
    ))
    return {
        "baseline_bucket": baseline_bucket,
        "baseline_sample_count": baseline["sample_count"],
        "recent_minus_baseline_percentage_points": rate_deltas,
        "variable_naming_percentage_point_deltas": variable_deltas,
        "average_variable_name_length_delta": round(
            recent["variable_naming"]["average_name_length"]
            - baseline["variable_naming"]["average_name_length"],
            3,
        ),
        "average_comments_per_sample_delta": round(
            recent["comments"]["average_per_sample"]
            - baseline["comments"]["average_per_sample"],
            3,
        ),
        "clear_style_difference": material_metrics >= 3,
        "material_metric_count": material_metrics,
        "criterion": "at least 3 core style metrics differ by 10 percentage points or more",
    }


def load_v1_index() -> tuple[dict[str, str], dict[str, Any]]:
    index: dict[str, str] = {}
    if not all((V1_ROOT / f"{split}.jsonl").is_file() for split in SPLITS):
        return index, {}
    for split in SPLITS:
        with (V1_ROOT / f"{split}.jsonl").open(encoding="utf-8") as source:
            for line in source:
                record = json.loads(line)
                index[record["path"]] = record["sha256"]
    stats_path = V1_ROOT / "stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.is_file() else {}
    return index, stats


def write_jsonl(path: Path, samples: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as output:
        for sample in samples:
            record = {key: value for key, value in sample.items() if key != "style"}
            output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path)
    args = parser.parse_args()
    source_root = choose_source_root(args.source_root)
    if not MODEL_PATH.is_dir():
        fail(f"local tokenizer not found: {MODEL_PATH}")
    if OUTPUT_ROOT.exists():
        fail(f"refusing to overwrite v2 dataset: {OUTPUT_ROOT}")
    if REPORT_PATH.exists():
        fail(f"refusing to overwrite report: {REPORT_PATH}")

    reference_time = datetime.now().astimezone()
    git_times = git_last_commit_times(source_root)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    cpp_files = sorted(path for path in source_root.rglob("*.cpp") if path.is_file())
    samples: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    counters = Counter()
    path_date_pattern = re.compile(r"(?:^|/)(20\d{6})(?:/|$)")

    for index, path in enumerate(cpp_files, start=1):
        raw = path.read_bytes()
        normalized = normalize_content(raw)
        if not normalized:
            counters["empty"] += 1
            continue
        if count_lines(normalized) < 10:
            counters["too_short"] += 1
            continue
        digest = hashlib.sha256(normalized).hexdigest()
        if digest in seen_hashes:
            counters["duplicates"] += 1
            continue
        seen_hashes.add(digest)
        text, encoding = decode_source(normalized)
        counters[f"encoding_{encoding}"] += 1
        relative_path = path.relative_to(source_root).as_posix()
        modified_timestamp = path.stat().st_mtime
        age_days = max(0.0, (reference_time.timestamp() - modified_timestamp) / 86400)
        bucket = time_bucket(age_days)
        git_timestamp = git_times.get(relative_path)
        match = path_date_pattern.search(relative_path)
        samples.append(
            {
                "path": relative_path,
                "source_type": "template" if "templates" in Path(relative_path).parts else "solution",
                "sha256": digest,
                "text": text,
                "modified_at": iso_timestamp(modified_timestamp),
                "git_last_commit_at": iso_timestamp(git_timestamp) if git_timestamp else None,
                "file_creation_time": None,
                "path_embedded_date": match.group(1) if match else None,
                "time_source": "filesystem_mtime",
                "age_days": round(age_days, 3),
                "time_bucket": bucket,
                "time_weight": BUCKET_WEIGHTS[bucket],
                "sampling_probability": BUCKET_WEIGHTS[bucket] / 3.0,
                "token_count": len(tokenizer.encode(text, add_special_tokens=False)),
                "style": style_metrics(text),
            }
        )
        if index % 500 == 0:
            print(f"Analyzed {index}/{len(cpp_files)} raw files", flush=True)

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(samples)
    train_end = int(len(samples) * 0.90)
    validation_end = train_end + int(len(samples) * 0.05)
    train_candidates = samples[:train_end]
    split_samples = {
        "train": [
            sample for sample in train_candidates
            if deterministic_probability(RANDOM_SEED, sample["sha256"])
            < sample["sampling_probability"]
        ],
        "validation": samples[train_end:validation_end],
        "test": samples[validation_end:],
    }
    split_hashes = {split: {sample["sha256"] for sample in values} for split, values in split_samples.items()}
    if any(split_hashes[left] & split_hashes[right] for left, right in (
        ("train", "validation"), ("train", "test"), ("validation", "test")
    )):
        fail("SHA256 leakage detected across splits")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    for split in SPLITS:
        write_jsonl(OUTPUT_ROOT / f"{split}.jsonl", split_samples[split])
    final_samples = [sample for split in SPLITS for sample in split_samples[split]]
    style_by_bucket = {
        bucket: style_summary([sample for sample in samples if sample["time_bucket"] == bucket])
        for bucket in BUCKET_ORDER
    }
    v1_index, v1_stats = load_v1_index()
    current_index = {sample["path"]: sample["sha256"] for sample in samples}
    shared_paths = set(v1_index) & set(current_index)
    input_distribution = distribution(samples)
    baseline_bucket = (
        "older_than_2_years"
        if input_distribution["older_than_2_years"]["samples"] >= 30
        else "1_to_2_years"
    )
    recent_share = input_distribution["recent_6_months"]["sample_share"]
    weighted_recent_mass = sum(
        sample["time_weight"] for sample in samples if sample["time_bucket"] == "recent_6_months"
    )
    total_weighted_mass = sum(sample["time_weight"] for sample in samples)
    weighted_recent_share = weighted_recent_mass / total_weighted_mass if total_weighted_mass else 0.0
    if input_distribution["older_than_2_years"]["samples"] < 30:
        recommendation = (
            "2年以上样本不足30个，1.0权重档缺乏统计意义；本次保留默认权重，"
            "后续建议改为最近半年3.0、半年至1年2.0、1年以上1.0。"
        )
    elif weighted_recent_share > 0.70:
        recommendation = "近期样本加权后占比超过70%，建议将最近半年权重由3.0降至2.0以避免过度集中。"
    elif recent_share < 0.10:
        recommendation = "近期样本不足10%，默认3.0权重合理；若训练仍无明显迁移，可补充近期代码而非继续加权。"
    else:
        recommendation = "默认3.0/2.0/1.0权重与当前分布匹配，暂不建议调整。"

    report = {
        "generated_at": reference_time.isoformat(),
        "source": {
            "root": str(source_root),
            "raw_cpp_files": len(cpp_files),
            "metadata_availability": {
                "filesystem_mtime": len(cpp_files),
                "git_last_commit_time": sum(sample["git_last_commit_at"] is not None for sample in samples),
                "file_creation_time": 0,
                "path_embedded_date": sum(sample["path_embedded_date"] is not None for sample in samples),
                "ctime_note": "WSL DrvFs ctime is metadata-change time, not file creation time",
            },
            "time_source_policy": "Use filesystem mtime for weighting; retain Git time for audit because batch imports can make Git time newer than the original edit time.",
        },
        "dataset": {
            "random_seed": RANDOM_SEED,
            "raw_samples": len(cpp_files),
            "filtered_empty": counters["empty"],
            "filtered_too_short": counters["too_short"],
            "removed_sha256_duplicates": counters["duplicates"],
            "valid_unique_samples": len(samples),
            "train_candidates_before_weighting": len(train_candidates),
            "final_samples": len(final_samples),
            "split_samples": {split: len(values) for split, values in split_samples.items()},
            "split_sha256_overlap": 0,
            "complete_code_preserved": True,
            "output_root": str(OUTPUT_ROOT),
            "encoding_counts": {
                "utf-8": counters["encoding_utf-8"],
                "gb18030": counters["encoding_gb18030"],
            },
        },
        "sampling": {
            "strategy": "deterministic acceptance sampling without replacement on train only",
            "formula": "sample_probability = time_weight / max_time_weight",
            "validation_and_test_unweighted": True,
            "weights": {
                "recent_6_months": 3.0,
                "6_months_to_2_years": 2.0,
                "older_than_2_years": 1.0,
            },
            "adjustment_recommendation": recommendation,
        },
        "time_distribution": {
            "valid_unique_sources": input_distribution,
            "final_dataset": distribution(final_samples),
        },
        "weight_distribution": {
            str(weight): {
                "source_samples": sum(sample["time_weight"] == weight for sample in samples),
                "final_samples": sum(sample["time_weight"] == weight for sample in final_samples),
            }
            for weight in (3.0, 2.0, 1.0)
        },
        "token_distribution": {
            "valid_unique_sources": token_summary(samples),
            "final_dataset": token_summary(final_samples),
            "per_time_bucket": {
                bucket: token_summary([sample for sample in samples if sample["time_bucket"] == bucket])
                for bucket in BUCKET_ORDER
            },
            "per_split": {split: token_summary(values) for split, values in split_samples.items()},
        },
        "style_change": {
            "per_time_bucket": style_by_bucket,
            "recent_vs_baseline": compare_recent_to_baseline(
                style_by_bucket, baseline_bucket
            ),
        },
        "v1_comparison": {
            "v1_valid_samples": v1_stats.get("valid_samples"),
            "v1_split_samples": {
                split: v1_stats.get(f"{split}_samples") for split in SPLITS
            },
            "v1_equal_weighting": True,
            "v2_final_samples": len(final_samples),
            "shared_paths": len(shared_paths),
            "same_sha256_on_shared_paths": sum(v1_index[path] == current_index[path] for path in shared_paths),
            "changed_sha256_on_shared_paths": sum(v1_index[path] != current_index[path] for path in shared_paths),
            "v2_additional_fields": [
                "modified_at", "git_last_commit_at", "time_bucket", "time_weight",
                "sampling_probability", "token_count",
            ],
            "difference": "v1 keeps every unique source equally; v2 keeps validation/test intact and time-weights train sampling while preserving complete source records.",
        },
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("x", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(json.dumps({
        "dataset": report["dataset"],
        "time_distribution": report["time_distribution"]["valid_unique_sources"],
        "style_change": report["style_change"]["recent_vs_baseline"],
        "recommendation": recommendation,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

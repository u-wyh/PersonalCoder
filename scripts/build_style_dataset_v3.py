#!/usr/bin/env python3
"""Build the complete Style LoRA v3 dataset with dynamic time weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V1_ROOT = PROJECT_ROOT / "data" / "processed" / "style"
V2_REPORT_PATH = PROJECT_ROOT / "outputs" / "style_v2_dataset_report.json"
DEFAULT_SOURCE_ROOT = Path("/mnt/d/algorithm")
MODEL_PATH = Path("/data/PersonalCoder/model")
OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "style_v3"
SAMPLING_PATH = PROJECT_ROOT / "outputs" / "style_v3_sampling.json"
REPORT_PATH = PROJECT_ROOT / "outputs" / "style_v3_dataset_report.json"
STYLE_ANALYSIS_PATH = PROJECT_ROOT / "outputs" / "style_time_analysis.json"
SPLITS = ("train", "validation", "test")
SIX_MONTH_DAYS = 183
ONE_YEAR_DAYS = 365
TWO_YEAR_DAYS = 730
BUCKET_ORDER = (
    "recent_6_months",
    "6_to_12_months",
    "1_to_2_years",
    "older_than_2_years",
    "unknown_time",
)
BUCKET_LABELS = {
    "recent_6_months": "最近半年",
    "6_to_12_months": "半年至1年",
    "1_to_2_years": "1年至2年",
    "older_than_2_years": "2年以上",
    "unknown_time": "时间信息不可靠",
}
BUCKET_WEIGHTS = {
    "recent_6_months": 4.0,
    "6_to_12_months": 3.0,
    "1_to_2_years": 1.0,
    "older_than_2_years": 0.5,
    "unknown_time": 1.0,
}
STYLE_KEYS = (
    "vector",
    "static_array",
    "maxn_or_maxm",
    "using_namespace_std",
    "bits_stdcpp",
    "ios_sync_with_stdio",
    "cin_tie",
    "global_array",
)


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def normalize_content(content: bytes) -> bytes:
    return content.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def time_bucket(age_days: float) -> str:
    if age_days <= SIX_MONTH_DAYS:
        return "recent_6_months"
    if age_days <= ONE_YEAR_DAYS:
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


def style_metrics(code: str) -> dict[str, bool]:
    return {
        "vector": bool(re.search(r"\b(?:std::)?vector\s*<", code)),
        "static_array": bool(
            re.search(
                r"(?m)^\s*(?:(?:static|const|constexpr)\s+)*(?:unsigned\s+)?"
                r"(?:bool|char|short|int|long\s+long|ll|float|double)\s+"
                r"[A-Za-z_]\w*\s*\[[^\]]+\]",
                code,
            )
        ),
        "maxn_or_maxm": bool(re.search(r"\bMAX[NM]\b", code)),
        "using_namespace_std": bool(re.search(r"\busing\s+namespace\s+std\s*;", code)),
        "bits_stdcpp": bool(
            re.search(r"#\s*include\s*<\s*bits/stdc\+\+\.h\s*>", code)
        ),
        "ios_sync_with_stdio": bool(
            re.search(r"\bios\s*::\s*sync_with_stdio\s*\(", code)
        ),
        "cin_tie": bool(re.search(r"\bcin\s*\.\s*tie\s*\(", code)),
        "global_array": global_array_present(code),
    }


def percentile(values: list[int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    ratio = position - lower
    return round(ordered[lower] * (1 - ratio) + ordered[upper] * ratio, 2)


def token_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    values = [int(record["token_length"]) for record in records]
    return {
        "samples": len(values),
        "total_tokens": sum(values),
        "mean_tokens": round(statistics.mean(values), 2) if values else 0.0,
        "median_tokens": round(statistics.median(values), 2) if values else 0.0,
        "p90_tokens": percentile(values, 0.90),
        "maximum_tokens": max(values) if values else 0,
    }


def time_distribution(records: list[dict[str, Any]]) -> dict[str, Any]:
    total_samples = len(records)
    total_tokens = sum(int(record["token_length"]) for record in records)
    result: dict[str, Any] = {}
    for bucket in BUCKET_ORDER:
        selected = [record for record in records if record["time_bucket"] == bucket]
        tokens = sum(int(record["token_length"]) for record in selected)
        result[bucket] = {
            "label": BUCKET_LABELS[bucket],
            "time_weight": BUCKET_WEIGHTS[bucket],
            "samples": len(selected),
            "sample_share": round(len(selected) / total_samples, 6) if total_samples else 0.0,
            "tokens": tokens,
            "token_share": round(tokens / total_tokens, 6) if total_tokens else 0.0,
        }
    return result


def style_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    metrics: dict[str, Any] = {}
    for key in STYLE_KEYS:
        count = sum(bool(record["style"][key]) for record in records)
        metrics[key] = {
            "count": count,
            "rate": round(count / total, 6) if total else 0.0,
        }
    return {"samples": total, "metrics": metrics}


def read_v1() -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for split in SPLITS:
        path = V1_ROOT / f"{split}.jsonl"
        if not path.is_file():
            fail(f"missing v1 split: {path}")
        records = []
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                try:
                    record = json.loads(line)
                    for key in ("path", "source_type", "sha256", "text"):
                        if key not in record:
                            raise KeyError(key)
                except (json.JSONDecodeError, KeyError) as error:
                    fail(f"invalid {split} record at line {line_number}: {error}")
                records.append(record)
        result[split] = records
    return result


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as output:
        for record in records:
            dataset_record = {
                key: value for key, value in record.items()
                if key not in ("split", "style", "time_reliable")
            }
            output.write(json.dumps(dataset_record, ensure_ascii=False, separators=(",", ":")))
            output.write("\n")


def load_v2_comparison() -> dict[str, Any] | None:
    if not V2_REPORT_PATH.is_file():
        return None
    report = json.loads(V2_REPORT_PATH.read_text(encoding="utf-8"))
    final = report["token_distribution"]["final_dataset"]
    return {
        "samples": report["dataset"]["final_samples"],
        "total_tokens": final["total_tokens"],
        "mean_tokens": final["mean"],
        "time_distribution": report["time_distribution"]["final_dataset"],
        "strategy": "3/2/1 deterministic sampling without replacement",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    if not source_root.is_dir():
        fail(f"raw source root not found: {source_root}")
    if not MODEL_PATH.is_dir():
        fail(f"local tokenizer not found: {MODEL_PATH}")
    targets = (OUTPUT_ROOT, SAMPLING_PATH, REPORT_PATH, STYLE_ANALYSIS_PATH)
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        fail("refusing to overwrite existing v3 output(s): " + ", ".join(existing))

    reference_time = datetime.now().astimezone()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    v1_splits = read_v1()
    processed_splits: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    metadata: list[dict[str, Any]] = []
    reliability = Counter()

    for split in SPLITS:
        for index, v1_record in enumerate(v1_splits[split], start=1):
            source_path = source_root / v1_record["path"]
            reliable = False
            modified_at: str | None = None
            bucket = "unknown_time"
            if source_path.is_file():
                normalized = normalize_content(source_path.read_bytes())
                current_digest = hashlib.sha256(normalized).hexdigest()
                if current_digest == v1_record["sha256"]:
                    modified_timestamp = source_path.stat().st_mtime
                    age_days = max(
                        0.0,
                        (reference_time.timestamp() - modified_timestamp) / 86400,
                    )
                    bucket = time_bucket(age_days)
                    modified_at = datetime.fromtimestamp(
                        modified_timestamp
                    ).astimezone().isoformat()
                    reliable = True
                    reliability["sha256_matched_mtime"] += 1
                else:
                    reliability["source_sha256_mismatch"] += 1
            else:
                reliability["source_missing"] += 1

            text = v1_record["text"]
            token_length = len(tokenizer.encode(text, add_special_tokens=False))
            weight = BUCKET_WEIGHTS[bucket]
            record = {
                **v1_record,
                "split": split,
                "modified_at": modified_at,
                "time_bucket": bucket,
                "time_weight": weight,
                "token_length": token_length,
                "time_reliable": reliable,
                "style": style_metrics(text),
            }
            processed_splits[split].append(record)
            metadata.append(
                {
                    "path": record["path"],
                    "split": split,
                    "token_length": token_length,
                    "time_weight": weight,
                }
            )
            if index % 500 == 0:
                print(f"Processed {split} {index}/{len(v1_splits[split])}", flush=True)

    split_hashes = {
        split: {record["sha256"] for record in records}
        for split, records in processed_splits.items()
    }
    if any(
        split_hashes[left] & split_hashes[right]
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        )
    ):
        fail("source SHA256 leakage detected across v1 splits")
    all_records = [record for split in SPLITS for record in processed_splits[split]]
    if len({record["sha256"] for record in all_records}) != len(all_records):
        fail("duplicate SHA256 detected in v1 source records")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    for split in SPLITS:
        write_jsonl(OUTPUT_ROOT / f"{split}.jsonl", processed_splits[split])
    SAMPLING_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SAMPLING_PATH.open("x", encoding="utf-8") as output:
        json.dump(
            {
                "strategy": "dynamic weighted sampling",
                "formula": "sample_probability proportional to time_weight",
                "training_logic_modified": False,
                "sample_count": len(metadata),
                "samples": metadata,
            },
            output,
            ensure_ascii=False,
            indent=2,
        )
        output.write("\n")

    distribution = time_distribution(all_records)
    per_bucket_style = {
        bucket: style_summary(
            [record for record in all_records if record["time_bucket"] == bucket]
        )
        for bucket in BUCKET_ORDER
    }
    baseline_bucket = (
        "older_than_2_years"
        if per_bucket_style["older_than_2_years"]["samples"] >= 30
        else "1_to_2_years"
    )
    recent_metrics = per_bucket_style["recent_6_months"]["metrics"]
    baseline_metrics = per_bucket_style[baseline_bucket]["metrics"]
    deltas = {
        key: round((recent_metrics[key]["rate"] - baseline_metrics[key]["rate"]) * 100, 3)
        for key in STYLE_KEYS
    }
    style_analysis = {
        "generated_at": reference_time.isoformat(),
        "time_source": "filesystem mtime accepted only when raw-source SHA256 matches v1",
        "time_distribution": distribution,
        "per_time_bucket": per_bucket_style,
        "recent_comparison": {
            "baseline_bucket": baseline_bucket,
            "baseline_samples": per_bucket_style[baseline_bucket]["samples"],
            "recent_minus_baseline_percentage_points": deltas,
            "clear_style_shift": sum(abs(value) >= 10 for value in deltas.values()) >= 3,
            "criterion": "at least 3 metrics differ by 10 percentage points or more",
        },
    }
    with STYLE_ANALYSIS_PATH.open("x", encoding="utf-8") as output:
        json.dump(style_analysis, output, ensure_ascii=False, indent=2)
        output.write("\n")

    v1_stats_path = V1_ROOT / "stats.json"
    v1_stats = json.loads(v1_stats_path.read_text(encoding="utf-8"))
    v3_tokens = token_summary(all_records)
    comparison = {
        "v1": {
            "samples": len(all_records),
            "total_tokens": v3_tokens["total_tokens"],
            "mean_tokens": v3_tokens["mean_tokens"],
            "time_distribution": distribution,
            "strategy": "all samples have equal training weight",
        },
        "v2": load_v2_comparison(),
        "v3": {
            "samples": len(all_records),
            "total_tokens": v3_tokens["total_tokens"],
            "mean_tokens": v3_tokens["mean_tokens"],
            "time_distribution": distribution,
            "strategy": "retain all samples; dynamic probability proportional to time_weight",
        },
    }
    report = {
        "generated_at": reference_time.isoformat(),
        "source": {
            "dataset": str(V1_ROOT),
            "raw_cpp_files": v1_stats.get("raw_cpp_files"),
            "valid_samples": len(all_records),
            "time_metadata": dict(reliability),
        },
        "dataset": {
            "output_root": str(OUTPUT_ROOT),
            "original_samples": v1_stats.get("raw_cpp_files"),
            "valid_samples": len(all_records),
            "final_samples": len(all_records),
            "all_valid_samples_retained": True,
            "split_samples": {
                split: len(records) for split, records in processed_splits.items()
            },
            "split_sha256_overlap": 0,
            "complete_code_preserved": True,
            "token_distribution": v3_tokens,
        },
        "time_distribution": distribution,
        "weight_distribution": {
            str(weight): {
                "samples": sum(record["time_weight"] == weight for record in all_records),
                "tokens": sum(
                    record["token_length"]
                    for record in all_records
                    if record["time_weight"] == weight
                ),
            }
            for weight in (4.0, 3.0, 1.0, 0.5)
        },
        "sampling": {
            "metadata_path": str(SAMPLING_PATH),
            "sample_probability": "proportional to time_weight",
            "training_logic_modified": False,
        },
        "comparison": comparison,
        "style_analysis_path": str(STYLE_ANALYSIS_PATH),
    }
    with REPORT_PATH.open("x", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(
        json.dumps(
            {
                "dataset": report["dataset"],
                "time_distribution": distribution,
                "weight_distribution": report["weight_distribution"],
                "recent_comparison": style_analysis["recent_comparison"],
                "comparison": comparison,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Analyze untruncated token lengths for the style training split."""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path("/data/PersonalCoder/model")
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "style" / "train.jsonl"
REPORT_PATH = PROJECT_ROOT / "outputs" / "token_length_report.json"
SEQUENCE_LENGTHS = (512, 768, 1024, 1536, 2048)


def nearest_rank(sorted_values: list[int], percentile: float) -> int:
    index = max(0, math.ceil(percentile * len(sorted_values)) - 1)
    return sorted_values[index]


def load_token_lengths(tokenizer: object) -> list[int]:
    lengths: list[int] = []
    with DATASET_PATH.open(encoding="utf-8") as dataset:
        for line_number, line in enumerate(dataset, start=1):
            try:
                sample = json.loads(line)
                text = sample["text"]
            except (json.JSONDecodeError, KeyError) as error:
                raise ValueError(f"Invalid sample at line {line_number}: {error}") from error
            if not isinstance(text, str):
                raise ValueError(f"Invalid text field at line {line_number}: expected string")
            token_ids = tokenizer.encode(
                text,
                add_special_tokens=False,
                truncation=False,
            )
            lengths.append(len(token_ids))
    return lengths


def build_report(lengths: list[int]) -> dict[str, object]:
    if not lengths:
        raise ValueError("The training dataset contains no samples")

    sorted_lengths = sorted(lengths)
    total_tokens = sum(lengths)
    sample_count = len(lengths)
    truncation: dict[str, dict[str, int | float]] = {}
    for limit in SEQUENCE_LENGTHS:
        truncated_count = sum(length > limit for length in lengths)
        truncated_tokens = sum(max(0, length - limit) for length in lengths)
        retained_count = sample_count - truncated_count
        truncation[str(limit)] = {
            "fully_retained_samples": retained_count,
            "fully_retained_percentage": round(retained_count * 100 / sample_count, 4),
            "truncated_samples": truncated_count,
            "truncated_tokens": truncated_tokens,
            "truncated_token_percentage": round(truncated_tokens * 100 / total_tokens, 4),
        }

    return {
        "model_path": str(MODEL_PATH),
        "dataset_path": DATASET_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "methodology": {
            "truncation": False,
            "add_special_tokens": False,
            "percentile_method": "nearest-rank",
        },
        "sample_count": sample_count,
        "total_tokens": total_tokens,
        "token_lengths": {
            "min": sorted_lengths[0],
            "mean": round(statistics.fmean(lengths), 2),
            "median": statistics.median(lengths),
            "p75": nearest_rank(sorted_lengths, 0.75),
            "p90": nearest_rank(sorted_lengths, 0.90),
            "p95": nearest_rank(sorted_lengths, 0.95),
            "p99": nearest_rank(sorted_lengths, 0.99),
            "max": sorted_lengths[-1],
        },
        "max_seq_length_analysis": truncation,
    }


def main() -> int:
    if not MODEL_PATH.is_dir():
        print(f"ERROR: Local tokenizer directory not found: {MODEL_PATH}", file=sys.stderr)
        return 1
    if not DATASET_PATH.is_file():
        print(f"ERROR: Training dataset not found: {DATASET_PATH}", file=sys.stderr)
        return 1

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
        report = build_report(load_token_lengths(tokenizer))
    except (OSError, ValueError) as error:
        print(f"ERROR: Token length analysis failed: {error}", file=sys.stderr)
        return 1

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    REPORT_PATH.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    print(f"\nReport saved to: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

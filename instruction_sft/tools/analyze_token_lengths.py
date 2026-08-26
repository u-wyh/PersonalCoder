#!/usr/bin/env python3
"""Measure Instruction SFT token lengths with the local official chat template."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT_ROOT / "instruction_sft" / "data" / "splits" / "train.jsonl"
DEFAULT_MODEL = Path("/data/PersonalCoder/model")
JSON_REPORT = PROJECT_ROOT / "instruction_sft" / "reports" / "token_lengths.json"
MD_REPORT = PROJECT_ROOT / "instruction_sft" / "reports" / "token_lengths.md"
PERCENTILES = (50, 75, 90, 95, 99)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as source:
        for line_number, raw in enumerate(source, 1):
            try:
                record = json.loads(raw)
                instruction = record["instruction"]
                response = record["response"]
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(f"invalid record at {path}:{line_number}: {exc}") from exc
            if not instruction or not response:
                raise ValueError(f"empty instruction/response at {path}:{line_number}")
            records.append(record)
    return records


def distribution(values: list[int]) -> dict[str, int | float]:
    ordered = sorted(values)
    result: dict[str, int | float] = {
        "count": len(ordered),
        "min": ordered[0],
        "mean": round(sum(ordered) / len(ordered), 2),
    }
    for percentile in PERCENTILES:
        index = max(0, math.ceil(percentile / 100 * len(ordered)) - 1)
        result[f"p{percentile}"] = ordered[index]
    result["max"] = ordered[-1]
    return result


def analyze(dataset: Path, model_path: Path) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    records = read_jsonl(dataset)
    instruction_lengths: list[int] = []
    response_lengths: list[int] = []
    total_lengths: list[int] = []
    for record in records:
        instruction = str(record["instruction"])
        response = str(record["response"])
        instruction_lengths.append(len(tokenizer.encode(instruction, add_special_tokens=False)))
        response_lengths.append(len(tokenizer.encode(response, add_special_tokens=False)))
        rendered = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": response},
            ],
            tokenize=True,
            add_generation_prompt=False,
        )
        input_ids = rendered["input_ids"] if hasattr(rendered, "keys") else rendered
        total_lengths.append(len(input_ids))
    limits = {}
    for limit in (1024, 1536, 2048, 3072, 4096):
        truncated = sum(length > limit for length in total_lengths)
        limits[str(limit)] = {
            "truncated_samples": truncated,
            "truncation_rate": round(truncated / len(total_lengths), 6),
            "coverage_rate": round(1 - truncated / len(total_lengths), 6),
        }
    return {
        "dataset": str(dataset),
        "tokenizer": str(model_path),
        "sample_count": len(records),
        "percentile_method": "nearest_rank",
        "instruction_tokens": distribution(instruction_lengths),
        "response_tokens": distribution(response_lengths),
        "total_chat_tokens": distribution(total_lengths),
        "candidate_sequence_lengths": limits,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--json-report", type=Path, default=JSON_REPORT)
    parser.add_argument("--md-report", type=Path, default=MD_REPORT)
    args = parser.parse_args()
    if not args.dataset.is_file() or not args.model_path.is_dir():
        parser.error("dataset or local model path is missing")
    report = analyze(args.dataset.resolve(), args.model_path.resolve())
    args.json_report.parent.mkdir(parents=True, exist_ok=True)
    args.json_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rows = []
    for name in ("instruction_tokens", "response_tokens", "total_chat_tokens"):
        value = report[name]
        rows.append(
            f"| {name} | {value['p50']} | {value['p75']} | {value['p90']} | "
            f"{value['p95']} | {value['p99']} | {value['max']} |"
        )
    limit_rows = [
        f"| {limit} | {value['truncated_samples']} | {value['truncation_rate']:.2%} | {value['coverage_rate']:.2%} |"
        for limit, value in report["candidate_sequence_lengths"].items()
    ]
    markdown = [
        "# Instruction SFT v1 Token Lengths",
        "",
        f"Tokenizer: `{report['tokenizer']}`; train samples: {report['sample_count']}; percentiles use nearest-rank.",
        "",
        "| Component | P50 | P75 | P90 | P95 | P99 | Max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        *rows,
        "",
        "| Max sequence length | Truncated | Truncation rate | Coverage |",
        "| ---: | ---: | ---: | ---: |",
        *limit_rows,
    ]
    args.md_report.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

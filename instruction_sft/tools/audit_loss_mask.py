#!/usr/bin/env python3
"""Audit assistant-only labels on 20 deterministic formal training samples."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INSTRUCTION_ROOT = PROJECT_ROOT / "instruction_sft"
if str(INSTRUCTION_ROOT) not in sys.path:
    sys.path.insert(0, str(INSTRUCTION_ROOT))
from train import AssistantOnlyCollator  # noqa: E402


DEFAULT_TRAIN = INSTRUCTION_ROOT / "data" / "splits" / "train.jsonl"
DEFAULT_REPORT = INSTRUCTION_ROOT / "reports" / "loss_mask_audit.md"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--model", type=Path, default=Path("/data/PersonalCoder/model"))
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.train.read_text(encoding="utf-8").splitlines() if line.strip()]
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    collator = AssistantOnlyCollator(tokenizer, 2048)
    encoded = [(record, collator.encode(record)) for record in records]
    rng = random.Random(args.seed)
    truncated = rng.sample([item for item in encoded if item[1]["truncated"]], 10)
    untruncated = rng.sample([item for item in encoded if not item[1]["truncated"]], 10)
    selected = truncated + untruncated
    rows = []
    all_valid = True
    for record, item in selected:
        prompt_masked = all(label == -100 for label in item["labels"][: item["assistant_start"]])
        assistant_active = bool(item["labels"][item["assistant_start"] :]) and all(
            label != -100 for label in item["labels"][item["assistant_start"] :]
        )
        valid = prompt_masked and assistant_active
        all_valid &= valid
        rows.append(
            {
                "id": record["id"],
                "truncated": item["truncated"],
                "input_tokens": len(item["input_ids"]),
                "assistant_start": item["assistant_start"],
                "masked_tokens": sum(label == -100 for label in item["labels"]),
                "valid_loss_tokens": sum(label != -100 for label in item["labels"]),
                "prompt_masked": prompt_masked,
                "assistant_active": assistant_active,
                "valid": valid,
            }
        )
    batch = collator([record for record, _item in selected])
    padding_mask_valid = bool((batch["labels"][batch["attention_mask"] == 0] == -100).all())
    all_valid &= padding_mask_valid
    lines = [
        "# Phase 3.5 Loss Mask Audit",
        "",
        f"Seed: {args.seed}; 10 truncated + 10 untruncated formal train samples.",
        "",
        f"Overall: **{'PASS' if all_valid else 'FAIL'}**. Prompt labels all masked: {all(row['prompt_masked'] for row in rows)}; assistant labels active: {all(row['assistant_active'] for row in rows)}; padding labels masked: {padding_mask_valid}.",
        "",
        "| ID | Truncated | Input tokens | Assistant start | Masked | Valid loss | Prompt masked | Assistant active |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    lines.extend(
        f"| {row['id']} | {row['truncated']} | {row['input_tokens']} | {row['assistant_start']} | {row['masked_tokens']} | {row['valid_loss_tokens']} | {row['prompt_masked']} | {row['assistant_active']} |"
        for row in rows
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"checked": len(rows), "truncated": 10, "untruncated": 10, "padding_mask_valid": padding_mask_valid, "all_valid": all_valid}, ensure_ascii=False, indent=2))
    return 0 if all_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())

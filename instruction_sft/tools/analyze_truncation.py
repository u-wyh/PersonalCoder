#!/usr/bin/env python3
"""Audit the exact Phase 3.3 truncation policy and a response-first simulation."""

from __future__ import annotations

import argparse
import json
import math
import os
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
DEFAULT_MODEL = Path("/data/PersonalCoder/model")
DEFAULT_JSON = INSTRUCTION_ROOT / "reports" / "truncation_analysis.json"
DEFAULT_MD = INSTRUCTION_ROOT / "reports" / "truncation_analysis.md"


def percentile(values: list[int], point: int) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(point / 100 * len(ordered)) - 1)]


def full_lengths(tokenizer: Any, record: dict[str, Any]) -> tuple[int, int, int]:
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": record["instruction"]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": record["instruction"]},
            {"role": "assistant", "content": record["response"]},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
    if not full_text.startswith(prompt_text):
        raise ValueError(f"chat prefix mismatch: {record.get('id')}")
    encoded = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    boundary = len(prompt_text)
    assistant_start = next(
        (index for index, (start, _end) in enumerate(encoded["offset_mapping"]) if start >= boundary),
        len(encoded["input_ids"]),
    )
    total = len(encoded["input_ids"])
    return assistant_start, total - assistant_start, total


def run(train_path: Path, model_path: Path, max_length: int) -> dict[str, Any]:
    records = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    collator = AssistantOnlyCollator(tokenizer, max_length)
    rows = []
    for record in records:
        prompt_tokens, response_tokens, total_tokens = full_lengths(tokenizer, record)
        encoded = collator.encode(record)
        kept_response = sum(label != -100 for label in encoded["labels"])
        kept_prompt = encoded["assistant_start"]
        response_removed = response_tokens - kept_response
        naive_kept_response = max(0, min(response_tokens, max_length - prompt_tokens))
        candidate_kept_response = min(response_tokens, max_length)
        candidate_kept_instruction = min(prompt_tokens, max(0, max_length - candidate_kept_response))
        rows.append(
            {
                "id": record["id"],
                "prompt_tokens": prompt_tokens,
                "response_tokens": response_tokens,
                "total_tokens": total_tokens,
                "current_kept_prompt": kept_prompt,
                "current_kept_response": kept_response,
                "current_response_removed": response_removed,
                "current_truncated": total_tokens > max_length,
                "current_response_truncated": response_removed > 0,
                "current_assistant_eos_truncated": response_removed > 0,
                "naive_right_kept_response": naive_kept_response,
                "naive_right_response_truncated": naive_kept_response < response_tokens,
                "naive_right_response_almost_absent": naive_kept_response <= response_tokens * 0.1,
                "candidate_kept_response": candidate_kept_response,
                "candidate_kept_instruction": candidate_kept_instruction,
                "candidate_response_complete": candidate_kept_response == response_tokens,
            }
        )
    truncated = [row for row in rows if row["current_truncated"]]
    response_cut = [row for row in rows if row["current_response_truncated"]]
    removed = [row["current_response_removed"] for row in response_cut]
    candidate_instruction = [row["candidate_kept_instruction"] for row in rows if row["current_truncated"]]
    summary = {
        "total_samples": len(rows),
        "max_seq_length": max_length,
        "actual_training_policy": "response-preserving: keep the full assistant response when response_tokens < max_seq_length; trim the instruction head/tail to fit; if response alone is too long, retain up to 256 prompt tokens and the response prefix",
        "categories": {
            "A_untruncated": sum(not row["current_truncated"] for row in rows),
            "B_instruction_only_truncated_response_complete": sum(row["current_truncated"] and not row["current_response_truncated"] for row in rows),
            "C_response_truncated": len(response_cut),
            "D_current_response_retained_at_most_10_percent": sum(row["current_kept_response"] <= row["response_tokens"] * 0.1 for row in response_cut),
            "E_assistant_eos_truncated": sum(row["current_assistant_eos_truncated"] for row in rows),
        },
        "current_policy": {
            "truncated_samples": len(truncated),
            "response_truncated_samples": len(response_cut),
            "removed_response_tokens_total": sum(removed),
            "removed_response_tokens_mean": round(sum(removed) / len(removed), 2) if removed else 0.0,
            "removed_response_tokens_p50": percentile(removed, 50),
            "removed_response_tokens_p90": percentile(removed, 90),
            "removed_response_tokens_p95": percentile(removed, 95),
            "assistant_eos_truncated": sum(row["current_assistant_eos_truncated"] for row in rows),
        },
        "naive_right_truncation_counterfactual": {
            "response_truncated_samples": sum(row["naive_right_response_truncated"] for row in rows),
            "response_almost_absent_samples": sum(row["naive_right_response_almost_absent"] for row in rows),
        },
        "response_first_simulation": {
            "responses_fully_preserved": sum(row["candidate_response_complete"] for row in rows),
            "responses_not_fully_preserved": sum(not row["candidate_response_complete"] for row in rows),
            "extreme_response_over_2048": sum(row["response_tokens"] > max_length for row in rows),
            "max_response_tokens": max(row["response_tokens"] for row in rows),
            "mean_instruction_tokens_retained_among_originally_truncated": round(sum(candidate_instruction) / len(candidate_instruction), 2),
        },
        "largest_response_truncations": sorted(
            (row for row in rows if row["current_response_truncated"]),
            key=lambda row: row["current_response_removed"],
            reverse=True,
        )[:20],
    }
    return {"summary": summary, "records": rows}


def markdown(report: dict[str, Any]) -> str:
    s = report["summary"]
    c, current, naive, candidate = s["categories"], s["current_policy"], s["naive_right_truncation_counterfactual"], s["response_first_simulation"]
    return "\n".join(
        [
            "# Phase 3.5 Truncation Analysis",
            "",
            f"Train samples: {s['total_samples']}; max sequence length: {s['max_seq_length']}.",
            "",
            "## Actual Phase 3.3 policy",
            "",
            s["actual_training_policy"],
            "",
            f"- A untruncated: {c['A_untruncated']}",
            f"- B instruction-only truncation, complete response: {c['B_instruction_only_truncated_response_complete']}",
            f"- C response truncated: {c['C_response_truncated']}",
            f"- D current response retained <=10%: {c['D_current_response_retained_at_most_10_percent']}",
            f"- E assistant EOS truncated: {c['E_assistant_eos_truncated']}",
            f"- Removed response tokens: total {current['removed_response_tokens_total']}, mean {current['removed_response_tokens_mean']}, P50/P90/P95 {current['removed_response_tokens_p50']}/{current['removed_response_tokens_p90']}/{current['removed_response_tokens_p95']}",
            "",
            "## Counterfactual ordinary right truncation",
            "",
            f"It would truncate {naive['response_truncated_samples']} responses; {naive['response_almost_absent_samples']} would retain at most 10% of the response.",
            "",
            "## Ideal response-first simulation",
            "",
            f"- Fully preserved responses: {candidate['responses_fully_preserved']} / {s['total_samples']}",
            f"- Response >2048: {candidate['extreme_response_over_2048']}; max response: {candidate['max_response_tokens']}",
            f"- Mean retained instruction tokens among originally truncated samples: {candidate['mean_instruction_tokens_retained_among_originally_truncated']}",
            "",
            "The training pipeline was already response-preserving for ordinary overlength pairs; only responses that themselves exceed the budget lose code tails/EOS.",
        ]
    ) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()
    report = run(args.train.resolve(), args.model.resolve(), args.max_seq_length)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(markdown(report), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build token-aware style chunks without changing source dataset splits."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path("/data/PersonalCoder/model")
INPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "style"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "style_chunks"
SPLITS = ("train", "validation", "test")
CHUNK_SIZE = 512
OVERLAP = 64
MIN_TAIL_TOKENS = 64
STRIDE = CHUNK_SIZE - OVERLAP


def split_token_ids(token_ids: list[int]) -> tuple[list[list[int]], int]:
    chunks: list[list[int]] = []
    start = 0
    while start < len(token_ids):
        chunk = token_ids[start:start + CHUNK_SIZE]
        if len(chunk) < MIN_TAIL_TOKENS:
            return chunks, len(chunk)
        chunks.append(chunk)
        if start + CHUNK_SIZE >= len(token_ids):
            break
        start += STRIDE
    return chunks, 0


def process_split(tokenizer: object, split: str) -> dict[str, int]:
    input_path = INPUT_ROOT / f"{split}.jsonl"
    output_path = OUTPUT_ROOT / f"{split}.jsonl"
    counters = {
        "source_files": 0,
        "chunks": 0,
        "original_tokens": 0,
        "covered_original_tokens": 0,
        "emitted_tokens": 0,
        "full_chunks": 0,
        "partial_chunks": 0,
        "discarded_short_tails": 0,
        "discarded_tail_tokens": 0,
    }

    with input_path.open(encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as output:
        for line_number, line in enumerate(source, start=1):
            try:
                sample = json.loads(line)
                source_path = sample["path"]
                source_type = sample["source_type"]
                text = sample["text"]
            except (json.JSONDecodeError, KeyError) as error:
                raise ValueError(f"Invalid {split} sample at line {line_number}: {error}") from error

            token_ids = tokenizer.encode(
                text,
                add_special_tokens=False,
                truncation=False,
            )
            chunks, discarded_tail_tokens = split_token_ids(token_ids)
            counters["source_files"] += 1
            counters["original_tokens"] += len(token_ids)
            counters["chunks"] += len(chunks)
            counters["emitted_tokens"] += sum(len(chunk) for chunk in chunks)
            counters["full_chunks"] += sum(len(chunk) == CHUNK_SIZE for chunk in chunks)
            counters["partial_chunks"] += sum(len(chunk) < CHUNK_SIZE for chunk in chunks)
            if discarded_tail_tokens:
                counters["discarded_short_tails"] += 1
                counters["discarded_tail_tokens"] += discarded_tail_tokens

            covered_tokens = len(token_ids) - discarded_tail_tokens
            counters["covered_original_tokens"] += covered_tokens
            for chunk_index, chunk in enumerate(chunks):
                record = {
                    "source_path": source_path,
                    "source_type": source_type,
                    "chunk_index": chunk_index,
                    "token_count": len(chunk),
                    "text": tokenizer.decode(
                        chunk,
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    ),
                }
                output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
                output.write("\n")
    return counters


def main() -> int:
    if not MODEL_PATH.is_dir():
        print(f"ERROR: Local tokenizer directory not found: {MODEL_PATH}", file=sys.stderr)
        return 1
    missing_inputs = [str(INPUT_ROOT / f"{split}.jsonl") for split in SPLITS if not (INPUT_ROOT / f"{split}.jsonl").is_file()]
    if missing_inputs:
        print(f"ERROR: Missing input dataset: {', '.join(missing_inputs)}", file=sys.stderr)
        return 1

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        split_stats = {split: process_split(tokenizer, split) for split in SPLITS}
    except (OSError, ValueError) as error:
        print(f"ERROR: Failed to build style chunks: {error}", file=sys.stderr)
        return 1

    totals = {
        key: sum(stats[key] for stats in split_stats.values())
        for key in next(iter(split_stats.values()))
    }
    overlap_extra_tokens = totals["emitted_tokens"] - totals["covered_original_tokens"]
    stats: dict[str, object] = {
        "model_path": str(MODEL_PATH),
        "chunk_size": CHUNK_SIZE,
        "overlap": OVERLAP,
        "minimum_tail_tokens": MIN_TAIL_TOKENS,
        "original_files": totals["source_files"],
        "total_chunks": totals["chunks"],
        "split_source_files": {
            split: split_stats[split]["source_files"] for split in SPLITS
        },
        "split_chunks": {split: split_stats[split]["chunks"] for split in SPLITS},
        "average_tokens_per_chunk": (
            round(totals["emitted_tokens"] / totals["chunks"], 2) if totals["chunks"] else 0.0
        ),
        "full_512_token_chunks": totals["full_chunks"],
        "partial_chunks": totals["partial_chunks"],
        "discarded_short_tail_count": totals["discarded_short_tails"],
        "discarded_short_tail_tokens": totals["discarded_tail_tokens"],
        "original_tokens": totals["original_tokens"],
        "covered_original_tokens": totals["covered_original_tokens"],
        "overlap_extra_tokens": overlap_extra_tokens,
        "original_token_coverage_percentage": (
            round(totals["covered_original_tokens"] * 100 / totals["original_tokens"], 6)
            if totals["original_tokens"] else 0.0
        ),
    }
    stats_path = OUTPUT_ROOT / "stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"Chunks saved to: {OUTPUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

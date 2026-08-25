#!/usr/bin/env python3
"""Build the isolated 1536-token style dataset while preserving source splits."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path("/data/PersonalCoder/model")
INPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "style"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "style_chunks_1536"
SPLITS = ("train", "validation", "test")
CHUNK_SIZE = 1536
OVERLAP = 128
MIN_TAIL_TOKENS = 128
STRIDE = CHUNK_SIZE - OVERLAP


def chunk_spans(token_count: int) -> tuple[list[tuple[int, int]], int]:
    spans: list[tuple[int, int]] = []
    start = 0
    discarded_tail = 0
    while start < token_count:
        end = min(start + CHUNK_SIZE, token_count)
        length = end - start
        if length < MIN_TAIL_TOKENS:
            discarded_tail = length
            break
        spans.append((start, end))
        if end == token_count:
            break
        start += STRIDE
    return spans, discarded_tail


def covered_token_count(spans: list[tuple[int, int]]) -> int:
    if not spans:
        return 0
    covered = 0
    current_start, current_end = spans[0]
    for start, end in spans[1:]:
        if start > current_end:
            covered += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    return covered + current_end - current_start


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
        "discarded_short_tail_candidates": 0,
        "discarded_short_tail_candidate_tokens": 0,
    }
    with input_path.open(encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as output:
        for line_number, line in enumerate(source, start=1):
            try:
                sample = json.loads(line)
                source_path = sample["path"]
                source_type = sample["source_type"]
                token_ids = tokenizer.encode(
                    sample["text"], add_special_tokens=False, truncation=False
                )
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise ValueError(f"Invalid {split} sample at line {line_number}: {error}") from error

            spans, discarded_tail = chunk_spans(len(token_ids))
            lengths = [end - start for start, end in spans]
            counters["source_files"] += 1
            counters["chunks"] += len(spans)
            counters["original_tokens"] += len(token_ids)
            counters["covered_original_tokens"] += covered_token_count(spans)
            counters["emitted_tokens"] += sum(lengths)
            counters["full_chunks"] += sum(length == CHUNK_SIZE for length in lengths)
            counters["partial_chunks"] += sum(length < CHUNK_SIZE for length in lengths)
            if discarded_tail:
                counters["discarded_short_tail_candidates"] += 1
                counters["discarded_short_tail_candidate_tokens"] += discarded_tail

            for chunk_index, (start, end) in enumerate(spans):
                record = {
                    "source_path": source_path,
                    "source_type": source_type,
                    "chunk_index": chunk_index,
                    "token_count": end - start,
                    "text": tokenizer.decode(
                        token_ids[start:end],
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    ),
                }
                output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    return counters


def main() -> int:
    if not MODEL_PATH.is_dir():
        print(f"ERROR: Local tokenizer directory not found: {MODEL_PATH}", file=sys.stderr)
        return 1
    missing = [INPUT_ROOT / f"{split}.jsonl" for split in SPLITS if not (INPUT_ROOT / f"{split}.jsonl").is_file()]
    if missing:
        print(f"ERROR: Missing inputs: {', '.join(map(str, missing))}", file=sys.stderr)
        return 1

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    split_stats = {split: process_split(tokenizer, split) for split in SPLITS}
    totals = {
        key: sum(values[key] for values in split_stats.values())
        for key in next(iter(split_stats.values()))
    }
    stats = {
        "model_path": str(MODEL_PATH),
        "chunk_size": CHUNK_SIZE,
        "overlap": OVERLAP,
        "minimum_tail_tokens": MIN_TAIL_TOKENS,
        "original_files": totals["source_files"],
        "total_chunks": totals["chunks"],
        "split_source_files": {split: split_stats[split]["source_files"] for split in SPLITS},
        "split_chunks": {split: split_stats[split]["chunks"] for split in SPLITS},
        "average_tokens_per_chunk": round(totals["emitted_tokens"] / totals["chunks"], 2),
        "full_chunks": totals["full_chunks"],
        "partial_chunks": totals["partial_chunks"],
        "discarded_short_tail_candidate_count": totals["discarded_short_tail_candidates"],
        "discarded_short_tail_candidate_tokens": totals["discarded_short_tail_candidate_tokens"],
        "original_tokens": totals["original_tokens"],
        "covered_original_tokens": totals["covered_original_tokens"],
        "overlap_extra_tokens": totals["emitted_tokens"] - totals["covered_original_tokens"],
        "original_token_coverage_percentage": round(
            totals["covered_original_tokens"] * 100 / totals["original_tokens"], 6
        ),
        "per_split": split_stats,
    }
    (OUTPUT_ROOT / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

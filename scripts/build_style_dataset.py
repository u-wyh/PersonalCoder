#!/usr/bin/env python3
"""Build a deterministic C++ style-adaptation dataset from raw source files."""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "data" / "raw" / "algorithm"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "style"
RANDOM_SEED = 42


def count_lines(content: bytes) -> int:
    if not content:
        return 0
    return content.count(b"\n") + (not content.endswith(b"\n"))


def decode_source(content: bytes) -> str:
    """Decode source losslessly using the dataset's observed common encodings."""
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return content.decode("gb18030")


def build_samples() -> tuple[list[dict[str, str]], dict[str, int | float]]:
    cpp_files = sorted(path for path in SOURCE_ROOT.rglob("*.cpp") if path.is_file())
    samples: list[dict[str, str]] = []
    seen_hashes: set[str] = set()
    empty_files = 0
    too_short_files = 0
    duplicate_files = 0

    for path in cpp_files:
        content = path.read_bytes()
        if not content:
            empty_files += 1
            continue
        if count_lines(content) < 10:
            too_short_files += 1
            continue

        digest = hashlib.sha256(content).hexdigest()
        if digest in seen_hashes:
            duplicate_files += 1
            continue
        seen_hashes.add(digest)

        relative_path = path.relative_to(SOURCE_ROOT)
        samples.append(
            {
                "path": relative_path.as_posix(),
                "source_type": "template" if "templates" in relative_path.parts else "solution",
                "sha256": digest,
                "text": decode_source(content),
            }
        )

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(samples)
    total = len(samples)
    train_count = int(total * 0.90)
    validation_count = int(total * 0.05)
    split_counts = {
        "train": train_count,
        "validation": validation_count,
        "test": total - train_count - validation_count,
    }
    template_count = sum(sample["source_type"] == "template" for sample in samples)
    total_characters = sum(len(sample["text"]) for sample in samples)
    stats: dict[str, int | float] = {
        "random_seed": RANDOM_SEED,
        "raw_cpp_files": len(cpp_files),
        "filtered_empty_files": empty_files,
        "filtered_too_short_files": too_short_files,
        "filtered_files_total": empty_files + too_short_files,
        "removed_duplicate_files": duplicate_files,
        "valid_samples": total,
        "template_samples": template_count,
        "solution_samples": total - template_count,
        "train_samples": split_counts["train"],
        "validation_samples": split_counts["validation"],
        "test_samples": split_counts["test"],
        "total_characters": total_characters,
        "average_characters": round(total_characters / total, 2) if total else 0.0,
    }
    return samples, stats


def write_jsonl(path: Path, samples: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for sample in samples:
            output.write(json.dumps(sample, ensure_ascii=False, separators=(",", ":")))
            output.write("\n")


def main() -> int:
    if not SOURCE_ROOT.is_dir():
        print(f"ERROR: Raw dataset directory not found: {SOURCE_ROOT}", file=sys.stderr)
        return 1

    try:
        samples, stats = build_samples()
    except UnicodeDecodeError as error:
        print(f"ERROR: A C++ file is not valid UTF-8: {error}", file=sys.stderr)
        return 1
    except OSError as error:
        print(f"ERROR: Failed to read the raw dataset: {error}", file=sys.stderr)
        return 1

    train_end = stats["train_samples"]
    validation_end = train_end + stats["validation_samples"]
    splits = {
        "train": samples[:train_end],
        "validation": samples[train_end:validation_end],
        "test": samples[validation_end:],
    }

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for split_name, split_samples in splits.items():
        write_jsonl(OUTPUT_ROOT / f"{split_name}.jsonl", split_samples)

    stats["jsonl_size_bytes"] = sum(
        (OUTPUT_ROOT / f"{split_name}.jsonl").stat().st_size for split_name in splits
    )
    stats_path = OUTPUT_ROOT / "stats.json"
    stats_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"Dataset saved to: {OUTPUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

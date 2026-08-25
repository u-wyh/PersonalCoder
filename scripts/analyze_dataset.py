#!/usr/bin/env python3
"""Create a read-only inventory report for the raw algorithm dataset."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "algorithm"
REPORT_PATH = PROJECT_ROOT / "outputs" / "dataset_report.json"


def line_count(content: bytes) -> int:
    """Count text lines, including a final line without a newline."""
    if not content:
        return 0
    return content.count(b"\n") + (not content.endswith(b"\n"))


def analyze_dataset() -> dict[str, object]:
    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_ROOT}")

    files = sorted(path for path in DATASET_ROOT.rglob("*") if path.is_file())
    cpp_stats: list[tuple[str, int]] = []
    hashes: dict[str, list[str]] = defaultdict(list)
    header_count = 0
    template_cpp_count = 0
    empty_file_count = 0

    for path in files:
        relative_path = path.relative_to(DATASET_ROOT).as_posix()
        suffix = path.suffix.lower()
        size = path.stat().st_size
        if size == 0:
            empty_file_count += 1
        if suffix in {".h", ".hpp"}:
            header_count += 1
        if suffix != ".cpp":
            continue

        content = path.read_bytes()
        lines = line_count(content)
        cpp_stats.append((relative_path, lines))
        hashes[hashlib.sha256(content).hexdigest()].append(relative_path)
        if "templates" in path.relative_to(DATASET_ROOT).parts:
            template_cpp_count += 1

    cpp_count = len(cpp_stats)
    total_cpp_lines = sum(lines for _, lines in cpp_stats)
    duplicate_groups = [paths for paths in hashes.values() if len(paths) > 1]
    duplicate_groups.sort(key=lambda paths: paths[0])

    shortest = min(cpp_stats, key=lambda item: (item[1], item[0]), default=None)
    longest = max(cpp_stats, key=lambda item: (item[1], item[0]), default=None)

    return {
        "dataset_root": DATASET_ROOT.relative_to(PROJECT_ROOT).as_posix(),
        "total_files": len(files),
        "cpp_files": cpp_count,
        "header_files": header_count,
        "templates_cpp_files": template_cpp_count,
        "non_templates_cpp_files": cpp_count - template_cpp_count,
        "total_cpp_lines": total_cpp_lines,
        "average_lines_per_cpp": round(total_cpp_lines / cpp_count, 2) if cpp_count else 0,
        "shortest_cpp_file": (
            {"path": shortest[0], "lines": shortest[1]} if shortest else None
        ),
        "longest_cpp_file": (
            {"path": longest[0], "lines": longest[1]} if longest else None
        ),
        "empty_files": empty_file_count,
        "cpp_files_under_10_lines": sum(lines < 10 for _, lines in cpp_stats),
        "cpp_files_over_500_lines": sum(lines > 500 for _, lines in cpp_stats),
        "duplicate_cpp_group_count": len(duplicate_groups),
        "duplicate_cpp_file_count": sum(len(paths) for paths in duplicate_groups),
        "duplicate_cpp_groups": duplicate_groups,
    }


def main() -> int:
    try:
        report = analyze_dataset()
    except (FileNotFoundError, OSError) as error:
        print(f"ERROR: {error}")
        return 1

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rendered_report = json.dumps(report, ensure_ascii=False, indent=2)
    REPORT_PATH.write_text(f"{rendered_report}\n", encoding="utf-8")
    print(rendered_report)
    print(f"\nReport saved to: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

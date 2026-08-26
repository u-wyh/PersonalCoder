#!/usr/bin/env python3
"""Discover recoverable problem IDs in the deduplicated historical C++ corpus."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from common import (
    DEFAULT_SOURCE_ROOT,
    INSTRUCTION_ROOT,
    DatasetError,
    discover_problem_id,
    git_latest_metadata,
    load_style_records,
    write_json,
    write_jsonl,
)


DEFAULT_JSON = INSTRUCTION_ROOT / "reports" / "problem_id_discovery.json"
DEFAULT_MD = INSTRUCTION_ROOT / "reports" / "problem_id_discovery.md"
DEFAULT_INDEX = INSTRUCTION_ROOT / "data" / "raw" / "code_index.jsonl"


def run(source_root: Path, json_path: Path, md_path: Path, index_path: Path) -> dict:
    records = load_style_records()
    git_index = git_latest_metadata(source_root)
    discovered = []
    for record in records:
        git_metadata = git_index.get(record["path"], {})
        identification = discover_problem_id(
            record["path"], record["text"], str(git_metadata.get("git_subject") or "")
        )
        discovered.append(
            {
                "path": record["path"],
                "source_type": record["source_type"],
                "sha256": record["sha256"],
                "style_split": record["style_split"],
                "source": identification["source"],
                "problem_id": identification["problem_id"],
                "detection_evidence": identification["evidence"],
                "candidates": identification["candidates"],
                "timestamp": record["timestamp"],
                "age_bucket": record["age_bucket"],
                "time_reliable": record["time_reliable"],
                "git_last_commit_at": git_metadata.get("git_last_commit_at"),
                "git_subject": git_metadata.get("git_subject"),
            }
        )

    identified = [record for record in discovered if record["problem_id"]]
    source_counts = Counter(record["source"] for record in discovered)
    unique_keys = {(record["source"], record["problem_id"]) for record in identified}
    problem_versions = Counter((record["source"], record["problem_id"]) for record in identified)
    duplicate_problem_records = sum(value for value in problem_versions.values() if value > 1)
    duplicate_problem_ids = sum(value > 1 for value in problem_versions.values())
    evidence_counts = Counter(record["detection_evidence"] for record in discovered)
    summary = {
        "total_codes": len(discovered),
        "identified_codes": len(identified),
        "identified_rate": round(len(identified) / len(discovered), 6) if discovered else 0.0,
        "source_distribution": {
            source: source_counts.get(source, 0)
            for source in ("luogu", "codeforces", "icpc", "unknown")
        },
        "unique_problems": len(unique_keys),
        "problem_ids_with_multiple_codes": duplicate_problem_ids,
        "codes_belonging_to_multi_code_problems": duplicate_problem_records,
        "unknown_codes": source_counts.get("unknown", 0),
        "source_type_distribution": dict(Counter(record["source_type"] for record in discovered)),
        "detection_evidence": dict(sorted(evidence_counts.items())),
        "filesystem_timestamp_available": sum(record["timestamp"] is not None for record in discovered),
        "git_timestamp_available": sum(record["git_last_commit_at"] is not None for record in discovered),
    }
    report = {
        "summary": summary,
        "identified": identified,
        "unknown_examples": [record for record in discovered if not record["problem_id"]][:100],
    }
    write_jsonl(index_path, discovered)
    write_json(json_path, report)
    lines = [
        "# Instruction SFT Problem ID Discovery",
        "",
        "This audit is local-only. IDs are accepted only from an unambiguous filename, path, header, Git subject, or explicit ICPC event path.",
        "",
        "| Metric | Count |",
        "| --- | ---: |",
        f"| Deduplicated C++ samples | {summary['total_codes']} |",
        f"| Identified samples | {summary['identified_codes']} |",
        f"| Unique problems | {summary['unique_problems']} |",
        f"| IDs with multiple code versions | {summary['problem_ids_with_multiple_codes']} |",
        f"| Unknown samples | {summary['unknown_codes']} |",
        "",
        "## Source distribution",
        "",
    ]
    for source, count in summary["source_distribution"].items():
        lines.append(f"- {source}: {count}")
    lines += [
        "",
        "## Time evidence",
        "",
        f"- SHA-matched filesystem timestamps: {summary['filesystem_timestamp_available']}",
        f"- Local Git timestamps: {summary['git_timestamp_available']}",
        "",
        "Ambiguous identifiers are retained in the index as metadata but are not promoted to formal problem mappings.",
    ]
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MD)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    args = parser.parse_args()
    try:
        report = run(
            args.source_root.resolve(), args.json.resolve(), args.markdown.resolve(), args.index.resolve()
        )
    except DatasetError as exc:
        parser.error(str(exc))
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

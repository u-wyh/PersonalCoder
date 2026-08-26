#!/usr/bin/env python3
"""Inventory eligible problem sources and freeze a deterministic 100-problem pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from common import INSTRUCTION_ROOT, DatasetError, read_jsonl, write_json, write_jsonl


SELECTED_PATH = INSTRUCTION_ROOT / "data" / "processed" / "selected_codes.jsonl"
INDEX_PATH = INSTRUCTION_ROOT / "data" / "raw" / "code_index.jsonl"
REPORT_JSON = INSTRUCTION_ROOT / "reports" / "source_inventory.json"
REPORT_MD = INSTRUCTION_ROOT / "reports" / "source_inventory.md"
PILOT_PATH = INSTRUCTION_ROOT / "data" / "statements" / "pilot_manifest.jsonl"
ALL_PATH = INSTRUCTION_ROOT / "data" / "statements" / "all_manifest.jsonl"
SOURCE_ORDER = (
    "luogu",
    "codeforces",
    "icpc",
    "atcoder",
    "nowcoder",
    "hdu",
    "poj",
    "other",
    "unknown",
)
SUPPORTED_PATTERNS = {
    "luogu": re.compile(r"^P\d{3,7}$", re.IGNORECASE),
    "codeforces": re.compile(r"^\d{1,7}[A-Z]\d?$", re.IGNORECASE),
    "atcoder": re.compile(r"^(?:ABC|ARC|AGC)\d{3}_[A-H]$", re.IGNORECASE),
    "hdu": re.compile(r"^\d{3,6}$"),
    "poj": re.compile(r"^\d{3,6}$"),
}


def normalized_source(value: str) -> str:
    key = value.strip().lower().replace("_", "")
    aliases = {
        "cf": "codeforces",
        "codeforces": "codeforces",
        "luogu": "luogu",
        "洛谷": "luogu",
        "icpc": "icpc",
        "ccpc": "icpc",
        "atcoder": "atcoder",
        "nowcoder": "nowcoder",
        "牛客": "nowcoder",
        "hdu": "hdu",
        "poj": "poj",
    }
    return aliases.get(key, value if value in SOURCE_ORDER else "other")


def uniquely_locatable(source: str, problem_id: str) -> bool:
    pattern = SUPPORTED_PATTERNS.get(source)
    return bool(pattern and pattern.fullmatch(problem_id))


def stable_key(record: dict[str, Any]) -> str:
    return hashlib.sha256(
        f"statement-pilot-v1:{record['source']}:{record['problem_id']}".encode()
    ).hexdigest()


def select_pilot(records: list[dict[str, Any]], size: int) -> list[dict[str, Any]]:
    """Cover both working resolvers: up to half CF, then fill from dominant Luogu."""
    by_source: dict[str, list[dict[str, Any]]] = {}
    for source in SOURCE_ORDER:
        by_source[source] = sorted(
            [record for record in records if record["source"] == source and record["uniquely_locatable"]],
            key=stable_key,
        )
    selected: list[dict[str, Any]] = []
    cf_quota = min(size // 2, len(by_source["codeforces"]))
    selected.extend(by_source["codeforces"][:cf_quota])
    remaining = size - len(selected)
    selected.extend(by_source["luogu"][:remaining])
    if len(selected) < size:
        used = {(item["source"], item["problem_id"]) for item in selected}
        fallback = sorted(
            [
                item
                for item in records
                if item["uniquely_locatable"]
                and (item["source"], item["problem_id"]) not in used
            ],
            key=stable_key,
        )
        selected.extend(fallback[: size - len(selected)])
    return selected


def run(selected_path: Path, pilot_size: int) -> dict[str, Any]:
    if pilot_size < 1:
        raise DatasetError("pilot size must be positive")
    raw_records = list(read_jsonl(selected_path))
    discovery_by_sha = {
        record["sha256"]: record for record in read_jsonl(INDEX_PATH)
    }
    records = []
    for record in raw_records:
        source = normalized_source(str(record.get("source") or "unknown"))
        problem_id = str(record.get("problem_id") or "")
        same_source_ids = {
            candidate.split(":", 1)[1]
            for candidate in discovery_by_sha.get(record["sha256"], {}).get("candidates", [])
            if candidate.startswith(f"{source}:")
        }
        conflicting_codeforces_ids = source == "codeforces" and len(same_source_ids) > 1
        records.append(
            {
                **record,
                "source": source,
                "uniquely_locatable": uniquely_locatable(source, problem_id)
                and not conflicting_codeforces_ids,
                "locator_error": (
                    "conflicting_codeforces_ids:" + ",".join(sorted(same_source_ids))
                    if conflicting_codeforces_ids
                    else ""
                ),
            }
        )
    if len({(record["source"], record["problem_id"]) for record in records}) != len(records):
        raise DatasetError("eligible records are not unique by problem ID")
    pilot = select_pilot(records, pilot_size)
    if len(pilot) != pilot_size:
        raise DatasetError(f"only {len(pilot)} uniquely locatable records available for pilot")
    counts = Counter(record["source"] for record in records)
    located = Counter(record["source"] for record in records if record["uniquely_locatable"])
    inventory = {
        source: {
            "eligible_problems": counts.get(source, 0),
            "uniquely_locatable": located.get(source, 0),
            "unlocatable": counts.get(source, 0) - located.get(source, 0),
        }
        for source in SOURCE_ORDER
    }
    report = {
        "eligible_total": len(records),
        "uniquely_locatable_total": sum(record["uniquely_locatable"] for record in records),
        "unlocatable_total": sum(not record["uniquely_locatable"] for record in records),
        "inventory": inventory,
        "pilot": {
            "size": len(pilot),
            "selection": "deterministic source-stratified; no difficulty/model filtering",
            "source_distribution": dict(Counter(record["source"] for record in pilot)),
        },
    }
    write_json(REPORT_JSON, report)
    write_jsonl(ALL_PATH, [record for record in records if record["uniquely_locatable"]])
    write_jsonl(PILOT_PATH, pilot)
    lines = [
        "# Eligible Problem Source Inventory",
        "",
        "| Source | Eligible | Uniquely locatable | Unlocatable |",
        "| --- | ---: | ---: | ---: |",
    ]
    for source in SOURCE_ORDER:
        item = inventory[source]
        lines.append(
            f"| {source} | {item['eligible_problems']} | {item['uniquely_locatable']} | {item['unlocatable']} |"
        )
    lines += [
        "",
        f"Total: {report['eligible_total']}; uniquely locatable: {report['uniquely_locatable_total']}; unlocatable: {report['unlocatable_total']}.",
        "",
        f"Frozen pilot: {report['pilot']['size']} problems, distribution={report['pilot']['source_distribution']}. Selection is independent of difficulty and model performance.",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected", type=Path, default=SELECTED_PATH)
    parser.add_argument("--pilot-size", type=int, default=100)
    args = parser.parse_args()
    try:
        report = run(args.selected.resolve(), args.pilot_size)
    except DatasetError as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

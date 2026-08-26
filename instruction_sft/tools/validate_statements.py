#!/usr/bin/env python3
"""Validate fetched/normalized statements before they can enter Instruction SFT."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from common import INSTRUCTION_ROOT, DatasetError, read_jsonl, write_json
from normalize_statement import NORMALIZED_ROOT, RAW_ROOT


DEFAULT_MANIFEST = INSTRUCTION_ROOT / "data" / "statements" / "pilot_manifest.jsonl"
REPORT_PATH = INSTRUCTION_ROOT / "reports" / "statement_validation.json"
MANUAL_PATH = INSTRUCTION_ROOT / "reports" / "manual_required.json"
ERROR_MARKERS = (
    "cloudflare",
    "checking your browser",
    "verify you are human",
    "captcha",
    "验证码",
    "登录后继续",
    "404 not found",
    "page not found",
)


def validate_record(expected: dict[str, Any], fetched: dict[str, Any], markdown: str) -> list[str]:
    errors = []
    if not fetched.get("success"):
        errors.append(f"fetch_failed:{fetched.get('error') or 'unknown'}")
        return errors
    problem_id = str(expected["problem_id"]).upper()
    if str(fetched.get("problem_id") or "").upper() != problem_id:
        errors.append("requested_problem_id_mismatch")
    if str(fetched.get("detected_problem_id") or "").upper() != problem_id:
        errors.append("page_problem_id_mismatch")
    for field in ("title", "statement", "input", "output"):
        if not str(fetched.get(field) or "").strip():
            errors.append(f"missing_{field}")
    if len(str(fetched.get("statement") or "").strip()) < 50:
        errors.append("statement_too_short")
    if len(markdown.strip()) < 150:
        errors.append("normalized_too_short")
    for heading in ("## 题目描述", "## 输入格式", "## 输出格式"):
        if heading not in markdown:
            errors.append(f"missing_heading:{heading}")
    lowered = (json.dumps(fetched, ensure_ascii=False) + "\n" + markdown).lower()
    if marker := next((value for value in ERROR_MARKERS if value in lowered), None):
        errors.append(f"error_page_marker:{marker}")
    if expected["source"] == "luogu" and not re.match(
        rf"^{re.escape(problem_id)}(?:\s|$)", str(fetched.get("title") or ""), re.IGNORECASE
    ):
        errors.append("luogu_title_id_mismatch")
    if expected["source"] == "codeforces":
        index = re.sub(r"^\d+", "", problem_id)
        if not re.match(rf"^{re.escape(index)}\s*\.", str(fetched.get("title") or ""), re.IGNORECASE):
            errors.append("codeforces_title_index_mismatch")
    return sorted(set(errors))


def run(
    manifest: Path,
    report_path: Path = REPORT_PATH,
    manual_path: Path = MANUAL_PATH,
) -> dict[str, Any]:
    records = []
    for expected in read_jsonl(manifest):
        source, problem_id = expected["source"], expected["problem_id"].upper()
        raw_path = RAW_ROOT / source / f"{problem_id}.json"
        normalized_path = NORMALIZED_ROOT / source / f"{problem_id}.md"
        if raw_path.is_file():
            fetched = json.loads(raw_path.read_text(encoding="utf-8"))
        else:
            fetched = {"success": False, "error": "raw record missing"}
        markdown = normalized_path.read_text(encoding="utf-8") if normalized_path.is_file() else ""
        errors = validate_record(expected, fetched, markdown)
        records.append(
            {
                "source": source,
                "problem_id": problem_id,
                "url": fetched.get("url", ""),
                "raw_path": str(raw_path),
                "normalized_path": str(normalized_path),
                "statement_length": len(str(fetched.get("statement") or "")),
                "normalized_length": len(markdown),
                "sample_count": len(fetched.get("samples") or []),
                "statement_verified": not errors,
                "errors": errors,
            }
        )
    summary = {
        "attempted": len(records),
        "fetched": sum(not any(error.startswith("fetch_failed") for error in item["errors"]) for item in records),
        "normalized": sum(Path(item["normalized_path"]).is_file() for item in records),
        "verified": sum(item["statement_verified"] for item in records),
        "failed": sum(not item["statement_verified"] for item in records),
        "manual_required": sum(not item["statement_verified"] for item in records),
        "source_distribution": {
            source: {
                "attempted": sum(item["source"] == source for item in records),
                "verified": sum(item["source"] == source and item["statement_verified"] for item in records),
            }
            for source in sorted({item["source"] for item in records})
        },
        "error_distribution": dict(
            Counter(error for item in records for error in item["errors"])
        ),
    }
    report = {"summary": summary, "records": records}
    write_json(report_path, report)
    write_json(manual_path, [item for item in records if not item["statement_verified"]])
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--manual-report", type=Path, default=MANUAL_PATH)
    args = parser.parse_args()
    try:
        report = run(
            args.manifest.resolve(), args.report.resolve(), args.manual_report.resolve()
        )
    except DatasetError as exc:
        parser.error(str(exc))
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

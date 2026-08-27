#!/usr/bin/env python3
"""Assign reproducible A/B/C/D semantic-confidence labels to Instruction-SFT pairs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VERIFICATION = ROOT / "instruction_sft/reports/sample_verification.json"
DEFAULT_REVIEWS = ROOT / "instruction_sft/reports/sample_failure_manual_review.json"
DEFAULT_OUTPUT = ROOT / "instruction_sft/reports/semantic_quality.json"

CLEAR_MANUAL_FAILURES = {
    "code_error", "problem_id_mapping_error", "statement_parse_error",
    "multiple_testcase_format",
}
CLEAR_RUNTIME_FAILURES = {
    "compile_error", "compile_timeout", "runtime_error",
    "time_limit_exceeded", "output_limit_exceeded",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verification", type=Path, default=DEFAULT_VERIFICATION)
    parser.add_argument("--manual-reviews", type=Path, default=DEFAULT_REVIEWS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    verification = json.loads(args.verification.read_text(encoding="utf-8"))
    reviews = {item["id"]: item for item in json.loads(args.manual_reviews.read_text(encoding="utf-8"))}
    labelled = []
    for record in verification["records"]:
        review = reviews.get(record["id"])
        case_statuses = {case["status"] for case in record["cases"]}
        if record["status"] == "all_samples_passed":
            label, reason = "A", "compiled and passed every cached official sample"
        elif record["status"] == "no_executable_sample":
            label, reason = "B", "compiled, but no executable official sample was available"
        elif (review and review["category"] in CLEAR_MANUAL_FAILURES) or case_statuses & CLEAR_RUNTIME_FAILURES:
            label = "D"
            reason = review["reason"] if review and review["category"] in CLEAR_MANUAL_FAILURES else f"objective execution failure: {sorted(case_statuses & CLEAR_RUNTIME_FAILURES)}"
        else:
            label = "C"
            reason = review["reason"] if review else "official-sample mismatch not manually proven to be a code/pair error"
        labelled.append({
            "id": record["id"], "source": record["source"], "semantic_confidence": label,
            "verification_status": record["status"], "reason": reason,
            "manual_category": review["category"] if review else None,
        })

    counts = Counter(item["semantic_confidence"] for item in labelled)
    total = len(labelled)
    report = {
        "definition": {
            "A": "compiled and passed all cached official samples; sample-level evidence only, not full OJ AC",
            "B": "compiled but has no executable official sample",
            "C": "sample mismatch remains uncertain or requires interactive/special checking",
            "D": "clear runtime/resource failure or manually confirmed code, mapping, parsing, or input-format error",
        },
        "method_note": "Labels are conservative confidence classes. A does not claim algorithmic correctness beyond official samples; unreviewed WA is C, not automatically D.",
        "summary": {label: {"count": counts[label], "rate": round(counts[label] / total, 6)} for label in "ABCD"},
        "total": total,
        "records": labelled,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

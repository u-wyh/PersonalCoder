#!/usr/bin/env python3
"""Audit frozen benchmark structure, metadata, checkers, and references."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

BENCHMARK_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = BENCHMARK_DIR.parent
DEFAULT_MANIFEST = BENCHMARK_DIR / "manifest.jsonl"
DEFAULT_JSON = BENCHMARK_DIR / "reports" / "benchmark_audit.json"
DEFAULT_MARKDOWN = BENCHMARK_DIR / "reports" / "benchmark_audit.md"
REQUIRED_FILES = ("statement.md", "meta.yaml", "reference.cpp")
MATCHED_META_FIELDS = ("id", "oj", "problem_id", "difficulty")

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
from benchmark.judge import (  # noqa: E402
    JudgeError,
    SUPPORTED_CHECKERS,
    judge,
    load_checker_type,
    load_limits,
    load_meta,
)


class AuditError(RuntimeError):
    """Raised when the manifest itself cannot be audited."""


def load_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise AuditError(f"manifest not found: {path}")
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not (line := raw.strip()) or line.startswith("#"):
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AuditError(f"{path}:{number}: {exc.msg}") from exc
        if not isinstance(entry, dict):
            raise AuditError(f"{path}:{number}: expected object")
        missing = [key for key in ("id", "path", *MATCHED_META_FIELDS[1:]) if not entry.get(key)]
        if missing:
            raise AuditError(f"{path}:{number}: missing {missing}")
        if entry["id"] in seen:
            raise AuditError(f"duplicate problem id: {entry['id']}")
        seen.add(entry["id"])
        entries.append(entry)
    if len(entries) != 30:
        raise AuditError(f"expected 30 manifest entries, found {len(entries)}")
    return entries


def audit_problem(entry: dict[str, Any], manifest_path: Path) -> dict[str, Any]:
    problem_id = entry["id"]
    raw_path = Path(entry["path"])
    problem_dir = raw_path if raw_path.is_absolute() else manifest_path.parent / raw_path
    errors: list[str] = []
    warnings: list[str] = []
    for name in REQUIRED_FILES:
        path = problem_dir / name
        if not path.is_file():
            errors.append(f"missing {name}")
        elif path.stat().st_size == 0:
            errors.append(f"empty {name}")
    tests_dir = problem_dir / "tests"
    inputs = sorted(tests_dir.glob("input*.txt")) if tests_dir.is_dir() else []
    outputs = sorted(tests_dir.glob("output*.txt")) if tests_dir.is_dir() else []
    input_suffixes = {path.name.removeprefix("input") for path in inputs}
    output_suffixes = {path.name.removeprefix("output") for path in outputs}
    for suffix in sorted(input_suffixes - output_suffixes):
        errors.append(f"missing output{suffix}")
    for suffix in sorted(output_suffixes - input_suffixes):
        errors.append(f"missing input{suffix}")
    for path in inputs + outputs:
        if path.stat().st_size == 0:
            errors.append(f"empty test file: {path.name}")
    if not inputs:
        errors.append("no fixed tests")

    checker_type = "invalid"
    reference_result: dict[str, Any] = {"compile": False, "ac": False, "error": "not run"}
    try:
        meta = load_meta(problem_dir)
        for field in MATCHED_META_FIELDS:
            if str(meta.get(field, "")) != str(entry.get(field, "")):
                errors.append(f"manifest/meta mismatch: {field}")
        load_limits(problem_dir)
        checker_type = load_checker_type(problem_dir)
        if checker_type not in SUPPORTED_CHECKERS:
            errors.append(f"invalid checker: {checker_type}")
    except (JudgeError, ValueError) as exc:
        errors.append(str(exc))

    reference = problem_dir / "reference.cpp"
    if reference.is_file() and inputs and input_suffixes == output_suffixes:
        try:
            reference_result = judge(reference, problem_dir)
            if not reference_result["compile"]:
                errors.append("reference compile failed")
            elif not reference_result["ac"]:
                errors.append(
                    f"reference failed fixed tests: {reference_result['passed']}/{reference_result['tests']}"
                )
        except (JudgeError, OSError) as exc:
            errors.append(f"reference judge failed: {exc}")

    return {
        "id": problem_id,
        "path": str(problem_dir),
        "status": "pass" if not errors else "fail",
        "checker": checker_type,
        "tests": len(inputs),
        "reference_compile": bool(reference_result.get("compile")),
        "reference_ac": bool(reference_result.get("ac")),
        "errors": errors,
        "warnings": warnings,
    }


def build_report(entries: list[dict[str, Any]], manifest_path: Path) -> dict[str, Any]:
    problems = [audit_problem(entry, manifest_path) for entry in entries]
    passed = sum(problem["status"] == "pass" for problem in problems)
    return {
        "summary": {
            "problems": len(problems),
            "passed": passed,
            "failed": len(problems) - passed,
            "tests": sum(problem["tests"] for problem in problems),
            "checker_counts": dict(sorted(Counter(problem["checker"] for problem in problems).items())),
        },
        "problems": problems,
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# PersonalCoder Benchmark Audit v1.1",
        "",
        f"Result: **{'PASS' if summary['failed'] == 0 else 'FAIL'}** — "
        f"{summary['passed']}/{summary['problems']} problems, {summary['tests']} fixed tests.",
        "",
        f"Checker distribution: `{json.dumps(summary['checker_counts'], ensure_ascii=False)}`.",
        "",
        "| Problem | Status | Checker | Tests | Reference Compile | Reference AC | Issues |",
        "| --- | --- | --- | ---: | --- | --- | --- |",
    ]
    for problem in report["problems"]:
        issues = "; ".join(problem["errors"] + problem["warnings"]) or "none"
        lines.append(
            f"| {problem['id']} | {problem['status'].upper()} | {problem['checker']} | "
            f"{problem['tests']} | {problem['reference_compile']} | {problem['reference_ac']} | {issues} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    try:
        manifest_path = args.manifest.resolve()
        report = build_report(load_manifest(manifest_path), manifest_path)
    except AuditError as exc:
        parser.error(str(exc))
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 1 if report["summary"]["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

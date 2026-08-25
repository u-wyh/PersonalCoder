#!/usr/bin/env python3
"""Summarize the 30-problem pilot experiment and render Markdown tables."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


BENCHMARK_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DETAILS = BENCHMARK_DIR / "reports" / "details.json"
DEFAULT_MANIFEST = BENCHMARK_DIR / "manifest.jsonl"
DEFAULT_SUMMARY = BENCHMARK_DIR / "reports" / "summary.json"
DEFAULT_MARKDOWN = BENCHMARK_DIR / "reports" / "pilot_results.md"
DEFAULT_CONTAMINATED = ("p013", "p015", "p016", "p017", "p025", "p026")
DIFFICULTIES = ("easy", "medium", "hard")
SOURCES = ("luogu", "codeforces", "icpc")


class SummaryError(RuntimeError):
    """Raised when manifest and evaluation details are inconsistent."""


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise SummaryError(f"JSON file not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SummaryError(f"invalid JSON in {path}: {exc.msg}") from exc


def load_manifest(path: str | Path) -> dict[str, dict[str, Any]]:
    manifest_path = Path(path).resolve()
    if not manifest_path.is_file():
        raise SummaryError(f"manifest not found: {manifest_path}")
    problems: dict[str, dict[str, Any]] = {}
    for line_number, raw_line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SummaryError(
                f"{manifest_path}:{line_number}: invalid JSON: {exc.msg}"
            ) from exc
        if not isinstance(value, dict):
            raise SummaryError(f"{manifest_path}:{line_number}: expected object")
        problem_id = value.get("id")
        difficulty = value.get("difficulty")
        source = value.get("oj")
        if not isinstance(problem_id, str) or not problem_id:
            raise SummaryError(f"{manifest_path}:{line_number}: missing id")
        if problem_id in problems:
            raise SummaryError(f"duplicate problem id: {problem_id}")
        if difficulty not in DIFFICULTIES:
            raise SummaryError(f"invalid difficulty for {problem_id}: {difficulty}")
        if source not in SOURCES:
            raise SummaryError(f"invalid source for {problem_id}: {source}")
        problems[problem_id] = value
    if not problems:
        raise SummaryError("manifest contains no problems")
    return problems


def load_details(
    path: str | Path, problems: dict[str, dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    details_path = Path(path).resolve()
    values = _read_json(details_path)
    if not isinstance(values, list):
        raise SummaryError(f"details must be a JSON array: {details_path}")
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for index, value in enumerate(values):
        if not isinstance(value, dict):
            raise SummaryError(f"details entry {index} must be an object")
        model = value.get("model")
        problem_id = value.get("problem")
        if not isinstance(model, str) or not model:
            raise SummaryError(f"details entry {index} has invalid model")
        if problem_id not in problems:
            raise SummaryError(f"details entry {index} has unknown problem {problem_id!r}")
        key = (model, problem_id)
        if key in seen:
            raise SummaryError(f"duplicate detail result: {model}/{problem_id}")
        seen.add(key)
        if not isinstance(value.get("compile"), bool) or not isinstance(
            value.get("ac"), bool
        ):
            raise SummaryError(f"details entry {index} has invalid compile/ac")
        by_model[model].append(value)

    expected = set(problems)
    for model, records in by_model.items():
        actual = {record["problem"] for record in records}
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise SummaryError(f"{model}: incomplete details; missing={missing}, extra={extra}")
    if not by_model:
        raise SummaryError("details contain no models")
    return dict(by_model)


def _metrics(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = list(records)
    total = len(values)
    compiled = sum(record["compile"] for record in values)
    accepted = sum(record["ac"] for record in values)
    return {
        "total": total,
        "compile": compiled,
        "compile_rate": round(compiled / total, 6) if total else 0.0,
        "ac": accepted,
        "ac_rate": round(accepted / total, 6) if total else 0.0,
    }


def build_summary(
    details: dict[str, list[dict[str, Any]]],
    problems: dict[str, dict[str, Any]],
    contaminated: set[str],
) -> dict[str, Any]:
    unknown = contaminated - set(problems)
    if unknown:
        raise SummaryError(f"unknown contaminated problem ids: {sorted(unknown)}")
    summary: dict[str, Any] = {}
    for model, records in details.items():
        full = _metrics(records)
        clean = _metrics(
            record for record in records if record["problem"] not in contaminated
        )
        difficulty = {
            level: _metrics(
                record
                for record in records
                if problems[record["problem"]]["difficulty"] == level
            )
            for level in DIFFICULTIES
        }
        source = {
            name: _metrics(
                record
                for record in records
                if problems[record["problem"]]["oj"] == name
            )
            for name in SOURCES
        }
        summary[model] = {
            "compile_rate": full["compile_rate"],
            "ac_rate": full["ac_rate"],
            "clean_ac_rate": clean["ac_rate"],
            "overall": {"full": full, "clean": clean},
            "difficulty": difficulty,
            "source": source,
            "contaminated_problems": sorted(contaminated),
        }
    return summary


def _result_cell(metrics: dict[str, Any], key: str) -> str:
    count = int(metrics[key])
    rate = float(metrics[f"{key}_rate"])
    return f"{count}/{metrics['total']} ({rate:.2%})"


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# PersonalCoder Pilot Benchmark Results",
        "",
        "## Overall",
        "",
        "| Model | Compile Rate | Offline AC Rate |",
        "| --- | ---: | ---: |",
    ]
    for model, value in summary.items():
        full = value["overall"]["full"]
        lines.append(
            f"| {model} | {_result_cell(full, 'compile')} | {_result_cell(full, 'ac')} |"
        )

    lines.extend(
        [
            "",
            "## AC by Difficulty",
            "",
            "| Model | Easy | Medium | Hard |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for model, value in summary.items():
        cells = [_result_cell(value["difficulty"][level], "ac") for level in DIFFICULTIES]
        lines.append(f"| {model} | {' | '.join(cells)} |")

    lines.extend(
        [
            "",
            "## Full vs Clean Offline AC",
            "",
            "| Model | Full AC | Clean AC |",
            "| --- | ---: | ---: |",
        ]
    )
    for model, value in summary.items():
        overall = value["overall"]
        lines.append(
            f"| {model} | {_result_cell(overall['full'], 'ac')} | "
            f"{_result_cell(overall['clean'], 'ac')} |"
        )
    lines.extend(
        [
            "",
            "Contaminated-record subset retained in Full and excluded from Clean: "
            "p013, p015, p016, p017, p025, p026.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    summary: dict[str, Any], summary_path: str | Path, markdown_path: str | Path
) -> None:
    json_path = Path(summary_path).resolve()
    md_path = Path(markdown_path).resolve()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    md_path.write_text(render_markdown(summary), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--details", type=Path, default=DEFAULT_DETAILS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument(
        "--contaminated", nargs="*", default=list(DEFAULT_CONTAMINATED)
    )
    args = parser.parse_args()
    try:
        problems = load_manifest(args.manifest)
        details = load_details(args.details, problems)
        summary = build_summary(details, problems, set(args.contaminated))
        write_outputs(summary, args.output, args.markdown_output)
    except SummaryError as exc:
        parser.error(str(exc))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

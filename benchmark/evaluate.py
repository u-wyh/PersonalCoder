#!/usr/bin/env python3
"""Evaluate generated C++ files for multiple models on a frozen manifest."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


BENCHMARK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BENCHMARK_DIR.parent
if __package__ in {None, ""}:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.judge import JudgeError, judge  # noqa: E402


@dataclass(frozen=True)
class Problem:
    problem_id: str
    directory: Path


class EvaluationError(RuntimeError):
    """Raised when evaluation inputs are invalid."""


def _validate_model_names(models: Sequence[str]) -> list[str]:
    if not models:
        raise EvaluationError("at least one model name is required")
    result: list[str] = []
    seen: set[str] = set()
    for model in models:
        if not model or model in {".", ".."} or Path(model).name != model:
            raise EvaluationError(f"invalid model directory name: {model!r}")
        if model in seen:
            raise EvaluationError(f"duplicate model name: {model}")
        seen.add(model)
        result.append(model)
    return result


def load_manifest(manifest_path: str | Path) -> list[Problem]:
    """Load JSONL entries with an id and optional manifest-relative path."""
    path = Path(manifest_path).resolve()
    if not path.is_file():
        raise EvaluationError(f"manifest not found: {path}")

    problems: list[Problem] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvaluationError(
                f"{path}:{line_number}: invalid JSON: {exc.msg}"
            ) from exc
        if not isinstance(entry, dict):
            raise EvaluationError(f"{path}:{line_number}: entry must be an object")

        problem_id = entry.get("id")
        if not isinstance(problem_id, str) or not problem_id:
            raise EvaluationError(f"{path}:{line_number}: missing string field 'id'")
        if problem_id in {".", ".."} or Path(problem_id).name != problem_id:
            raise EvaluationError(f"{path}:{line_number}: invalid id {problem_id!r}")
        if problem_id in seen:
            raise EvaluationError(f"{path}:{line_number}: duplicate id {problem_id!r}")
        seen.add(problem_id)

        raw_problem_path = entry.get(
            "path", entry.get("problem_dir", f"problems/{problem_id}")
        )
        if not isinstance(raw_problem_path, str) or not raw_problem_path:
            raise EvaluationError(
                f"{path}:{line_number}: 'path' must be a non-empty string"
            )
        problem_dir = Path(raw_problem_path)
        if not problem_dir.is_absolute():
            problem_dir = path.parent / problem_dir
        problem_dir = problem_dir.resolve()
        if not problem_dir.is_dir():
            raise EvaluationError(
                f"{path}:{line_number}: problem directory not found: {problem_dir}"
            )
        problems.append(Problem(problem_id=problem_id, directory=problem_dir))

    if not problems:
        raise EvaluationError(f"manifest contains no problems: {path}")
    return problems


def _failed_result(error: str) -> dict[str, Any]:
    return {
        "compile": False,
        "tests": 0,
        "passed": 0,
        "ac": False,
        "time": 0.0,
        "error": error,
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temp_file:
        json.dump(value, temp_file, ensure_ascii=False, indent=2)
        temp_file.write("\n")
        temp_path = Path(temp_file.name)
    os.replace(temp_path, path)


def evaluate(
    models: Sequence[str],
    manifest_path: str | Path,
    outputs_dir: str | Path = BENCHMARK_DIR / "outputs",
    reports_dir: str | Path = BENCHMARK_DIR / "reports",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Judge every model/problem pair and write aggregate and detail reports."""
    model_names = _validate_model_names(models)
    problems = load_manifest(manifest_path)
    generated_root = Path(outputs_dir).resolve()
    report_root = Path(reports_dir).resolve()
    generated_root.mkdir(parents=True, exist_ok=True)

    details: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for model in model_names:
        model_results: list[dict[str, Any]] = []
        for problem in problems:
            source_path = generated_root / model / f"{problem.problem_id}.cpp"
            try:
                result = judge(source_path, problem.directory)
            except (JudgeError, OSError) as exc:
                result = _failed_result(str(exc))

            detail = {
                "model": model,
                "problem": problem.problem_id,
                "compile": bool(result["compile"]),
                "ac": bool(result["ac"]),
                "time": float(result["time"]),
                "tests": int(result["tests"]),
                "passed": int(result["passed"]),
                "error": str(result["error"]),
            }
            details.append(detail)
            model_results.append(detail)

        total = len(model_results)
        compile_count = sum(item["compile"] for item in model_results)
        ac_count = sum(item["ac"] for item in model_results)
        summaries[model] = {
            "total": total,
            "compile": compile_count,
            "compile_rate": round(compile_count / total, 6),
            "ac": ac_count,
            "ac_rate": round(ac_count / total, 6),
        }

    summary = {"models": summaries}
    _write_json(report_root / "result.json", summary)
    _write_json(report_root / "details.json", details)
    return summary, details


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True, help="model names")
    parser.add_argument("--manifest", type=Path, required=True, help="JSONL manifest")
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=BENCHMARK_DIR / "outputs",
        help="directory containing <model>/<problem>.cpp",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=BENCHMARK_DIR / "reports",
        help="directory for result.json and details.json",
    )
    args = parser.parse_args()

    try:
        summary, _ = evaluate(
            models=args.models,
            manifest_path=args.manifest,
            outputs_dir=args.outputs_dir,
            reports_dir=args.reports_dir,
        )
    except EvaluationError as exc:
        parser.error(str(exc))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Compare simple C++ style features across pilot model outputs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

BENCHMARK_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DETAILS = BENCHMARK_DIR / "reports" / "details.json"
DEFAULT_MANIFEST = BENCHMARK_DIR / "manifest.jsonl"
DEFAULT_OUTPUTS = BENCHMARK_DIR / "outputs"
DEFAULT_REPORT = BENCHMARK_DIR / "reports" / "style_benchmark.json"
MODEL_ORDER = ("Base", "LoRA-512", "LoRA-1536")


class StyleError(RuntimeError):
    """Raised when style-analysis inputs are missing or inconsistent."""


FEATURES: dict[str, tuple[str, re.Pattern[str]]] = {
    "using_namespace_std": ("using namespace std", re.compile(r"\busing\s+namespace\s+std\s*;")),
    "bits_stdcpp": ("#include<bits/stdc++.h> with optional whitespace", re.compile(r"#\s*include\s*<\s*bits/stdc\+\+\.h\s*>", re.I)),
    "maxn_maxm_constant": ("MAXN/MAXM/MAX_* style identifier", re.compile(r"\b(?:MAXN|MAXM|MAX_[A-Za-z0-9_]+)\b")),
    "static_array": ("C-style fixed-size array declaration", re.compile(r"(?m)^\s*(?:static\s+)?(?:const\s+)?(?:unsigned\s+|signed\s+)?(?:long\s+long|int|char|bool|double|float|short|size_t|[A-Z]\w*)\s+\w+\s*(?:\[[^\]\n]+\])+\s*(?:[;=,])")),
    "vector": ("std::vector or vector", re.compile(r"\b(?:std\s*::\s*)?vector\s*<")),
    "ios_sync_with_stdio": ("ios::sync_with_stdio call", re.compile(r"\b(?:std\s*::\s*)?ios(?:_base)?\s*::\s*sync_with_stdio\s*\(")),
    "cin_tie": ("cin.tie call", re.compile(r"\b(?:std\s*::\s*)?cin\s*\.\s*tie\s*\(")),
    "long_long": ("long long type", re.compile(r"\blong\s+long\b")),
    "define_int_long_long": ("#define int long long", re.compile(r"(?m)^\s*#\s*define\s+int\s+long\s+long\b")),
}


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise StyleError(f"JSON file not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise StyleError(f"invalid JSON in {path}: {exc.msg}") from exc


def load_problem_ids(path: Path) -> list[str]:
    if not path.is_file():
        raise StyleError(f"manifest not found: {path}")
    result: list[str] = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not (line := raw.strip()) or line.startswith("#"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise StyleError(f"{path}:{number}: {exc.msg}") from exc
        problem_id = value.get("id") if isinstance(value, dict) else None
        if not isinstance(problem_id, str) or not problem_id:
            raise StyleError(f"{path}:{number}: missing problem id")
        result.append(problem_id)
    if len(result) != len(set(result)):
        raise StyleError("manifest contains duplicate problem ids")
    return result


def load_details(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    values = load_json(path)
    if not isinstance(values, list):
        raise StyleError("details.json must contain an array")
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for index, item in enumerate(values):
        if not isinstance(item, dict):
            raise StyleError(f"details entry {index} is not an object")
        model, problem = item.get("model"), item.get("problem")
        if not isinstance(model, str) or not isinstance(problem, str):
            raise StyleError(f"details entry {index} has invalid model/problem")
        if (model, problem) in result:
            raise StyleError(f"duplicate result: {model}/{problem}")
        result[(model, problem)] = item
    return result


def scope_metrics(samples: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(samples)
    metrics: dict[str, Any] = {}
    for name in FEATURES:
        count = sum(sample["features"][name] for sample in samples)
        metrics[name] = {"count": count, "rate": round(count / total, 6) if total else 0.0}
    return {"samples": total, "features": metrics}


def analyze(details_path: Path, manifest_path: Path, outputs_dir: Path) -> dict[str, Any]:
    problem_ids = load_problem_ids(manifest_path)
    details = load_details(details_path)
    models = [model for model in MODEL_ORDER if any(key[0] == model for key in details)]
    models.extend(sorted({key[0] for key in details} - set(models)))
    report: dict[str, Any] = {
        "metadata": {
            "details": str(details_path), "manifest": str(manifest_path), "outputs": str(outputs_dir),
            "scope_definitions": {"Style-All": "all generated submissions", "Style-Compiled": "submissions with compile=true in details.json", "Style-AC": "submissions with ac=true in details.json"},
            "feature_definitions": {name: value[0] for name, value in FEATURES.items()},
        },
        "models": {},
    }
    for model in models:
        samples: list[dict[str, Any]] = []
        for problem_id in problem_ids:
            detail = details.get((model, problem_id))
            if detail is None:
                raise StyleError(f"missing detail result: {model}/{problem_id}")
            source = outputs_dir / model / f"{problem_id}.cpp"
            if not source.is_file():
                raise StyleError(f"generated source not found: {source}")
            code = source.read_text(encoding="utf-8", errors="replace")
            samples.append({
                "problem": problem_id, "compile": bool(detail.get("compile")), "ac": bool(detail.get("ac")),
                "features": {name: bool(pattern.search(code)) for name, (_, pattern) in FEATURES.items()},
            })
        report["models"][model] = {
            "Style-All": scope_metrics(samples),
            "Style-Compiled": scope_metrics([sample for sample in samples if sample["compile"]]),
            "Style-AC": scope_metrics([sample for sample in samples if sample["ac"]]),
            "per_problem": samples,
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--details", type=Path, default=DEFAULT_DETAILS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--outputs", type=Path, default=DEFAULT_OUTPUTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    try:
        report = analyze(args.details.resolve(), args.manifest.resolve(), args.outputs.resolve())
    except StyleError as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    compact = {model: {scope: values[scope]["features"] for scope in ("Style-All", "Style-Compiled", "Style-AC")} for model, values in report["models"].items()}
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

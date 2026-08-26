#!/usr/bin/env python3
"""Re-judge pilot outputs to classify failures and compare models per problem."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

BENCHMARK_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = BENCHMARK_DIR.parent
DEFAULT_DETAILS = BENCHMARK_DIR / "reports" / "details.json"
DEFAULT_MANIFEST = BENCHMARK_DIR / "manifest.jsonl"
DEFAULT_OUTPUTS = BENCHMARK_DIR / "outputs"
DEFAULT_REPORT = BENCHMARK_DIR / "reports" / "failure_analysis.json"
DEFAULT_COMPARISON = BENCHMARK_DIR / "reports" / "comparison.md"
MODEL_ORDER = ("Base", "LoRA-512", "LoRA-1536")

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
from benchmark.judge import JudgeError, judge  # noqa: E402


class AnalysisError(RuntimeError):
    """Raised when benchmark artifacts are missing or inconsistent."""


COMPILE_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("template_error", (r"template argument", r"template-id", r"no matching function.*template", r"type/value mismatch at argument")),
    ("missing_symbol", (r"undefined reference", r"is not a member of", r"incomplete type", r"cannot find -l", r"ld returned \d+ exit status")),
    ("undeclared_identifier", (r"was not declared in this scope", r"undeclared identifier", r"has not been declared")),
    ("type_error", (r"invalid conversion", r"cannot convert", r"no match for .?operator", r"incompatible type", r"invalid operands", r"reference to .+ is ambiguous", r"has no member named")),
    ("syntax_error", (r"missing terminating", r"stray .+ in program", r"expected .+ before", r"expected .+ at end", r"expected declaration", r"does not name a type", r"redefinition of")),
)

CODE_FINDINGS = (
    ("p002", "Base 与 LoRA-1536 都把条件误写为可被 4 整除；Base 仅因固定测试未暴露反例且带换行而得到 Offline AC。LoRA-1536 还缺少末尾换行，在严格字节 diff 下 WA；LoRA-512 对 sqrt(w) 使用取模，直接 CE。"),
    ("p005", "Base 调用 std::accumulate 却未包含 <numeric>，CE；两个 LoRA 都改用显式三项求和并 AC。"),
    ("p008", "Base 输出了自测函数和错误断言，未读取题目输入并运行时中止；两个 LoRA 都生成了正确的输入、循环和输出。"),
    ("p012", "Base 与 LoRA-1536 使用 int，并把两列分别排序后再求差，改变了输入配对；LoRA-512 按 EOF 流式读取 long long，唯一 AC。"),
    ("p021", "Base 每次插入后排序并立即输出，固定测试全部通过；LoRA-512 读完后插入额外的 0，LoRA-1536 只整体排序一次且偶数分支可能访问 i+1 越界，二者均 WA。"),
)


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise AnalysisError(f"JSON file not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AnalysisError(f"invalid JSON in {path}: {exc.msg}") from exc


def load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise AnalysisError(f"manifest not found: {path}")
    problems: dict[str, dict[str, Any]] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not (line := raw.strip()) or line.startswith("#"):
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AnalysisError(f"{path}:{number}: {exc.msg}") from exc
        problem_id = item.get("id") if isinstance(item, dict) else None
        if not isinstance(problem_id, str) or not problem_id:
            raise AnalysisError(f"{path}:{number}: missing problem id")
        if problem_id in problems:
            raise AnalysisError(f"duplicate problem id: {problem_id}")
        problems[problem_id] = item
    if not problems:
        raise AnalysisError("manifest contains no problems")
    return problems


def load_details(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    values = read_json(path)
    if not isinstance(values, list):
        raise AnalysisError("details.json must contain an array")
    details: dict[tuple[str, str], dict[str, Any]] = {}
    for index, item in enumerate(values):
        if not isinstance(item, dict):
            raise AnalysisError(f"details entry {index} is not an object")
        model, problem = item.get("model"), item.get("problem")
        if not isinstance(model, str) or not isinstance(problem, str):
            raise AnalysisError(f"details entry {index} has invalid model/problem")
        if (model, problem) in details:
            raise AnalysisError(f"duplicate result: {model}/{problem}")
        details[(model, problem)] = item
    return details


def classify_compile_error(stderr: str) -> str:
    text = stderr.lower()
    for category, patterns in COMPILE_PATTERNS:
        if any(re.search(pattern, text, re.DOTALL) for pattern in patterns):
            return category
    return "other"


def classify_result(result: dict[str, Any], baseline_error: str = "") -> str:
    if result["ac"]:
        return "accepted"
    if not result["compile"]:
        return "compile_error"
    original = baseline_error.lower()
    if "output limit" in original or "size limit exceeded" in original:
        return "output_limit_exceeded"
    if "time limit exceeded" in original:
        return "time_limit_exceeded"
    runtime_markers = ("process exited", "assert", "double free", "corruption", "segmentation", "invalid pointer", "core dumped")
    if any(marker in original for marker in runtime_markers):
        return "runtime_error"
    statuses = {case.get("status") for case in result.get("cases", [])}
    for status in ("output_limit_exceeded", "time_limit_exceeded", "runtime_error", "wrong_answer"):
        if status in statuses:
            return status
    return "wrong_answer"


def problem_dir(manifest: Path, item: dict[str, Any]) -> Path:
    raw = item.get("path")
    if not isinstance(raw, str) or not raw:
        raise AnalysisError(f"missing path for {item.get('id')}")
    path = Path(raw)
    return path if path.is_absolute() else manifest.parent / path


def analyze(details_path: Path, manifest_path: Path, outputs_dir: Path) -> tuple[dict[str, Any], dict[str, dict[str, str]], dict[str, dict[str, Any]]]:
    problems = load_manifest(manifest_path)
    original = load_details(details_path)
    models = [model for model in MODEL_ORDER if any(key[0] == model for key in original)]
    models.extend(sorted({key[0] for key in original} - set(models)))
    expected = {(model, problem) for model in models for problem in problems}
    if set(original) != expected:
        raise AnalysisError(f"incomplete details; missing={sorted(expected-set(original))}, extra={sorted(set(original)-expected)}")
    report: dict[str, Any] = {
        "metadata": {"details": str(details_path), "manifest": str(manifest_path), "outputs": str(outputs_dir), "models": models, "problems": len(problems), "classification_unit": "one submission; testcase statuses retained separately", "wa_category_default": "unknown"},
        "models": {},
    }
    matrix = {problem: {} for problem in problems}
    for model in models:
        status_counts: Counter[str] = Counter()
        case_counts: Counter[str] = Counter()
        compile_counts: Counter[str] = Counter()
        records: list[dict[str, Any]] = []
        for problem_id, problem in problems.items():
            source = outputs_dir / model / f"{problem_id}.cpp"
            if not source.is_file():
                raise AnalysisError(f"generated source not found: {source}")
            try:
                rerun = judge(source, problem_dir(manifest_path, problem))
            except JudgeError as exc:
                raise AnalysisError(f"judge failed for {model}/{problem_id}: {exc}") from exc
            baseline = original[(model, problem_id)]
            if (rerun["compile"], rerun["ac"]) != (baseline.get("compile"), baseline.get("ac")):
                raise AnalysisError(f"re-judge mismatch for {model}/{problem_id}")
            status = classify_result(rerun, str(baseline.get("error", "")))
            status_counts[status] += 1
            case_counts.update(case.get("status", "unknown") for case in rerun["cases"])
            compile_category = classify_compile_error(rerun["error"]) if status == "compile_error" else None
            if compile_category:
                compile_counts[compile_category] += 1
            records.append({
                "problem": problem_id, "difficulty": problem.get("difficulty"), "status": status,
                "compile": rerun["compile"], "ac": rerun["ac"], "tests": rerun["tests"], "passed": rerun["passed"],
                "compile_error_category": compile_category, "wa_category": "unknown" if status == "wrong_answer" else None,
                "error": rerun["error"], "case_status_counts": dict(sorted(Counter(case["status"] for case in rerun["cases"]).items())),
            })
            matrix[problem_id][model] = status
        report["models"][model] = {"total": len(records), "failure_distribution": dict(sorted(status_counts.items())), "compile_error_distribution": dict(sorted(compile_counts.items())), "testcase_status_distribution": dict(sorted(case_counts.items())), "problems": records}
    return report, matrix, problems


def label(status: str) -> str:
    return {"accepted": "AC", "compile_error": "CE", "runtime_error": "RE", "time_limit_exceeded": "TLE", "output_limit_exceeded": "OLE", "wrong_answer": "WA"}.get(status, status)


def render_comparison(matrix: dict[str, dict[str, str]], problems: dict[str, dict[str, Any]]) -> str:
    models = [model for model in MODEL_ORDER if any(model in row for row in matrix.values())]
    lines = ["# Pilot Per-problem Comparison", "", "状态来自对已有生成代码按原 Judge 配置的复评；AC 指固定离线测试集通过。", "", f"| Problem | Difficulty | {' | '.join(models)} |", f"| --- | --- | {' | '.join('---:' for _ in models)} |"]
    for problem_id, row in matrix.items():
        lines.append(f"| {problem_id} | {problems[problem_id].get('difficulty')} | {' | '.join(label(row[m]) for m in models)} |")
    base_to_lora = [p for p, row in matrix.items() if row.get("Base") != "accepted" and any(row.get(m) == "accepted" for m in models if m != "Base")]
    base_to_fail = [p for p, row in matrix.items() if row.get("Base") == "accepted" and any(row.get(m) != "accepted" for m in models if m != "Base")]
    all_fail = [p for p, row in matrix.items() if all(value != "accepted" for value in row.values())]
    lora_diff = [p for p, row in matrix.items() if row.get("LoRA-512") != row.get("LoRA-1536")]
    lines += ["", "## 重点结果组", "", f"- Base FAIL → 至少一个 LoRA AC：{', '.join(base_to_lora) or 'none'}", f"- Base AC → 至少一个 LoRA FAIL：{', '.join(base_to_fail) or 'none'}", f"- 三模型全部 FAIL（{len(all_fail)}）：{', '.join(all_fail) or 'none'}", f"- LoRA-512 与 LoRA-1536 状态不同：{', '.join(lora_diff) or 'none'}", "", "## 关键代码差异核查", ""]
    lines.extend(f"- **{problem}**：{finding}" for problem, finding in CODE_FINDINGS)
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--details", type=Path, default=DEFAULT_DETAILS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--outputs", type=Path, default=DEFAULT_OUTPUTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--comparison-output", type=Path, default=DEFAULT_COMPARISON)
    args = parser.parse_args()
    try:
        report, matrix, problems = analyze(args.details.resolve(), args.manifest.resolve(), args.outputs.resolve())
    except AnalysisError as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.comparison_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.comparison_output.write_text(render_comparison(matrix, problems), encoding="utf-8")
    print(json.dumps({m: v["failure_distribution"] for m, v in report["models"].items()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

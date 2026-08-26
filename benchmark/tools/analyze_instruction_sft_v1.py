#!/usr/bin/env python3
"""Build the frozen Phase 3.4 Instruction-SFT-v1 benchmark report."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_ROOT = BENCHMARK_ROOT / "reports" / "instruction_sft_v1"
DEFAULT_OUTPUTS = BENCHMARK_ROOT / "ablation" / "prompt" / "P0"
DEFAULT_RAW = DEFAULT_OUTPUTS / "raw"
MODELS = ("Base", "LoRA-512", "LoRA-1536", "Instruction-SFT-v1")
OLD_MODELS = MODELS[:3]
FENCE = "`" * 3
COMPLETE_FENCE = re.compile(r"```[^\r\n`]*\r?\n.*?```", re.DOTALL)
CODE_MARKER = re.compile(r"#\s*include|\b(?:int|signed)\s+main\s*\(")
STATUS_LABEL = {
    "accepted": "AC",
    "compile_error": "CE",
    "runtime_error": "RE",
    "time_limit_exceeded": "TLE",
    "output_limit_exceeded": "OLE",
    "wrong_answer": "WA",
}
CODE_FINDINGS = {
    "p005": "Instruction 将 n 题误读为固定三组各三个数，未按 n 行逐题计数；因此仅过 2/5。两个 Style LoRA 使用逐行三数求和并 AC。",
    "p008": "Instruction 正确读取 n,k 并逐次执行末位减一/除十，修复 Base 运行自测断言且不读取输入的问题；三个 LoRA 均 AC。",
    "p012": "Instruction 延续 Base 的错误：把两列分别排序后配对，且使用 int 处理 1e18；仅 LoRA-512 按 EOF 使用 long long 原对计算并 AC。",
    "p014": "三个旧模型分别因类成员缺失、rank 名称歧义或数组 rank 冲突而 CE；Instruction 采用 MAXN 静态父数组与路径压缩，4/4 AC，是唯一旧模型全失败后的新增 AC。",
    "p019": "Instruction 用 ans 记录当前非递减段长度，下降时直接重置，却未保存历史最大值；Base 与两个 Style LoRA 均维护 maxLength 并 AC。",
    "p011": "Instruction 调用不存在于 C++17 标准库的 std::split；这是 API/符号幻觉，不是单纯缺少头文件。",
    "p017": "Instruction 直接输出 vector<int>，触发 operator<< 模板类型错误。",
    "p030": "Instruction 命中 1024-token 上限，输出在 bfs 函数中截断，造成缺分号/右花括号的语法错误。",
}
CE_CATEGORIES = (
    "missing_include",
    "undeclared_identifier",
    "syntax_error",
    "type_error",
    "template_error",
    "other",
)
MISSING_INCLUDE_SYMBOLS = {
    "accumulate": "numeric",
    "reverse": "algorithm",
    "__gcd": "algorithm",
    "setprecision": "iomanip",
    "int_max": "climits",
    "llong_max": "climits",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_manifest(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def failure_index(failure: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (model, item["problem"]): item
        for model, values in failure["models"].items()
        for item in values["problems"]
    }


def normalize_compile_categories(failure: dict[str, Any], outputs_root: Path) -> None:
    """Add the Phase 3.4 requested CE vocabulary, including explicit zeros."""
    for model, values in failure["models"].items():
        counts = {category: 0 for category in CE_CATEGORIES}
        for item in values["problems"]:
            if item["status"] != "compile_error":
                item["compile_error_category_phase3_4"] = None
                continue
            error = item["error"].lower()
            code = (outputs_root / model / f"{item['problem']}.cpp").read_text(
                encoding="utf-8", errors="replace"
            ).lower()
            missing_include = any(
                symbol in error
                and f"#include <{header}>" not in code
                and "bits/stdc++.h" not in code
                for symbol, header in MISSING_INCLUDE_SYMBOLS.items()
            )
            original = item.get("compile_error_category")
            if missing_include:
                category = "missing_include"
            elif original == "missing_symbol":
                category = "undeclared_identifier"
            elif original in CE_CATEGORIES:
                category = original
            else:
                category = "other"
            item["compile_error_category_phase3_4"] = category
            counts[category] += 1
        values["compile_error_distribution_phase3_4"] = counts


def instruction_following(raw_root: Path, outputs_root: Path, problems: list[str]) -> dict[str, Any]:
    report: dict[str, Any] = {
        "definitions": {
            "pure_code": "raw output is unfenced C++ with no surrounding prose",
            "markdown_code_fence": "raw output uses a Markdown code fence, including an unclosed fence caused by truncation",
            "extra_natural_language": "non-code prose exists outside the code block",
            "unextractable": "saved extraction is empty or has no recognizable C++ include/main marker",
            "strict_code_only": "pure_code=true; fenced output does not satisfy the literal P0 code-only instruction",
        },
        "models": {},
    }
    for model in MODELS:
        per_problem = []
        for problem in problems:
            raw = (raw_root / model / f"{problem}.txt").read_text(encoding="utf-8", errors="replace").strip()
            code = (outputs_root / model / f"{problem}.cpp").read_text(encoding="utf-8", errors="replace").strip()
            fenced = FENCE in raw
            complete_blocks = list(COMPLETE_FENCE.finditer(raw))
            if complete_blocks:
                outside = COMPLETE_FENCE.sub("", raw).strip()
                extra = bool(outside)
            elif fenced:
                # An unclosed fence followed by recognizable C++ is truncation,
                # not natural-language explanation.
                after_first_line = raw.split("\n", 1)[1] if "\n" in raw else ""
                extra = not bool(CODE_MARKER.search(after_first_line))
            else:
                extra = bool(raw != code and raw.removesuffix(code).strip())
            extractable = bool(code and CODE_MARKER.search(code))
            pure = bool(not fenced and not extra and extractable)
            primary = (
                "unextractable"
                if not extractable
                else "extra_natural_language"
                if extra
                else "markdown_code_fence"
                if fenced
                else "pure_code"
            )
            per_problem.append(
                {
                    "problem": problem,
                    "primary_category": primary,
                    "pure_code": pure,
                    "markdown_code_fence": fenced,
                    "extra_natural_language": extra,
                    "extraction_success": extractable,
                    "strict_code_only": pure,
                }
            )
        report["models"][model] = {
            "total": len(per_problem),
            "pure_code": sum(item["pure_code"] for item in per_problem),
            "markdown_code_fence": sum(item["markdown_code_fence"] for item in per_problem),
            "extra_natural_language": sum(item["extra_natural_language"] for item in per_problem),
            "extraction_success": sum(item["extraction_success"] for item in per_problem),
            "unextractable": sum(not item["extraction_success"] for item in per_problem),
            "strict_code_only": sum(item["strict_code_only"] for item in per_problem),
            "per_problem": per_problem,
        }
    return report


def status_matrix(index: dict[tuple[str, str], dict[str, Any]], problems: list[str]) -> dict[str, dict[str, str]]:
    return {
        problem: {model: index[model, problem]["status"] for model in MODELS}
        for problem in problems
    }


def transition(matrix: dict[str, dict[str, str]], baseline: str) -> dict[str, list[str] | int]:
    gained = [p for p, row in matrix.items() if row[baseline] != "accepted" and row["Instruction-SFT-v1"] == "accepted"]
    lost = [p for p, row in matrix.items() if row[baseline] == "accepted" and row["Instruction-SFT-v1"] != "accepted"]
    return {"gained_ac": gained, "lost_ac": lost, "net_ac": len(gained) - len(lost)}


def render_comparison(
    matrix: dict[str, dict[str, str]], manifest: list[dict[str, Any]], paired: dict[str, Any]
) -> str:
    lines = [
        "# Instruction SFT v1 Paired Comparison",
        "",
        "同一 audited 30题、P0、生成参数与 Judge 的逐题比较。",
        "",
        "| Problem | Difficulty | Base | LoRA-512 | LoRA-1536 | Instruction-SFT-v1 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for item in manifest:
        row = matrix[item["id"]]
        lines.append(
            f"| {item['id']} | {item['difficulty']} | "
            + " | ".join(STATUS_LABEL[row[model]] for model in MODELS)
            + " |"
        )
    lines += ["", "## Required paired sets", ""]
    for baseline in OLD_MODELS:
        values = paired[baseline]
        lines.append(f"- {baseline} FAIL → Instruction AC：{', '.join(values['gained_ac']) or 'none'}")
        lines.append(f"- {baseline} AC → Instruction FAIL：{', '.join(values['lost_ac']) or 'none'}")
    lines += [
        f"- 三种旧模型全部 FAIL → Instruction AC：{', '.join(paired['all_old_fail_instruction_ac']) or 'none'}",
        f"- 四模型全部 FAIL（{len(paired['all_four_fail'])}）：{', '.join(paired['all_four_fail']) or 'none'}",
        "",
        "## Actual code findings",
        "",
    ]
    lines.extend(f"- **{problem}**：{finding}" for problem, finding in CODE_FINDINGS.items())
    return "\n".join(lines) + "\n"


def counts_table(summary: dict[str, Any]) -> list[str]:
    lines = [
        "| Model | Compile | Compile Rate | Offline AC | AC Rate |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for model in MODELS:
        value = summary[model]["overall"]["full"]
        lines.append(
            f"| {model} | {value['compile']}/30 | {value['compile_rate']:.2%} | {value['ac']}/30 | {value['ac_rate']:.2%} |"
        )
    return lines


def difficulty_table(summary: dict[str, Any]) -> list[str]:
    lines = ["| Model | Easy AC | Medium AC | Hard AC |", "| --- | ---: | ---: | ---: |"]
    for model in MODELS:
        d = summary[model]["difficulty"]
        lines.append(f"| {model} | {d['easy']['ac']}/12 | {d['medium']['ac']}/12 | {d['hard']['ac']}/6 |")
    return lines


def render_report(
    summary: dict[str, Any], failure: dict[str, Any], style: dict[str, Any], following: dict[str, Any], paired: dict[str, Any], generation: dict[str, Any]
) -> str:
    instr_fail = failure["models"]["Instruction-SFT-v1"]["failure_distribution"]
    compile_categories = failure["models"]["Instruction-SFT-v1"]["compile_error_distribution_phase3_4"]
    features = style["models"]["Instruction-SFT-v1"]["Style-All"]["features"]
    base_features = style["models"]["Base"]["Style-All"]["features"]
    source = summary["Instruction-SFT-v1"]["source"]
    lines = [
        "# Phase 3.4 Instruction-SFT-v1 Benchmark",
        "",
        "Frozen audited 30-problem / 138-test benchmark; P0; identical tokenizer, chat template, quantization, greedy generation (`max_new_tokens=1024`) and Judge. Old P0 outputs are reused.",
        "",
        "## Overall",
        "",
        *counts_table(summary),
        "",
        "## Difficulty",
        "",
        *difficulty_table(summary),
        "",
        "## Source AC",
        "",
        f"Instruction-SFT-v1: Luogu {source['luogu']['ac']}/10, Codeforces {source['codeforces']['ac']}/10, ICPC {source['icpc']['ac']}/10.",
        "",
        "## Failure diagnosis",
        "",
        f"Instruction-SFT-v1 submission status: AC {instr_fail.get('accepted', 0)}, CE {instr_fail.get('compile_error', 0)}, RE {instr_fail.get('runtime_error', 0)}, WA {instr_fail.get('wrong_answer', 0)}, TLE {instr_fail.get('time_limit_exceeded', 0)}, OLE {instr_fail.get('output_limit_exceeded', 0)}.",
        f"CE categories: `{json.dumps(compile_categories, ensure_ascii=False)}`.",
        "The smoke missing-header issue is not systematic: none of the 30 formal outputs fails solely from a missing include. The three CEs are nonexistent `std::split` (p011), streaming a vector (p017), and a 1024-token truncation (p030).",
        "",
        "## Paired AC",
        "",
        f"- Base → Instruction: gained {paired['Base']['gained_ac']}, lost {paired['Base']['lost_ac']}, net {paired['Base']['net_ac']:+d}.",
        f"- LoRA-512 → Instruction: gained {paired['LoRA-512']['gained_ac']}, lost {paired['LoRA-512']['lost_ac']}, net {paired['LoRA-512']['net_ac']:+d}.",
        f"- LoRA-1536 → Instruction: gained {paired['LoRA-1536']['gained_ac']}, lost {paired['LoRA-1536']['lost_ac']}, net {paired['LoRA-1536']['net_ac']:+d}.",
        "- The unique all-old-fail recovery is p014; p019 is the only Base AC regression.",
        "",
        "## Instruction following",
        "",
        f"All four models use Markdown fences for 30/30 outputs; strict unfenced code-only is 0/30, extra prose is 0/30, and extraction succeeds 30/30. Instruction-SFT-v1 therefore does not improve literal P0 code-only adherence. One Instruction output (p030) reaches the 1024-token cap.",
        "",
        "## Style side effect",
        "",
        f"Instruction-SFT-v1 partially learns personal style: `using namespace std` {features['using_namespace_std']['rate']:.2%} vs Base {base_features['using_namespace_std']['rate']:.2%}; MAX constants {features['maxn_maxm_constant']['rate']:.2%} vs {base_features['maxn_maxm_constant']['rate']:.2%}; fixed arrays {features['static_array']['rate']:.2%} vs {base_features['static_array']['rate']:.2%}. It does not increase bits/stdc++.h, fast IO, or long long in this benchmark.",
        "",
        "## Answers and decision",
        "",
        "1. **Compile is higher than Base:** 90.00% vs 76.67% (+4 submissions).",
        "2. **Offline AC is only slightly higher than Base:** 20.00% vs 16.67% (+1 net AC).",
        "3. **It is below Style-LoRA-512:** 6 vs 8 AC.",
        "4. **It is below Style-LoRA-1536:** 6 vs 7 AC, despite higher compile rate.",
        "5. **Easy changes most:** 4 vs Base 3; Medium stays 2 and Hard stays 0.",
        "6. **Error change:** CE and RE each fall by 4 vs Base, while WA rises by 7; semantic correctness is the bottleneck.",
        "7. **Smoke missing declarations are not systemic** in the formal set; API/type hallucinations and one truncation remain.",
        "8. **Instruction following does not improve** under the literal unfenced-code criterion; all models fence every output.",
        "9. **Personal style is learned partially**, especially namespace/MAX/fixed-array features.",
        "10. **The experiment does not prove Instruction SFT is more effective than pure Style LoRA:** compile improves, but AC remains below both Style adapters.",
        "",
        "Decision: this is case C. Do not start Style+Instruction yet. Phase 3.5 should diagnose training targets and semantic data quality, stratify by problem difficulty/response truncation, audit erroneous historical solutions, and test whether 1.5B model capacity limits algorithm reasoning—without changing this frozen result.",
        "",
        f"Generation completeness: {generation['problems']}/30; configuration `{json.dumps(generation['generation_config'])}`.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--outputs", type=Path, default=DEFAULT_OUTPUTS)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--manifest", type=Path, default=BENCHMARK_ROOT / "manifest.jsonl")
    args = parser.parse_args()
    root = args.report_root.resolve()
    manifest = load_manifest(args.manifest.resolve())
    problems = [item["id"] for item in manifest]
    if len(problems) != 30:
        raise ValueError(f"expected frozen 30 problems, found {len(problems)}")
    summary = load_json(root / "summary.json")
    failure = load_json(root / "failure_analysis.json")
    style = load_json(root / "style_analysis.json")
    generation = load_json(root / "generation.json")
    if set(summary) != set(MODELS) or generation["problems"] != 30:
        raise ValueError("incomplete four-model summary or generation metadata")
    normalize_compile_categories(failure, args.outputs.resolve())
    (root / "failure_analysis.json").write_text(
        json.dumps(failure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    index = failure_index(failure)
    matrix = status_matrix(index, problems)
    paired = {model: transition(matrix, model) for model in OLD_MODELS}
    paired["all_old_fail_instruction_ac"] = [
        p for p, row in matrix.items() if all(row[m] != "accepted" for m in OLD_MODELS) and row["Instruction-SFT-v1"] == "accepted"
    ]
    paired["all_four_fail"] = [p for p, row in matrix.items() if all(row[m] != "accepted" for m in MODELS)]
    following = instruction_following(args.raw.resolve(), args.outputs.resolve(), problems)
    (root / "instruction_following.json").write_text(json.dumps(following, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (root / "comparison.md").write_text(render_comparison(matrix, manifest, paired), encoding="utf-8")
    (root / "report.md").write_text(render_report(summary, failure, style, following, paired, generation), encoding="utf-8")
    print(json.dumps({"paired": paired, "instruction_following": following["models"]["Instruction-SFT-v1"] | {"per_problem": "omitted"}}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

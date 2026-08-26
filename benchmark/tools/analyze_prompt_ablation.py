#!/usr/bin/env python3
"""Analyze paired Prompt × Model benchmark results."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

BENCHMARK_DIR = Path(__file__).resolve().parent.parent
PROMPTS = ("P0", "P1", "P2")
MODELS = ("Base", "LoRA-512", "LoRA-1536")
DEFAULT_REPORTS = BENCHMARK_DIR / "reports" / "prompt_ablation"
DEFAULT_OUTPUTS = BENCHMARK_DIR / "ablation" / "prompt"
DEFAULT_MANIFEST = BENCHMARK_DIR / "manifest.jsonl"
RUNTIME_MARKERS = (
    "process exited",
    "assert",
    "double free",
    "corruption",
    "segmentation",
    "invalid pointer",
    "core dumped",
)
STYLE_PATTERNS = {
    "using_namespace_std": re.compile(r"\busing\s+namespace\s+std\s*;"),
    "maxn_maxm_constant": re.compile(r"\b(?:MAXN|MAXM|MAX_[A-Za-z0-9_]+)\b"),
    "static_array": re.compile(
        r"(?m)^\s*(?:static\s+)?(?:const\s+)?(?:unsigned\s+|signed\s+)?"
        r"(?:long\s+long|int|char|bool|double|float|short|size_t|[A-Z]\w*)"
        r"\s+\w+\s*(?:\[[^\]\n]+\])+\s*(?:[;=,])"
    ),
}


class AblationError(RuntimeError):
    """Raised when prompt-ablation artifacts are incomplete or inconsistent."""


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise AblationError(f"JSON file not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AblationError(f"invalid JSON in {path}: {exc.msg}") from exc


def load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    problems: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        raise AblationError(f"manifest not found: {path}")
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not (line := raw.strip()) or line.startswith("#"):
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AblationError(f"{path}:{number}: {exc.msg}") from exc
        problem_id = entry.get("id") if isinstance(entry, dict) else None
        if not isinstance(problem_id, str) or not problem_id:
            raise AblationError(f"{path}:{number}: missing id")
        problems[problem_id] = entry
    if len(problems) != 30:
        raise AblationError(f"expected 30 problems, found {len(problems)}")
    return problems


def status(record: dict[str, Any]) -> str:
    if record["ac"]:
        return "AC"
    if not record["compile"]:
        return "CE"
    error = str(record.get("error", "")).lower()
    if "time limit" in error:
        return "TLE"
    if "output limit" in error or "size limit" in error:
        return "OLE"
    if any(marker in error for marker in RUNTIME_MARKERS):
        return "RE"
    return "WA"


def load_details(reports_root: Path, problems: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str], dict[str, Any]]]:
    combined: list[dict[str, Any]] = []
    indexed: dict[tuple[str, str, str], dict[str, Any]] = {}
    for prompt in PROMPTS:
        values = load_json(reports_root / prompt / "details.json")
        if not isinstance(values, list):
            raise AblationError(f"{prompt}/details.json must be an array")
        for item in values:
            model, problem = item.get("model"), item.get("problem")
            if model not in MODELS or problem not in problems:
                raise AblationError(f"unexpected result: {prompt}/{model}/{problem}")
            key = (prompt, model, problem)
            if key in indexed:
                raise AblationError(f"duplicate result: {key}")
            record = {"prompt": prompt, **item, "status": status(item)}
            indexed[key] = record
            combined.append(record)
    expected = {(p, m, q) for p in PROMPTS for m in MODELS for q in problems}
    if set(indexed) != expected:
        raise AblationError(f"incomplete results; missing={sorted(expected-set(indexed))}")
    return combined, indexed


def metrics(records: list[dict[str, Any]], problems: dict[str, Any]) -> dict[str, Any]:
    total = len(records)
    compiled = sum(record["compile"] for record in records)
    accepted = sum(record["ac"] for record in records)
    failures = Counter(record["status"] for record in records)
    difficulty = {}
    for level in ("easy", "medium", "hard"):
        subset = [r for r in records if problems[r["problem"]]["difficulty"] == level]
        difficulty[level] = {"total": len(subset), "ac": sum(r["ac"] for r in subset)}
    return {
        "total": total,
        "compile": compiled,
        "compile_rate": round(compiled / total, 6),
        "ac": accepted,
        "ac_rate": round(accepted / total, 6),
        "failure_distribution": dict(sorted(failures.items())),
        "difficulty": difficulty,
    }


def paired(indexed: dict[tuple[str, str, str], dict[str, Any]], problems: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for model in MODELS:
        result[model] = {}
        for left, right in (("P0", "P1"), ("P0", "P2"), ("P1", "P2")):
            gained = [q for q in problems if not indexed[left, model, q]["ac"] and indexed[right, model, q]["ac"]]
            lost = [q for q in problems if indexed[left, model, q]["ac"] and not indexed[right, model, q]["ac"]]
            status_changes = Counter(
                f"{indexed[left, model, q]['status']}->{indexed[right, model, q]['status']}"
                for q in problems
                if indexed[left, model, q]["status"] != indexed[right, model, q]["status"]
            )
            result[model][f"{left}_vs_{right}"] = {
                "gained_ac": gained,
                "lost_ac": lost,
                "net_ac": len(gained) - len(lost),
                "status_changes": dict(sorted(status_changes.items())),
            }
    return result


def style_metrics(outputs_root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for prompt in PROMPTS:
        result[prompt] = {}
        for model in MODELS:
            files = sorted((outputs_root / prompt / model).glob("p*.cpp"))
            if len(files) != 30:
                raise AblationError(f"expected 30 outputs for {prompt}/{model}, found {len(files)}")
            counts = Counter()
            for path in files:
                code = path.read_text(encoding="utf-8", errors="replace")
                counts.update(name for name, pattern in STYLE_PATTERNS.items() if pattern.search(code))
            result[prompt][model] = {
                name: {"count": counts[name], "rate": round(counts[name] / 30, 6)}
                for name in STYLE_PATTERNS
            }
    return result


def build_summary(reports_root: Path, outputs_root: Path, manifest: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    problems = load_manifest(manifest)
    details, indexed = load_details(reports_root, problems)
    rows = []
    for prompt in PROMPTS:
        for model in MODELS:
            rows.append({"prompt": prompt, "model": model, **metrics([r for r in details if r["prompt"] == prompt and r["model"] == model], problems)})
    pair_data = paired(indexed, problems)
    common_p0_fail = [q for q in problems if all(not indexed["P0", m, q]["ac"] for m in MODELS)]
    recovered = {
        prompt: {
            model: [q for q in common_p0_fail if indexed[prompt, model, q]["ac"]]
            for model in MODELS
        }
        for prompt in ("P1", "P2")
    }
    totals = {prompt: sum(row["ac"] for row in rows if row["prompt"] == prompt) for prompt in PROMPTS}
    aggregate_failures = {
        prompt: dict(sorted(sum(
            (Counter(row["failure_distribution"]) for row in rows if row["prompt"] == prompt),
            Counter(),
        ).items()))
        for prompt in PROMPTS
    }
    best_total = max(totals.values())
    best = [prompt for prompt, value in totals.items() if value == best_total]
    stable_lora = all(
        next(row["ac"] for row in rows if row["prompt"] == prompt and row["model"] == "LoRA-512")
        > next(row["ac"] for row in rows if row["prompt"] == prompt and row["model"] == "Base")
        and next(row["ac"] for row in rows if row["prompt"] == prompt and row["model"] == "LoRA-1536")
        > next(row["ac"] for row in rows if row["prompt"] == prompt and row["model"] == "Base")
        for prompt in PROMPTS
    )
    summary = {
        "metadata": {"prompts": list(PROMPTS), "models": list(MODELS), "problems": 30, "generations": 270},
        "rows": rows,
        "paired": pair_data,
        "p0_common_failures": common_p0_fail,
        "p0_common_failure_recovery": recovered,
        "style_auxiliary": style_metrics(outputs_root),
        "aggregate_ac": totals,
        "aggregate_failure_distribution": aggregate_failures,
        "best_prompt": best,
        "lora_both_above_base_all_prompts": stable_lora,
    }
    return summary, details


def fmt_count(value: int, total: int) -> str:
    return f"{value}/{total} ({value/total:.2%})"


def render_markdown(summary: dict[str, Any]) -> str:
    rows = summary["rows"]
    lookup = {(row["prompt"], row["model"]): row for row in rows}
    lines = ["# PersonalCoder Prompt Ablation", "", "## Overall", "", "| Prompt | Model | Compile | Offline AC |", "| --- | --- | ---: | ---: |"]
    for row in rows:
        lines.append(f"| {row['prompt']} | {row['model']} | {fmt_count(row['compile'],30)} | {fmt_count(row['ac'],30)} |")
    lines += ["", "## AC by Difficulty", "", "| Prompt | Model | Easy | Medium | Hard |", "| --- | --- | ---: | ---: | ---: |"]
    for row in rows:
        d = row["difficulty"]
        lines.append(f"| {row['prompt']} | {row['model']} | {d['easy']['ac']}/12 | {d['medium']['ac']}/12 | {d['hard']['ac']}/6 |")
    lines += ["", "## Failure distribution", "", "| Prompt | Model | CE | RE | TLE | OLE | WA | AC |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        f = row["failure_distribution"]
        lines.append(f"| {row['prompt']} | {row['model']} | {f.get('CE',0)} | {f.get('RE',0)} | {f.get('TLE',0)} | {f.get('OLE',0)} | {f.get('WA',0)} | {f.get('AC',0)} |")
    lines += ["", "Aggregate across the three models:", ""]
    for prompt in PROMPTS:
        lines.append(f"- {prompt}: {summary['aggregate_failure_distribution'][prompt]}")
    lines += ["", "## Paired problem-level changes", ""]
    for model in MODELS:
        lines.append(f"### {model}")
        lines.append("")
        for key, value in summary["paired"][model].items():
            label = key.replace("_vs_", " → ")
            lines.append(f"- {label}: gained={value['gained_ac'] or 'none'}; lost={value['lost_ac'] or 'none'}; net={value['net_ac']:+d}; status changes={value['status_changes']}")
        lines.append("")
    lines += ["## Previously common P0 failures recovered", ""]
    lines.append(f"P0 common failures ({len(summary['p0_common_failures'])}): {summary['p0_common_failures']}")
    for prompt, models in summary["p0_common_failure_recovery"].items():
        for model, problems in models.items():
            lines.append(f"- {prompt}/{model}: {problems or 'none'}")
    lines += [
        "",
        "## Key code-level observations",
        "",
        "- Base/P1/p008 changes a hard-coded assertion harness into the required stdin/stdout solution and becomes AC; this is a genuine instruction-following repair.",
        "- Base/P1/p003 replaces the required abbreviation count with asterisks, while Base/P1/p019 replaces the input-driven solution with a hard-coded self-check. Both were P0 AC and become WA, showing that a longer checklist can distract generation.",
        "- LoRA-1536/P1/p009 removes unsolicited Chinese input prompts and becomes AC; p014 replaces the conflicting `rank` identifier with a parent vector and changes CE to AC. These are output/implementation repairs rather than new algorithms.",
        "- LoRA-512 and LoRA-1536 both lose p008 under enhanced prompts after replacing the correct direct loop with incorrect digit-vector simulations.",
        "- LoRA-1536/P2/p012 restores pair-by-pair `long long` absolute differences and becomes AC. Base/P2/p028 misparses the multi-case input and floods output until OLE, so stronger wording did not ensure constraint compliance.",
    ]
    lines += ["", "## Auxiliary style check", "", "Rates below are Style-All and are not optimization targets.", "", "| Prompt | Model | using namespace std | MAX constant | static array |", "| --- | --- | ---: | ---: | ---: |"]
    for prompt in PROMPTS:
        for model in MODELS:
            s = summary["style_auxiliary"][prompt][model]
            lines.append(f"| {prompt} | {model} | {s['using_namespace_std']['count']}/30 | {s['maxn_maxm_constant']['count']}/30 | {s['static_array']['count']}/30 |")
    lines += ["", "## Required answers", ""]
    lines.append("1. **P1 is not better than P0.** AC deltas are Base -1, LoRA-512 -2, LoRA-1536 0; aggregate AC falls from 20/90 to 17/90.")
    lines.append("2. **P2 is not better than P0.** AC deltas are Base -1, LoRA-512 -1, LoRA-1536 0; aggregate AC falls to 18/90.")
    lines.append(f"3. **P0 has the highest aggregate AC:** {max(summary['aggregate_ac'].values())}/90; totals={summary['aggregate_ac']}.")
    lines.append("4. **Base sensitivity is small in total but material per problem:** AC is 5/4/4 (range 1), with P1 gaining p008 but losing p003 and p019.")
    lines.append("5. **LoRA-512 is the most prompt-sensitive:** AC is 8/6/7 (range 2); P1 loses p008 and p012, while P2 still loses p008.")
    lines.append("6. **LoRA-1536 is aggregate-stable but not problem-stable:** AC remains 7/7/7, while P1 gains p009/p014 and loses p005/p008; P2 gains p012 and loses p008.")
    lines.append(f"7. **Both LoRAs exceed Base under every prompt:** {summary['lora_both_above_base_all_prompts']}. This is stable within this audited 30-problem sample, not proof of a general capability gain.")
    lines.append("8. **Enhanced prompts do not reduce semantic failures.** P1 raises aggregate CE from 19 to 23 and lowers AC by 3. P2 lowers CE/RE to 17/4 from 19/5, but WA rises from 46 to 50 (plus one OLE) and AC falls by 2; its gains are compile-side, not correctness-side.")
    lines.append("9. **There are cross-model trade-offs.** P1 recovers two common failures only for LoRA-1536, yet lowers Base and LoRA-512 AC; enhanced prompts also exchange different AC identities even when LoRA-1536's total is unchanged.")
    lines.append("10. **The bottleneck is not Prompt alone.** The evidence points mainly to base algorithmic/semantic capability and the instruction-training data form; Style LoRA changes code priors but does not reliably repair reasoning. Stop prompt tuning, validate the persistent LoRA advantage on a larger audited benchmark, and prioritize a controlled Instruction SFT if training is resumed.")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--outputs-root", type=Path, default=DEFAULT_OUTPUTS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    try:
        summary, details = build_summary(args.reports_root.resolve(), args.outputs_root.resolve(), args.manifest.resolve())
    except AblationError as exc:
        parser.error(str(exc))
    args.reports_root.mkdir(parents=True, exist_ok=True)
    (args.reports_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.reports_root / "details.json").write_text(json.dumps(details, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.reports_root / "prompt_ablation.md").write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps({"aggregate_ac": summary["aggregate_ac"], "best_prompt": summary["best_prompt"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

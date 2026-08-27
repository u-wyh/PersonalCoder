#!/usr/bin/env python3
"""Measure Instruction-SFT complexity and compare it with the frozen benchmark."""

from __future__ import annotations

import argparse
import html
import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "instruction_sft/data/processed/dataset.jsonl"
MANIFEST = ROOT / "benchmark/manifest.jsonl"
MODEL = Path("/data/PersonalCoder/model")
DEFAULT_JSON = ROOT / "instruction_sft/reports/dataset_distribution.json"
DEFAULT_MD = ROOT / "instruction_sft/reports/dataset_distribution.md"
DEFAULT_GAP = ROOT / "instruction_sft/reports/train_benchmark_gap.md"

LUOGU_BANDS = {0: "unknown", 1: "easy", 2: "easy", 3: "medium", 4: "medium", 5: "hard", 6: "hard", 7: "hard"}
CATEGORY_TAGS = {
    "dp": {"dp"},
    "graph": {"graphs", "trees", "dfs and similar", "dsu", "shortest paths", "2-sat"},
    "data_structure": {"data structures"},
    "math": {"math", "combinatorics", "number theory", "fft", "probabilities", "geometry"},
    "string": {"strings", "string suffix structures", "hashing"},
    "greedy": {"greedy"},
    "search": {"brute force", "binary search", "ternary search", "meet-in-the-middle"},
    "implementation": {"implementation", "constructive algorithms", "sortings"},
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def quantiles(values: list[int]) -> dict[str, float]:
    ordered = sorted(values)
    def pick(q: float) -> int:
        return ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1)))]
    return {"min": ordered[0], "mean": round(statistics.mean(ordered), 2), "p50": pick(.5), "p90": pick(.9), "p95": pick(.95), "max": ordered[-1]}


def code_stats(code: str) -> tuple[int, int, int]:
    lines = len(code.splitlines())
    functions = len(re.findall(r"(?m)^\s*(?!if\b|for\b|while\b|switch\b)(?:[\w:<>,~*&]+\s+)+[A-Za-z_]\w*\s*\([^;{}]*\)\s*(?:const\s*)?\{", code))
    templates = len(re.findall(r"\btemplate\s*<", code))
    return lines, functions, templates


def metadata(record: dict[str, Any]) -> tuple[str, str | None, list[str], str]:
    html_path = ROOT / "instruction_sft/data/statements/raw" / record["source"] / f"{record['problem_id']}.html"
    page = html_path.read_text(encoding="utf-8", errors="ignore") if html_path.is_file() else ""
    if record["source"] == "luogu":
        match = re.search(r'"difficulty"\s*:\s*(\d+)', page)
        raw = match.group(1) if match else None
        band = LUOGU_BANDS.get(int(raw), "unknown") if raw is not None else "unknown"
        return band, raw, [], "luogu_official_difficulty"
    spans = re.findall(r'<span[^>]*class="tag-box"[^>]*>(.*?)</span>', page, flags=re.S)
    tags = [html.unescape(re.sub(r"<[^>]+>", "", value)).strip() for value in spans]
    rating = next((int(tag[1:]) for tag in tags if re.fullmatch(r"\*\d+", tag)), None)
    algo = [tag for tag in tags if tag and not tag.startswith("*")]
    if rating is None:
        return "unknown", None, algo, "codeforces_tags"
    band = "easy" if rating <= 1200 else "medium" if rating <= 1900 else "hard"
    return band, str(rating), algo, "codeforces_rating"


def render_md(report: dict[str, Any]) -> str:
    train = report["training"]
    lines = ["# Instruction-SFT Dataset Distribution", "", f"Pairs: **{train['total']}**; sources: `{train['sources']}`.", "", "## Difficulty", "", "| Band | Count | Rate |", "| --- | ---: | ---: |"]
    for band in ("easy", "medium", "hard", "unknown"):
        value = train["difficulty_bands"].get(band, 0)
        lines.append(f"| {band} | {value} | {value/train['total']:.2%} |")
    lines += ["", "Difficulty mapping: Luogu 1–2 easy, 3–4 medium, 5–7 hard; Codeforces ≤1200 easy, 1300–1900 medium, ≥2000 hard. Missing metadata stays unknown.", "", "## Response structure", "", f"- Response tokens: `{train['response_tokens']}`", f"- Code lines: `{train['code_lines']}`", f"- Functions: `{train['function_count']}`", f"- Template declarations: `{train['template_declarations']}`", f"- Length bands: `{train['response_length_bands']}`", "", "## Algorithm metadata", "", f"Reliable named tags cover **{train['named_tag_coverage']}/{train['total']}** pairs. Luogu cached pages expose numeric tag IDs without a local name dictionary, so they are deliberately not guessed.", "", f"Canonical categories (multi-label; untagged pairs count as other/unknown): `{train['algorithm_categories']}`", f"Codeforces tags: `{train['named_algorithm_tags']}`", ""]
    return "\n".join(lines)


def render_gap(report: dict[str, Any]) -> str:
    train, bench = report["training"], report["benchmark"]
    return f"""# Train–Benchmark Gap

## Difficulty

| Split | Easy | Medium | Hard | Unknown |
| --- | ---: | ---: | ---: | ---: |
| Instruction-SFT ({train['total']}) | {train['difficulty_bands'].get('easy',0)} | {train['difficulty_bands'].get('medium',0)} | {train['difficulty_bands'].get('hard',0)} | {train['difficulty_bands'].get('unknown',0)} |
| Benchmark ({bench['total']}) | {bench['difficulty_bands'].get('easy',0)} | {bench['difficulty_bands'].get('medium',0)} | {bench['difficulty_bands'].get('hard',0)} | {bench['difficulty_bands'].get('unknown',0)} |

The training corpus is historical accepted/compiled-looking code paired with statements, not curated reasoning supervision. Its official difficulty mix is measurable, but named algorithm coverage is reliable only for {train['named_tag_coverage']}/{train['total']} Codeforces-tagged pairs. The benchmark intentionally fixes a 12/12/6 easy/medium/hard mix and tests unseen full solutions under stronger tests.

## Length and task shape

- Training response tokens: `{train['response_tokens']}`.
- Benchmark reference-code tokens: `{bench['reference_tokens']}`.
- Training instruction tokens: `{train['instruction_tokens']}`.
- Benchmark statement tokens: `{bench['statement_tokens']}`.

The dominant gap is supervision quality rather than proof of a pure difficulty mismatch: passing a few official samples does not establish AC, and the corpus includes demonstrably wrong/mismatched programs. Medium/Hard benchmark performance may also expose model-capacity limits, but capacity cannot be isolated before semantic cleaning.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MD)
    parser.add_argument("--gap", type=Path, default=DEFAULT_GAP)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    rows = read_jsonl(args.dataset)
    difficulties, raw_difficulties, tag_counts, sources, categories = Counter(), Counter(), Counter(), Counter(), Counter()
    response_tokens, instruction_tokens, line_counts, function_counts, template_counts = [], [], [], [], []
    tagged = 0
    for row in rows:
        sources[row["source"]] += 1
        band, raw, tags, provenance = metadata(row)
        difficulties[band] += 1
        raw_difficulties[f"{provenance}:{raw or 'unknown'}"] += 1
        tag_counts.update(tags)
        tagged += bool(tags)
        matched_categories = {category for category, accepted in CATEGORY_TAGS.items() if accepted.intersection(tags)}
        categories.update(matched_categories or {"other/unknown"})
        response_tokens.append(int(row["response_token_length"]))
        instruction_tokens.append(len(tokenizer.encode(row["instruction"], add_special_tokens=False)))
        lines, functions, templates = code_stats(row["response"])
        line_counts.append(lines); function_counts.append(functions); template_counts.append(templates)
    length_bands = Counter("short_<=512" if n <= 512 else "medium_513_1024" if n <= 1024 else "long_>1024" for n in response_tokens)

    manifest = read_jsonl(args.manifest)
    benchmark_difficulty, benchmark_tags, benchmark_sources = Counter(), Counter(), Counter()
    statement_tokens, reference_tokens = [], []
    for item in manifest:
        benchmark_sources[item["oj"]] += 1
        benchmark_difficulty[item["difficulty"]] += 1
        benchmark_tags.update(item.get("tags", []))
        problem = ROOT / "benchmark" / item["path"]
        statement_tokens.append(len(tokenizer.encode((problem / "statement.md").read_text(encoding="utf-8"), add_special_tokens=False)))
        reference_tokens.append(len(tokenizer.encode((problem / "reference.cpp").read_text(encoding="utf-8"), add_special_tokens=False)))
    report = {
        "method": {"tokenizer": str(args.model), "local_files_only": True, "difficulty_mapping": "Luogu 1-2 easy, 3-4 medium, 5-7 hard; CF <=1200 easy, 1300-1900 medium, >=2000 hard"},
        "training": {
            "total": len(rows), "sources": dict(sources), "difficulty_bands": dict(difficulties), "raw_difficulty": dict(sorted(raw_difficulties.items())),
            "response_tokens": quantiles(response_tokens), "instruction_tokens": quantiles(instruction_tokens),
            "response_length_bands": dict(length_bands), "code_lines": quantiles(line_counts),
            "function_count": quantiles(function_counts), "template_declarations": quantiles(template_counts),
            "named_tag_coverage": tagged, "algorithm_categories": dict(categories), "named_algorithm_tags": dict(tag_counts.most_common()),
        },
        "benchmark": {
            "total": len(manifest), "sources": dict(benchmark_sources), "difficulty_bands": dict(benchmark_difficulty), "tags": dict(benchmark_tags.most_common()),
            "statement_tokens": quantiles(statement_tokens), "reference_tokens": quantiles(reference_tokens),
        },
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(render_md(report), encoding="utf-8")
    args.gap.write_text(render_gap(report), encoding="utf-8")
    print(json.dumps({"training_difficulty": report["training"]["difficulty_bands"], "benchmark_difficulty": report["benchmark"]["difficulty_bands"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

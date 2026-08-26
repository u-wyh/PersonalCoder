#!/usr/bin/env python3
"""Build verified real-statement/code pairs from the Phase 3.2 cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from build_dataset import (
    MODEL_PATH,
    RANDOM_SEED,
    SIMILARITY_THRESHOLD,
    contamination,
    load_benchmark,
    load_tokenizer,
    token_length_distribution,
)
from common import INSTRUCTION_ROOT, DatasetError, load_style_records, read_jsonl, write_json, write_jsonl
from normalize_statement import NORMALIZED_ROOT


MANIFEST_PATH = INSTRUCTION_ROOT / "data" / "statements" / "all_manifest.jsonl"
SELECTED_PATH = INSTRUCTION_ROOT / "data" / "processed" / "selected_codes.jsonl"
VALIDATION_PATH = INSTRUCTION_ROOT / "reports" / "statement_validation.json"
DATASET_PATH = INSTRUCTION_ROOT / "data" / "processed" / "dataset.jsonl"
TRAIN_PATH = INSTRUCTION_ROOT / "data" / "splits" / "train.jsonl"
VAL_PATH = INSTRUCTION_ROOT / "data" / "splits" / "val.jsonl"
REPORT_ROOT = INSTRUCTION_ROOT / "reports"


def deterministic_key(record: dict[str, Any]) -> str:
    identity = f"{RANDOM_SEED}:{record['source']}:{record['problem_id']}"
    return hashlib.sha256(identity.encode()).hexdigest()


def run(manifest_path: Path, validation_path: Path) -> dict[str, Any]:
    manifest = {
        (str(item["source"]), str(item["problem_id"]).upper()): item
        for item in read_jsonl(manifest_path)
    }
    selected = {
        (str(item["source"]), str(item["problem_id"]).upper()): item
        for item in read_jsonl(SELECTED_PATH)
    }
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    verified = {
        (str(item["source"]), str(item["problem_id"]).upper())
        for item in validation["records"]
        if item["statement_verified"]
    }
    code_by_sha = {item["sha256"]: item for item in load_style_records()}
    benchmark = load_benchmark()
    tokenizer = load_tokenizer()

    records: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []
    for key in sorted(verified):
        metadata = manifest.get(key) or selected.get(key)
        chosen = selected.get(key)
        if not metadata or not chosen:
            missing.append({"source": key[0], "problem_id": key[1], "reason": "metadata_missing"})
            continue
        code_record = code_by_sha.get(chosen["sha256"])
        statement_path = NORMALIZED_ROOT / key[0] / f"{key[1]}.md"
        if not code_record or not statement_path.is_file():
            missing.append(
                {
                    "source": key[0],
                    "problem_id": key[1],
                    "reason": "code_or_normalized_statement_missing",
                }
            )
            continue
        statement = statement_path.read_text(encoding="utf-8").strip()
        code = str(code_record["text"])
        matches = contamination(chosen, code, statement, benchmark)
        if matches:
            exclusions.append(
                {
                    "source": key[0],
                    "problem_id": key[1],
                    "origin_path": chosen["path"],
                    "code_sha256": chosen["sha256"],
                    "matches": matches,
                }
            )
            continue
        records.append(
            {
                "id": f"{key[0]}_{key[1]}",
                "source": key[0],
                "problem_id": key[1],
                "instruction": statement,
                "response": code,
                "code_sha256": chosen["sha256"],
                "origin_path": chosen["path"],
                "statement_path": str(statement_path.relative_to(INSTRUCTION_ROOT.parent)),
                "timestamp": chosen.get("timestamp"),
                "age_bucket": chosen.get("age_bucket", "unknown_time"),
                "verified": True,
                "statement_verified": True,
                "verification_level": "local_cpp17_compile_only",
                "response_token_length": len(tokenizer.encode(code, add_special_tokens=False)),
            }
        )

    ordered = sorted(records, key=deterministic_key)
    val_count = round(len(ordered) * 0.1) if len(ordered) >= 10 else 0
    val = ordered[:val_count]
    train = ordered[val_count:]
    train_ids = {(item["source"], item["problem_id"]) for item in train}
    val_ids = {(item["source"], item["problem_id"]) for item in val}
    if train_ids & val_ids:
        raise DatasetError("problem ID leakage detected between train and validation")
    if len({item["code_sha256"] for item in ordered}) != len(ordered):
        raise DatasetError("duplicate code SHA256 detected in final pairs")

    write_jsonl(DATASET_PATH, ordered)
    write_jsonl(TRAIN_PATH, train)
    write_jsonl(VAL_PATH, val)
    write_json(
        REPORT_ROOT / "benchmark_exclusion_phase3_2.json",
        {
            "policy": {
                "held_out_problems": 30,
                "checks": ["problem_id", "sha256", "code_similarity", "statement_similarity"],
                "similarity_threshold": SIMILARITY_THRESHOLD,
                "action": "exclude",
            },
            "excluded_count": len(exclusions),
            "excluded": exclusions,
        },
    )
    write_json(REPORT_ROOT / "pair_build_missing.json", missing)

    statement_lengths = [len(item["instruction"]) for item in ordered]
    response_lengths = [item["response_token_length"] for item in ordered]
    fetch_summary = json.loads((REPORT_ROOT / "statement_fetch.json").read_text(encoding="utf-8"))
    source_distribution = dict(sorted(Counter(item["source"] for item in ordered).items()))
    summary = {
        "eligible_selected_codes": len(selected),
        "locatable_statements": len(manifest),
        "fetch_attempted": fetch_summary["attempted"],
        "fetch_success": fetch_summary["fetched"],
        "fetch_success_rate": round(fetch_summary["fetched"] / max(fetch_summary["attempted"], 1), 6),
        "verified_statements": len(verified),
        "statement_verification_rate": round(len(verified) / max(fetch_summary["attempted"], 1), 6),
        "manual_required": validation["summary"]["manual_required"],
        "benchmark_excluded": len(exclusions),
        "pair_build_missing": len(missing),
        "final_pairs": len(ordered),
        "train": len(train),
        "validation": len(val),
        "source_distribution": source_distribution,
        "statement_character_distribution": token_length_distribution(statement_lengths),
        "response_token_distribution": token_length_distribution(response_lengths),
        "training_threshold": ">=500",
        "threshold_met": len(ordered) >= 500,
        "recommendation": "ready_for_phase3_3_instruction_sft" if len(ordered) >= 500 else "collect_more_verified_pairs",
        "tokenizer_path": str(MODEL_PATH),
        "split_seed": RANDOM_SEED,
    }
    write_json(REPORT_ROOT / "dataset_summary.json", summary)
    card = [
        "# PersonalCoder Instruction SFT Dataset v1",
        "",
        "Real public problem statements paired with the user's compile-passed historical solutions. No statement is synthesized or rewritten.",
        "",
        "| Metric | Count |",
        "| --- | ---: |",
        f"| Eligible selected codes | {summary['eligible_selected_codes']} |",
        f"| Locatable statements | {summary['locatable_statements']} |",
        f"| Fetch success | {summary['fetch_success']} / {summary['fetch_attempted']} |",
        f"| Verified statements | {summary['verified_statements']} |",
        f"| Manual review required / excluded | {summary['manual_required']} |",
        f"| Benchmark contamination excluded | {summary['benchmark_excluded']} |",
        f"| Final instruction-response pairs | {summary['final_pairs']} |",
        f"| Train / validation | {summary['train']} / {summary['validation']} |",
        "",
        "## Provenance and validation",
        "",
        "- Statements were acquired from public Codeforces/Codeforces Gym and Luogu problem pages with a low-rate, resumable cache.",
        "- A deterministic 100-problem pilot passed the expansion gate; ten pilot pairs were manually checked against their code.",
        "- Failed, short, incomplete, ID-mismatched, login/challenge, or error pages are excluded rather than repaired synthetically.",
        "- Every response was already SHA256-deduplicated and passed local `g++ -std=c++17 -O2 -pipe -fsyntax-only` selection in Phase 3.1.",
        "- Held-out benchmark contamination is checked by source/problem ID, SHA256, code similarity, and statement similarity.",
        "- Train/validation is a deterministic 90/10 problem-level split with seed 42 and no problem-ID overlap.",
        "- Raw page caches and normalized per-problem Markdown are local ignored artifacts; the committed JSONL preserves the full verified instruction text.",
        "",
        "## Distribution",
        "",
        f"- Source: `{json.dumps(source_distribution, ensure_ascii=False)}`",
        f"- Statement characters: `{json.dumps(summary['statement_character_distribution'], ensure_ascii=False)}`",
        f"- Response tokens: `{json.dumps(summary['response_token_distribution'], ensure_ascii=False)}`",
        "",
        "## Training decision",
        "",
        f"Threshold `{summary['training_threshold']}` met: **{str(summary['threshold_met']).lower()}**. Recommendation: `{summary['recommendation']}`. This phase does not train a model.",
    ]
    (REPORT_ROOT / "dataset_card.md").write_text("\n".join(card) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--validation", type=Path, default=VALIDATION_PATH)
    args = parser.parse_args()
    try:
        summary = run(args.manifest.resolve(), args.validation.resolve())
    except (DatasetError, OSError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

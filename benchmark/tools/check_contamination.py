#!/usr/bin/env python3
"""Check pilot references against local Style LoRA datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


TOOLS_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = TOOLS_DIR.parent
PROJECT_ROOT = BENCHMARK_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.evaluate import EvaluationError, load_manifest  # noqa: E402


DEFAULT_DATASETS = (
    PROJECT_ROOT / "data" / "processed" / "style",
    PROJECT_ROOT / "data" / "processed" / "style_v2",
    PROJECT_ROOT / "data" / "processed" / "style_v3",
)
DEFAULT_OUTPUT = BENCHMARK_DIR / "reports" / "contamination_report.json"
SPLITS = ("train.jsonl", "validation.jsonl", "test.jsonl")
TOKEN_PATTERN = re.compile(r"[A-Za-z_]\w*|\d+|[^\s]", re.UNICODE)
PATH_TOKEN_PATTERN = re.compile(r"[A-Za-z]+\d+|\d+[A-Za-z]+|[A-Za-z]+|\d+")


class ContaminationError(RuntimeError):
    """Raised when contamination inputs are unavailable or malformed."""


@dataclass
class DatasetRecord:
    sha256: str
    normalized_sha256: str
    text: str
    token_count: int
    shingles: frozenset[int]
    paths: set[str] = field(default_factory=set)
    datasets: set[str] = field(default_factory=set)


def _normalized_text(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.replace("\r\n", "\n").split("\n")).strip()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _tokens(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text)


def _shingles(tokens: list[str], width: int = 5) -> frozenset[int]:
    if len(tokens) < width:
        return frozenset(hash(token) for token in tokens)
    return frozenset(hash(tuple(tokens[i : i + width])) for i in range(len(tokens) - width + 1))


def _similarity(left: frozenset[int], right: frozenset[int]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ContaminationError(
                    f"{path}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(value, dict):
                raise ContaminationError(f"{path}:{line_number}: expected object")
            yield value


def load_dataset_records(dataset_dirs: Iterable[str | Path]) -> list[DatasetRecord]:
    """Load and deduplicate records shared by style v1/v2/v3."""
    records: dict[str, DatasetRecord] = {}
    for raw_dir in dataset_dirs:
        dataset_dir = Path(raw_dir).resolve()
        if not dataset_dir.is_dir():
            raise ContaminationError(f"dataset directory not found: {dataset_dir}")
        for split in SPLITS:
            split_path = dataset_dir / split
            if not split_path.is_file():
                raise ContaminationError(f"dataset split not found: {split_path}")
            for value in _read_jsonl(split_path):
                text = value.get("text")
                if not isinstance(text, str):
                    continue
                normalized = _normalized_text(text)
                normalized_sha = _sha256(normalized)
                declared_sha = str(value.get("sha256") or normalized_sha).lower()
                key = declared_sha + ":" + normalized_sha
                path_value = str(value.get("path") or "")
                record = records.get(key)
                if record is None:
                    tokens = _tokens(normalized)
                    record = DatasetRecord(
                        sha256=declared_sha,
                        normalized_sha256=normalized_sha,
                        text=normalized,
                        token_count=len(tokens),
                        shingles=_shingles(tokens),
                    )
                    records[key] = record
                if path_value:
                    record.paths.add(path_value)
                record.datasets.add(dataset_dir.name)
    return list(records.values())


def load_manifest_metadata(manifest_path: Path) -> list[dict[str, Any]]:
    try:
        problems = load_manifest(manifest_path)
    except EvaluationError as exc:
        raise ContaminationError(str(exc)) from exc
    values = list(_read_jsonl(manifest_path.resolve()))
    if len(values) != len(problems):
        raise ContaminationError("manifest metadata and resolved problems differ")
    result: list[dict[str, Any]] = []
    for value, problem in zip(values, problems):
        result.append(
            {
                "id": problem.problem_id,
                "problem_id": str(value.get("problem_id") or problem.problem_id),
                "reference": problem.directory / "reference.cpp",
            }
        )
    return result


def _path_has_identifier(path: str, identifiers: set[str]) -> bool:
    tokens = {token.lower() for token in PATH_TOKEN_PATTERN.findall(path)}
    normalized_identifiers = {
        re.sub(r"[^a-z0-9]", "", identifier) for identifier in identifiers
    }
    return bool(tokens & normalized_identifiers)


def _text_has_identifier(text: str, problem_id: str) -> bool:
    if len(problem_id) < 3:
        return False
    return bool(
        re.search(
            rf"(?<![A-Za-z0-9]){re.escape(problem_id)}(?![A-Za-z0-9])",
            text,
            re.IGNORECASE,
        )
    )


def check_problem(
    metadata: dict[str, Any],
    records: list[DatasetRecord],
    similarity_threshold: float,
) -> dict[str, Any]:
    reference_path = Path(metadata["reference"])
    reference_text = reference_path.read_text(encoding="utf-8")
    normalized = _normalized_text(reference_text)
    raw_sha = hashlib.sha256(reference_path.read_bytes()).hexdigest()
    normalized_sha = _sha256(normalized)
    tokens = _tokens(normalized)
    reference_shingles = _shingles(tokens)
    problem_id = metadata["problem_id"].lower()
    problem_id_parts = re.findall(r"[a-z]+|\d+", problem_id)
    identifiers = {metadata["id"].lower(), problem_id}
    if len(problem_id_parts) > 1 and problem_id_parts[0] in {
        "kattis",
        "boj",
        "uva",
        "icpc",
    }:
        identifiers.add("".join(problem_id_parts[1:]))

    reasons: set[str] = set()
    matches: list[dict[str, Any]] = []
    max_similarity = 0.0
    for record in records:
        evidence: set[str] = set()
        matching_paths = sorted(
            path for path in record.paths if _path_has_identifier(path, identifiers)
        )
        if matching_paths:
            evidence.add("path")
        if _text_has_identifier(record.text, metadata["problem_id"]):
            evidence.add("problem_id")
        if record.sha256 in {raw_sha, normalized_sha} or record.normalized_sha256 == normalized_sha:
            evidence.add("sha256")

        size_ratio = min(len(tokens), record.token_count) / max(len(tokens), record.token_count, 1)
        score = 0.0
        if size_ratio >= 0.35:
            score = _similarity(reference_shingles, record.shingles)
            max_similarity = max(max_similarity, score)
            if score >= similarity_threshold:
                evidence.add("similarity")
        if not evidence:
            continue
        reasons.update(evidence)
        matches.append(
            {
                "reasons": sorted(evidence),
                "datasets": sorted(record.datasets),
                "paths": matching_paths[:3] or sorted(record.paths)[:1],
                "sha256": record.sha256,
                "similarity": round(score, 6),
            }
        )

    return {
        "matched": bool(reasons),
        "reason": ",".join(sorted(reasons)) if reasons else "none",
        "max_similarity": round(max_similarity, 6),
        "matches": matches[:10],
    }


def run_check(
    manifest_path: str | Path,
    dataset_dirs: Iterable[str | Path],
    output_path: str | Path,
    similarity_threshold: float = 0.8,
) -> dict[str, Any]:
    if not 0.0 <= similarity_threshold <= 1.0:
        raise ContaminationError("similarity threshold must be between 0 and 1")
    manifest = Path(manifest_path).resolve()
    metadata = load_manifest_metadata(manifest)
    records = load_dataset_records(dataset_dirs)
    report = {
        item["id"]: check_problem(item, records, similarity_threshold)
        for item in metadata
    }
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=BENCHMARK_DIR / "manifest.jsonl"
    )
    parser.add_argument(
        "--datasets", nargs="+", type=Path, default=list(DEFAULT_DATASETS)
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--similarity-threshold", type=float, default=0.8)
    args = parser.parse_args()
    try:
        report = run_check(
            args.manifest, args.datasets, args.output, args.similarity_threshold
        )
    except ContaminationError as exc:
        parser.error(str(exc))
    matched = sum(item["matched"] for item in report.values())
    print(
        json.dumps(
            {"problems": len(report), "matched": matched, "clean": len(report) - matched},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

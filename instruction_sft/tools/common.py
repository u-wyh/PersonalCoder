#!/usr/bin/env python3
"""Shared local-only helpers for Instruction SFT dataset construction."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INSTRUCTION_ROOT = PROJECT_ROOT / "instruction_sft"
STYLE_ROOT = PROJECT_ROOT / "data" / "processed" / "style"
STYLE_V3_ROOT = PROJECT_ROOT / "data" / "processed" / "style_v3"
DEFAULT_SOURCE_ROOT = Path("/mnt/d/algorithm")
SPLITS = ("train", "validation", "test")
TOKEN_PATTERN = re.compile(r"[A-Za-z_]\w*|\d+|[^\s]", re.UNICODE)
LUOGU_PATTERN = re.compile(r"(?<![A-Za-z0-9])P\d{3,7}(?![A-Za-z0-9])", re.IGNORECASE)
LUOGU_URL_PATTERN = re.compile(
    r"luogu\.com\.cn/problem/(P\d{3,7})(?![A-Za-z0-9])", re.IGNORECASE
)
CODEFORCES_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])CF(\d{1,7})([A-Z]\d?)(?![A-Za-z0-9])", re.IGNORECASE
)
CODEFORCES_URL_PATTERN = re.compile(
    r"codeforces\.com/(?:problemset/problem/(\d{1,7})/|contest/(\d{1,7})/problem/)([A-Z]\d?)",
    re.IGNORECASE,
)
EXPLICIT_ICPC_PATTERN = re.compile(r"(?:ICPC|CCPC)", re.IGNORECASE)


class DatasetError(RuntimeError):
    """Raised for malformed or missing local dataset inputs."""


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.is_file():
        raise DatasetError(f"JSONL not found: {path}")
    with path.open(encoding="utf-8") as source:
        for line_number, raw in enumerate(source, 1):
            if not (line := raw.strip()):
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{path}:{line_number}: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise DatasetError(f"{path}:{line_number}: expected object")
            yield value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            output.write("\n")


def normalize_code(text: str) -> str:
    return "\n".join(
        line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    ).strip()


def code_tokens(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(normalize_code(text))


def shingles(tokens: list[str], width: int = 5) -> frozenset[str]:
    if len(tokens) < width:
        return frozenset(hashlib.sha1(token.encode()).hexdigest() for token in tokens)
    return frozenset(
        hashlib.sha1("\0".join(tokens[index : index + width]).encode()).hexdigest()
        for index in range(len(tokens) - width + 1)
    )


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def load_style_records() -> list[dict[str, Any]]:
    """Load the deduplicated v1 corpus and attach reliable v3 time metadata."""
    time_index: dict[tuple[str, str], dict[str, Any]] = {}
    for split in SPLITS:
        for record in read_jsonl(STYLE_V3_ROOT / f"{split}.jsonl"):
            time_index[(str(record.get("path")), str(record.get("sha256")))] = record

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for split in SPLITS:
        for record in read_jsonl(STYLE_ROOT / f"{split}.jsonl"):
            digest = str(record.get("sha256") or "")
            if not digest or digest in seen:
                raise DatasetError(f"duplicate or missing SHA256 in style v1: {digest}")
            seen.add(digest)
            key = (str(record.get("path")), digest)
            time_record = time_index.get(key, {})
            records.append(
                {
                    "path": str(record["path"]),
                    "source_type": str(record.get("source_type") or "unknown"),
                    "sha256": digest,
                    "text": str(record.get("text") or ""),
                    "style_split": split,
                    "timestamp": time_record.get("modified_at"),
                    "age_bucket": time_record.get("time_bucket", "unknown_time"),
                    "time_reliable": bool(time_record.get("time_reliable", False)),
                    "token_length": time_record.get("token_length"),
                }
            )
    return records


def git_latest_metadata(source_root: Path) -> dict[str, dict[str, Any]]:
    """Read the latest local commit timestamp/subject for each C++ path in one pass."""
    if not (source_root / ".git").is_dir():
        return {}
    command = [
        "git",
        "-C",
        str(source_root),
        "-c",
        "core.quotepath=false",
        "log",
        "--format=__INSTRUCTION_COMMIT__%ct%x09%s",
        "--name-only",
        "--",
        "*.cpp",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=120)
    if completed.returncode != 0:
        return {}
    current_time: int | None = None
    current_subject = ""
    result: dict[str, dict[str, Any]] = {}
    for raw in completed.stdout.splitlines():
        line = raw.strip()
        if line.startswith("__INSTRUCTION_COMMIT__"):
            fields = line.removeprefix("__INSTRUCTION_COMMIT__").split("\t", 1)
            current_time = int(fields[0])
            current_subject = fields[1] if len(fields) > 1 else ""
        elif current_time is not None and line.lower().endswith(".cpp") and line not in result:
            result[line] = {
                "git_last_commit_at": datetime.fromtimestamp(current_time).astimezone().isoformat(),
                "git_subject": current_subject,
            }
    return result


def _extract_candidates(text: str) -> set[tuple[str, str]]:
    candidates: set[tuple[str, str]] = set()
    for match in LUOGU_PATTERN.finditer(text):
        candidates.add(("luogu", match.group(0).upper()))
    for match in LUOGU_URL_PATTERN.finditer(text):
        candidates.add(("luogu", match.group(1).upper()))
    for match in CODEFORCES_PATTERN.finditer(text):
        candidates.add(("codeforces", f"{match.group(1)}{match.group(2).upper()}"))
    for match in CODEFORCES_URL_PATTERN.finditer(text):
        contest = match.group(1) or match.group(2)
        candidates.add(("codeforces", f"{contest}{match.group(3).upper()}"))
    return candidates


def _icpc_candidate(path: str, header: str) -> tuple[str, str] | None:
    context = f"{path}\n{header}"
    if not EXPLICIT_ICPC_PATTERN.search(context):
        return None
    stem = Path(path).stem
    if not re.fullmatch(r"[A-Ma-m](?:[_-]?\d+)?", stem):
        return None
    event_parts = [
        part for part in Path(path).parts[:-1]
        if EXPLICIT_ICPC_PATTERN.search(part) or re.search(r"20\d{2}", part)
    ]
    if not event_parts:
        return None
    slug = re.sub(r"[^A-Za-z0-9]+", "_", "_".join(event_parts)).strip("_").lower()
    return "icpc", f"{slug}_{stem.upper()}"


def discover_problem_id(
    path: str,
    text: str,
    git_subject: str = "",
) -> dict[str, Any]:
    """Conservatively recover one unambiguous problem ID with evidence."""
    header = "\n".join(text.splitlines()[:100])
    filename_candidates = _extract_candidates(Path(path).stem)
    path_candidates = _extract_candidates(path)
    header_candidates = _extract_candidates(header)
    git_candidates = _extract_candidates(git_subject)
    levels = (
        ("filename", filename_candidates),
        ("path", path_candidates),
        ("header", header_candidates),
        ("git_subject", git_candidates),
    )
    all_candidates = sorted(set().union(*(values for _, values in levels)))
    for evidence, candidates in levels:
        if len(candidates) == 1:
            source, problem_id = next(iter(candidates))
            return {
                "source": source,
                "problem_id": problem_id,
                "evidence": evidence,
                "candidates": [f"{item[0]}:{item[1]}" for item in all_candidates],
            }
        if len(candidates) > 1:
            return {
                "source": "unknown",
                "problem_id": None,
                "evidence": f"ambiguous_{evidence}",
                "candidates": [f"{item[0]}:{item[1]}" for item in all_candidates],
            }
    if icpc := _icpc_candidate(path, header):
        return {
            "source": icpc[0],
            "problem_id": icpc[1],
            "evidence": "explicit_icpc_path",
            "candidates": [f"{icpc[0]}:{icpc[1]}"],
        }
    return {
        "source": "unknown",
        "problem_id": None,
        "evidence": "none",
        "candidates": [],
    }

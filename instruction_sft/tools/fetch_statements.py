#!/usr/bin/env python3
"""Low-rate, resumable resolver for public original problem statements."""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from common import INSTRUCTION_ROOT, DatasetError, read_jsonl, write_json
from normalize_statement import parse_statement_html


RAW_ROOT = INSTRUCTION_ROOT / "data" / "statements" / "raw"
DEFAULT_MANIFEST = INSTRUCTION_ROOT / "data" / "statements" / "pilot_manifest.jsonl"
FETCH_REPORT = INSTRUCTION_ROOT / "reports" / "statement_fetch.json"
MANUAL_REPORT = INSTRUCTION_ROOT / "reports" / "manual_required.json"
USER_AGENT = "PersonalCoder-Dataset-Audit/1.0 (low-rate public statement retrieval)"
BLOCK_MARKERS = (
    "cf-chl-",
    "cloudflare ray id",
    "checking your browser",
    "verify you are human",
    "captcha",
    "验证码",
    "登录后继续",
    "page not found",
    "404 not found",
)


def statement_url(source: str, problem_id: str) -> str:
    if source == "luogu" and re.fullmatch(r"P\d{3,7}", problem_id, re.IGNORECASE):
        return f"https://www.luogu.com.cn/problem/{problem_id.upper()}"
    if source == "codeforces":
        match = re.fullmatch(r"(\d{1,7})([A-Z]\d?)", problem_id, re.IGNORECASE)
        if match:
            if int(match.group(1)) >= 100000:
                return f"https://codeforces.com/gym/{match.group(1)}/problem/{match.group(2).upper()}"
            return f"https://codeforces.com/problemset/problem/{match.group(1)}/{match.group(2).upper()}"
    raise DatasetError(f"unsupported or invalid source/problem ID: {source}/{problem_id}")


def _failure(source: str, problem_id: str, url: str, error: str, manual: bool) -> dict[str, Any]:
    return {
        "success": False,
        "source": source,
        "problem_id": problem_id,
        "detected_problem_id": "",
        "title": "",
        "statement": "",
        "input": "",
        "output": "",
        "constraints": "",
        "samples": [],
        "url": url,
        "fetch_time": datetime.now(timezone.utc).isoformat(),
        "error": error,
        "manual_required": manual,
    }


def fetch_statement(
    source: str,
    problem_id: str,
    *,
    raw_root: Path = RAW_ROOT,
    timeout: float = 30.0,
    max_attempts: int = 2,
) -> dict[str, Any]:
    """Fetch one authentic public statement; never synthesize missing fields."""
    source = source.lower()
    problem_id = problem_id.upper()
    cache_path = raw_root / source / f"{problem_id}.json"
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("success"):
            return {**cached, "_cache_hit": True}
    try:
        url = statement_url(source, problem_id)
    except DatasetError as exc:
        result = _failure(source, problem_id, "", str(exc), True)
        write_json(cache_path, result)
        return result

    page = ""
    error = ""
    manual = False
    for attempt in range(1, max_attempts + 1):
        try:
            request = Request(
                url,
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                    "Accept": "text/html,application/xhtml+xml",
                },
            )
            with urlopen(request, timeout=timeout) as response:
                page = response.read().decode(response.headers.get_content_charset() or "utf-8", "replace")
            break
        except HTTPError as exc:
            error = f"HTTP {exc.code}"
            manual = exc.code in {401, 403, 429}
            if exc.code not in {408, 429, 500, 502, 503, 504}:
                break
        except (URLError, TimeoutError, OSError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        if attempt < max_attempts:
            time.sleep(2.0)
    if not page:
        result = _failure(source, problem_id, url, error or "empty response", manual)
        write_json(cache_path, result)
        return result

    html_path = raw_root / source / f"{problem_id}.html"
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(page, encoding="utf-8")
    lowered = page.lower()
    marker = next((value for value in BLOCK_MARKERS if value in lowered), None)
    if marker:
        result = _failure(source, problem_id, url, f"blocked/error page marker: {marker}", True)
        write_json(cache_path, result)
        return result
    try:
        result = parse_statement_html(source, problem_id, url, page)
        result["fetch_time"] = datetime.now(timezone.utc).isoformat()
        result["manual_required"] = False
    except (DatasetError, ValueError) as exc:
        result = _failure(source, problem_id, url, f"parse error: {exc}", True)
    write_json(cache_path, result)
    return result


def run_manifest(
    manifest: Path,
    delay: float,
    limit: int | None,
    fetch_report: Path = FETCH_REPORT,
    manual_report: Path = MANUAL_REPORT,
) -> dict[str, Any]:
    if delay < 1.0:
        raise DatasetError("request delay must be at least 1 second")
    items = list(read_jsonl(manifest))
    if limit is not None:
        items = items[:limit]
    records = []
    last_request_time = 0.0
    for index, item in enumerate(items, 1):
        cache_path = RAW_ROOT / item["source"] / f"{item['problem_id'].upper()}.json"
        cached_success = False
        if cache_path.is_file():
            try:
                cached_success = bool(json.loads(cache_path.read_text(encoding="utf-8")).get("success"))
            except json.JSONDecodeError:
                cached_success = False
        if not cached_success:
            elapsed = time.monotonic() - last_request_time
            if last_request_time and elapsed < delay:
                time.sleep(delay - elapsed)
            last_request_time = time.monotonic()
        result = fetch_statement(item["source"], item["problem_id"])
        result.pop("_cache_hit", None)
        records.append(result)
        if index % 20 == 0 or index == len(items):
            print(f"Fetched {index}/{len(items)}; success={sum(record['success'] for record in records)}", flush=True)
    compact_records = [
        {
            "source": record["source"],
            "problem_id": record["problem_id"],
            "detected_problem_id": record.get("detected_problem_id", ""),
            "url": record.get("url", ""),
            "success": record["success"],
            "error": record.get("error", ""),
            "manual_required": record.get("manual_required", False),
            "fetch_time": record.get("fetch_time", ""),
            "title_length": len(record.get("title", "")),
            "statement_length": len(record.get("statement", "")),
            "input_length": len(record.get("input", "")),
            "output_length": len(record.get("output", "")),
            "sample_count": len(record.get("samples", [])),
        }
        for record in records
    ]
    summary = {
        "attempted": len(records),
        "fetched": sum(record["success"] for record in records),
        "failed": sum(not record["success"] for record in records),
        "manual_required": sum(record.get("manual_required", False) for record in records),
        "source_distribution": {
            source: {
                "attempted": sum(record["source"] == source for record in records),
                "fetched": sum(record["source"] == source and record["success"] for record in records),
            }
            for source in sorted({record["source"] for record in records})
        },
        "delay_seconds": delay,
        # Full fetched content remains in the ignored raw cache.  Reports are
        # deliberately metadata-only so they are safe and useful to commit.
        "records": compact_records,
    }
    write_json(fetch_report, summary)
    write_json(
        manual_report,
        [record for record in compact_records if record.get("manual_required") or not record["success"]],
    )
    return summary


def reparse_manifest(manifest: Path) -> dict[str, int]:
    counts = {"available_html": 0, "reparsed": 0, "failed": 0}
    for item in read_jsonl(manifest):
        source, problem_id = item["source"], item["problem_id"].upper()
        html_path = RAW_ROOT / source / f"{problem_id}.html"
        if not html_path.is_file():
            continue
        counts["available_html"] += 1
        raw_path = RAW_ROOT / source / f"{problem_id}.json"
        old = json.loads(raw_path.read_text(encoding="utf-8")) if raw_path.is_file() else {}
        try:
            record = parse_statement_html(
                source,
                problem_id,
                statement_url(source, problem_id),
                html_path.read_text(encoding="utf-8"),
            )
            record["fetch_time"] = old.get("fetch_time") or datetime.now(timezone.utc).isoformat()
            record["manual_required"] = False
            counts["reparsed"] += 1
        except (DatasetError, ValueError) as exc:
            record = _failure(source, problem_id, old.get("url", ""), f"parse error: {exc}", True)
            counts["failed"] += 1
        write_json(raw_path, record)
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--delay", type=float, default=1.2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--reparse-cache", action="store_true")
    parser.add_argument("--report", type=Path, default=FETCH_REPORT)
    parser.add_argument("--manual-report", type=Path, default=MANUAL_REPORT)
    args = parser.parse_args()
    if args.reparse_cache:
        try:
            counts = reparse_manifest(args.manifest.resolve())
        except DatasetError as exc:
            parser.error(str(exc))
        print(json.dumps(counts, ensure_ascii=False, indent=2))
        return 0
    try:
        report = run_manifest(
            args.manifest.resolve(),
            args.delay,
            args.limit,
            args.report.resolve(),
            args.manual_report.resolve(),
        )
    except DatasetError as exc:
        parser.error(str(exc))
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Compile and judge one C++ submission against one local problem."""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_TIME_LIMIT_SECONDS = 2.0
DEFAULT_MEMORY_LIMIT_MB = 256
DEFAULT_STDOUT_LIMIT_KB = 1024
COMPILE_TIMEOUT_SECONDS = 30.0
ERROR_TEXT_LIMIT = 4096


class JudgeError(RuntimeError):
    """Raised when the benchmark problem or judge configuration is invalid."""


@dataclass(frozen=True)
class Limits:
    time_seconds: float = DEFAULT_TIME_LIMIT_SECONDS
    memory_mb: int = DEFAULT_MEMORY_LIMIT_MB
    stdout_kb: int = DEFAULT_STDOUT_LIMIT_KB


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return ""
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    try:
        return float(value) if "." in value else int(value)
    except ValueError:
        return value


def load_meta(problem_dir: Path) -> dict[str, Any]:
    """Read the flat key/value subset of YAML used by benchmark metadata."""
    meta_path = problem_dir / "meta.yaml"
    if not meta_path.exists():
        return {}

    meta: dict[str, Any] = {}
    for line_number, raw_line in enumerate(
        meta_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise JudgeError(f"meta.yaml:{line_number}: expected 'key: value'")
        key, value = line.split(":", 1)
        key = key.strip()
        if not key:
            raise JudgeError(f"meta.yaml:{line_number}: empty key")
        meta[key] = _parse_scalar(value)
    return meta


def load_limits(problem_dir: Path) -> Limits:
    meta = load_meta(problem_dir)
    limits = Limits(
        time_seconds=float(
            meta.get("time_limit_seconds", DEFAULT_TIME_LIMIT_SECONDS)
        ),
        memory_mb=int(meta.get("memory_limit_mb", DEFAULT_MEMORY_LIMIT_MB)),
        stdout_kb=int(meta.get("stdout_limit_kb", DEFAULT_STDOUT_LIMIT_KB)),
    )
    if limits.time_seconds <= 0 or limits.memory_mb <= 0 or limits.stdout_kb <= 0:
        raise JudgeError("time, memory, and stdout limits must be positive")
    return limits


def discover_tests(problem_dir: Path) -> list[tuple[str, Path, Path]]:
    tests_dir = problem_dir / "tests"
    if not tests_dir.is_dir():
        raise JudgeError(f"tests directory not found: {tests_dir}")

    cases: list[tuple[str, Path, Path]] = []
    for input_path in sorted(tests_dir.glob("input*.txt")):
        suffix = input_path.name[len("input") :]
        output_path = tests_dir / f"output{suffix}"
        if not output_path.is_file():
            raise JudgeError(f"missing expected output for {input_path.name}")
        cases.append((input_path.stem, input_path, output_path))
    if not cases:
        raise JudgeError(f"no input*.txt testcases found in {tests_dir}")
    return cases


def _truncate(text: str) -> str:
    return text[-ERROR_TEXT_LIMIT:]


def compile_cpp(source: Path, executable: Path) -> tuple[bool, str]:
    command = [
        "g++",
        "-std=c++17",
        "-O2",
        "-pipe",
        str(source),
        "-o",
        str(executable),
    ]
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=COMPILE_TIMEOUT_SECONDS,
            check=False,
        )
    except FileNotFoundError:
        return False, "g++ not found"
    except subprocess.TimeoutExpired:
        return False, f"compilation timed out after {COMPILE_TIMEOUT_SECONDS:g}s"

    if completed.returncode != 0:
        detail = completed.stderr or completed.stdout or "unknown compiler error"
        return False, _truncate(detail.strip())
    return True, ""


def _limit_process(limits: Limits) -> None:
    memory_bytes = limits.memory_mb * 1024 * 1024
    output_bytes = limits.stdout_kb * 1024
    cpu_seconds = max(1, math.ceil(limits.time_seconds))
    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (output_bytes, output_bytes))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))


def run_case(
    executable: Path,
    case_name: str,
    input_path: Path,
    expected_path: Path,
    limits: Limits,
    work_dir: Path,
) -> dict[str, Any]:
    actual_path = work_dir / f"{case_name}.actual"
    stderr_path = work_dir / f"{case_name}.stderr"
    started = time.perf_counter()

    with input_path.open("rb") as stdin_file, actual_path.open(
        "wb"
    ) as stdout_file, stderr_path.open("wb") as stderr_file:
        process = subprocess.Popen(
            [str(executable)],
            stdin=stdin_file,
            stdout=stdout_file,
            stderr=stderr_file,
            cwd=work_dir,
            preexec_fn=lambda: _limit_process(limits),
            start_new_session=True,
        )
        timed_out = False
        try:
            process.wait(timeout=limits.time_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()

    elapsed = time.perf_counter() - started
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
    result: dict[str, Any] = {
        "name": case_name,
        "passed": False,
        "status": "",
        "time": round(elapsed, 6),
        "error": "",
    }

    if timed_out:
        result.update(status="time_limit_exceeded", error="time limit exceeded")
        return result

    if process.returncode != 0:
        if process.returncode == -signal.SIGXFSZ:
            status = "output_limit_exceeded"
            error = "stdout/stderr size limit exceeded"
        else:
            status = "runtime_error"
            error = stderr.strip() or f"process exited with code {process.returncode}"
        result.update(status=status, error=_truncate(error))
        return result

    actual = actual_path.read_bytes()
    expected = expected_path.read_bytes()
    if actual != expected:
        result.update(status="wrong_answer", error="output differs from expected")
        return result

    result.update(passed=True, status="accepted")
    return result


def judge(source: str | Path, problem: str | Path) -> dict[str, Any]:
    """Judge one source file and return a JSON-serializable result."""
    source_path = Path(source).resolve()
    problem_dir = Path(problem).resolve()
    if not source_path.is_file():
        raise JudgeError(f"source file not found: {source_path}")
    if not problem_dir.is_dir():
        raise JudgeError(f"problem directory not found: {problem_dir}")

    limits = load_limits(problem_dir)
    testcases = discover_tests(problem_dir)
    with tempfile.TemporaryDirectory(prefix="personalcoder-judge-") as temp_name:
        temp_dir = Path(temp_name)
        executable = temp_dir / "submission"
        compile_ok, compile_error = compile_cpp(source_path, executable)
        if not compile_ok:
            return {
                "compile": False,
                "tests": len(testcases),
                "passed": 0,
                "ac": False,
                "time": 0.0,
                "error": compile_error,
                "cases": [],
            }

        case_results = [
            run_case(executable, name, input_path, output_path, limits, temp_dir)
            for name, input_path, output_path in testcases
        ]

    passed = sum(case["passed"] for case in case_results)
    first_error = next((case["error"] for case in case_results if case["error"]), "")
    return {
        "compile": True,
        "tests": len(case_results),
        "passed": passed,
        "ac": passed == len(case_results),
        "time": round(sum(case["time"] for case in case_results), 6),
        "error": first_error,
        "cases": case_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="C++ source file to judge")
    parser.add_argument("problem", type=Path, help="problem directory")
    args = parser.parse_args()

    try:
        result = judge(args.source, args.problem)
    except JudgeError as exc:
        result = {
            "compile": False,
            "tests": 0,
            "passed": 0,
            "ac": False,
            "time": 0.0,
            "error": str(exc),
            "cases": [],
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

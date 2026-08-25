#!/usr/bin/env python3
"""Analyze C++ style in the raw PersonalCoder dataset without changing it."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "algorithm"
REPORT_PATH = PROJECT_ROOT / "outputs" / "style_report.json"
FUNCTION_NAMES = (
    "dfs", "bfs", "find", "join", "build", "update", "query", "solve",
    "tarjan", "dijkstra",
)
CONTAINERS = {
    "vector": re.compile(r"\bvector\s*<"),
    "static_array": re.compile(
        r"\b(?:bool|char|short|int|long|float|double|signed|unsigned|ll|i64|u64)"
        r"(?:\s+long)?\s+[A-Za-z_]\w*\s*\["
    ),
    "map": re.compile(r"(?<!unordered_)\bmap\s*<"),
    "unordered_map": re.compile(r"\bunordered_map\s*<"),
    "set": re.compile(r"(?<!unordered_)\b(?:multi)?set\s*<"),
    "unordered_set": re.compile(r"\bunordered_(?:multi)?set\s*<"),
    "queue": re.compile(r"\b(?:priority_)?queue\s*<"),
    "stack": re.compile(r"\bstack\s*<"),
    "deque": re.compile(r"\bdeque\s*<"),
}
CPP_KEYWORDS = {
    "alignas", "alignof", "and", "auto", "bool", "break", "case", "catch",
    "char", "class", "const", "continue", "default", "delete", "do", "double",
    "else", "enum", "explicit", "extern", "false", "float", "for", "friend",
    "goto", "if", "inline", "int", "long", "main", "namespace", "new", "noexcept",
    "not", "nullptr", "operator", "or", "private", "protected", "public", "register",
    "return", "short", "signed", "sizeof", "static", "struct", "switch", "template",
    "this", "throw", "true", "try", "typedef", "typename", "union", "unsigned",
    "using", "virtual", "void", "volatile", "while", "std", "cin", "cout", "endl",
}


def without_comments_and_literals(source: str) -> str:
    """Mask comments and quoted literals while retaining newlines."""
    pattern = re.compile(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'', re.S)
    return pattern.sub(lambda match: re.sub(r"[^\n]", " ", match.group()), source)


def percentage(count: int, total: int) -> float:
    return round(count * 100 / total, 2) if total else 0.0


def top_level_code(code: str) -> str:
    """Keep lines beginning at brace depth zero for global declaration checks."""
    depth = 0
    lines: list[str] = []
    for line in code.splitlines():
        if depth == 0:
            lines.append(line)
        depth = max(0, depth + line.count("{") - line.count("}"))
    return "\n".join(lines)


def occurrence_metric(file_count: int, occurrences: int, total: int) -> dict[str, int | float]:
    return {
        "files": file_count,
        "file_percentage": percentage(file_count, total),
        "occurrences": occurrences,
    }


def indentation_style(source: str) -> str:
    indented = [line for line in source.splitlines() if line[:1] in {" ", "\t"} and line.strip()]
    tab_lines = sum(line.startswith("\t") for line in indented)
    four_space_lines = sum(
        len(line) - len(line.lstrip(" ")) >= 4
        and (len(line) - len(line.lstrip(" "))) % 4 == 0
        for line in indented
    )
    other_lines = len(indented) - tab_lines - four_space_lines
    if tab_lines and tab_lines > four_space_lines and tab_lines > other_lines:
        return "tab"
    if four_space_lines and four_space_lines >= tab_lines and four_space_lines >= other_lines:
        return "4_spaces"
    return "other"


def main_form(code: str) -> str:
    match = re.search(r"\b(signed|int|int32_t|void)\s+main\s*\(([^)]*)\)", code)
    if not match:
        return "other_or_missing"
    return_type, arguments = match.groups()
    arguments = re.sub(r"\s+", " ", arguments.strip())
    if not arguments:
        arguments = "empty"
    elif arguments == "void":
        arguments = "void"
    else:
        arguments = "arguments"
    return f"{return_type}_main_{arguments}"


def analyze_group(items: list[tuple[Path, str]]) -> dict[str, object]:
    total = len(items)
    feature_files: Counter[str] = Counter()
    feature_occurrences: Counter[str] = Counter()
    container_files: Counter[str] = Counter()
    container_occurrences: Counter[str] = Counter()
    function_files: Counter[str] = Counter()
    function_occurrences: Counter[str] = Counter()
    variables: Counter[str] = Counter()
    main_forms: Counter[str] = Counter()
    indent_styles: Counter[str] = Counter()
    brace_styles: Counter[str] = Counter()

    patterns = {
        "bits_stdcpp_include": re.compile(r"^\s*#\s*include\s*<bits/stdc\+\+\.h>", re.M),
        "using_namespace_std": re.compile(r"\busing\s+namespace\s+std\s*;"),
        "ios_sync_with_stdio_0": re.compile(r"\bios::sync_with_stdio\s*\(\s*(?:0|false)\s*\)"),
        "cin_tie_0": re.compile(r"\bcin\.tie\s*\(\s*(?:0|nullptr)\s*\)"),
        "return_0": re.compile(r"\breturn\s+0\s*;"),
    }
    global_const_pattern = re.compile(
        r"^(?:inline\s+|static\s+)?const(?:expr)?\s+(?:unsigned\s+)?(?:int|long\s+long|ll|size_t)"
        r"\s+(?:MAX|N|M)[A-Z0-9_]*\b",
        re.M,
    )

    for _, source in items:
        code = without_comments_and_literals(source)
        for name, pattern in patterns.items():
            count = len(pattern.findall(code))
            feature_occurrences[name] += count
            feature_files[name] += bool(count)
        global_const_count = len(global_const_pattern.findall(top_level_code(code)))
        feature_occurrences["global_const_max"] += global_const_count
        feature_files["global_const_max"] += bool(global_const_count)

        line_comments = len(re.findall(r"//[^\n]*", source))
        block_comments = len(re.findall(r"/\*.*?\*/", source, re.S))
        for name, count in (("line_comments", line_comments), ("block_comments", block_comments)):
            feature_occurrences[name] += count
            feature_files[name] += bool(count)

        for name, pattern in CONTAINERS.items():
            count = len(pattern.findall(code))
            container_occurrences[name] += count
            container_files[name] += bool(count)

        for name in FUNCTION_NAMES:
            count = len(re.findall(rf"\b{re.escape(name)}\s*\(", code))
            function_occurrences[name] += count
            function_files[name] += bool(count)

        identifiers = re.findall(r"\b[A-Za-z_]\w{0,3}\b", code)
        variables.update(
            identifier for identifier in identifiers
            if identifier not in CPP_KEYWORDS and identifier not in FUNCTION_NAMES
        )
        main_forms[main_form(code)] += 1
        indent_styles[indentation_style(source)] += 1

        same_line = sum(
            bool(line.strip()) and not line.lstrip().startswith("#") and "{" in line and line.split("{", 1)[0].strip() != ""
            for line in code.splitlines()
        )
        next_line = sum(line.strip() == "{" for line in code.splitlines())
        brace_styles["same_line"] += same_line
        brace_styles["next_line"] += next_line

    metric_names = (
        "bits_stdcpp_include", "using_namespace_std", "ios_sync_with_stdio_0",
        "cin_tie_0", "global_const_max", "return_0", "line_comments", "block_comments",
    )
    return {
        "nonempty_cpp_files": total,
        "features": {
            name: occurrence_metric(feature_files[name], feature_occurrences[name], total)
            for name in metric_names
        },
        "containers": {
            name: occurrence_metric(container_files[name], container_occurrences[name], total)
            for name in CONTAINERS
        },
        "main_forms": dict(main_forms.most_common()),
        "common_functions": {
            name: occurrence_metric(function_files[name], function_occurrences[name], total)
            for name in FUNCTION_NAMES
        },
        "top_short_variable_identifiers": [
            {"name": name, "occurrences": count} for name, count in variables.most_common(25)
        ],
        "indentation_style_files": dict(indent_styles),
        "brace_style_occurrences": dict(brace_styles),
    }


def main() -> int:
    if not DATASET_ROOT.is_dir():
        print(f"ERROR: Dataset directory not found: {DATASET_ROOT}")
        return 1

    all_items: list[tuple[Path, str]] = []
    for path in sorted(DATASET_ROOT.rglob("*.cpp")):
        if not path.is_file() or path.stat().st_size == 0:
            continue
        source = path.read_text(encoding="utf-8", errors="replace")
        if source:
            all_items.append((path, source))

    template_items: list[tuple[Path, str]] = []
    non_template_items: list[tuple[Path, str]] = []
    for item in all_items:
        target = template_items if "templates" in item[0].relative_to(DATASET_ROOT).parts else non_template_items
        target.append(item)
    report = {
        "dataset_root": DATASET_ROOT.relative_to(PROJECT_ROOT).as_posix(),
        "methodology": {
            "scope": "All nonempty .cpp files; regex-based approximate style analysis.",
            "ratios": "File percentages use each group's nonempty .cpp count as denominator.",
            "duplicate_files": "Files are analyzed independently; duplicates are not removed.",
        },
        "groups": {
            "all": analyze_group(all_items),
            "templates": analyze_group(template_items),
            "non_templates": analyze_group(non_template_items),
        },
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    REPORT_PATH.write_text(f"{rendered}\n", encoding="utf-8")
    print(rendered)
    print(f"\nReport saved to: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

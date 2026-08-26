#!/usr/bin/env python3
"""Parse fetched public problem pages and normalize them to faithful Markdown."""

from __future__ import annotations

import argparse
import html
import json
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable

from common import INSTRUCTION_ROOT, DatasetError, read_jsonl


RAW_ROOT = INSTRUCTION_ROOT / "data" / "statements" / "raw"
NORMALIZED_ROOT = INSTRUCTION_ROOT / "data" / "statements" / "normalized"
DEFAULT_MANIFEST = INSTRUCTION_ROOT / "data" / "statements" / "pilot_manifest.jsonl"
BLOCK_TAGS = {
    "article", "section", "div", "p", "h1", "h2", "h3", "h4", "li", "tr", "pre"
}


@dataclass
class Node:
    tag: str
    attrs: dict[str, str] = field(default_factory=dict)
    children: list[Any] = field(default_factory=list)
    parent: "Node | None" = None

    def classes(self) -> set[str]:
        return set(self.attrs.get("class", "").split())

    def descendants(self) -> Iterable["Node"]:
        for child in self.children:
            if isinstance(child, Node):
                yield child
                yield from child.descendants()

    def find(self, tag: str | None = None, class_name: str | None = None) -> "Node | None":
        for node in self.descendants():
            if tag is not None and node.tag != tag:
                continue
            if class_name is not None and class_name not in node.classes():
                continue
            return node
        return None

    def find_all(self, tag: str | None = None, class_name: str | None = None) -> list["Node"]:
        return [
            node
            for node in self.descendants()
            if (tag is None or node.tag == tag)
            and (class_name is None or class_name in node.classes())
        ]


class TreeParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = Node("root")
        self.stack = [self.root]

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        node = Node(tag.lower(), {key: value or "" for key, value in attrs}, parent=self.stack[-1])
        self.stack[-1].children.append(node)
        if tag.lower() not in {"br", "img", "meta", "link", "input", "hr", "source", "wbr"}:
            self.stack.append(node)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if self.stack[-1].tag == tag.lower():
            self.stack.pop()

    def handle_endtag(self, tag: str) -> None:
        target = tag.lower()
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index].tag == target:
                del self.stack[index:]
                return

    def handle_data(self, data: str) -> None:
        if self.stack[-1].tag not in {"script", "style"} or (
            self.stack[-1].tag == "script"
            and self.stack[-1].attrs.get("type") == "application/json"
        ):
            self.stack[-1].children.append(data)


def _raw_text(node: Node | str, in_pre: bool = False) -> str:
    if isinstance(node, str):
        return node
    if node.tag in {"script", "style", "svg"}:
        return ""
    if node.tag == "br":
        return "\n"
    pre = in_pre or node.tag == "pre"
    body = "".join(_raw_text(child, pre) for child in node.children)
    if node.tag == "img" and node.attrs.get("alt"):
        body = node.attrs["alt"]
    if node.tag in BLOCK_TAGS and not pre:
        body = f"\n{body}\n"
    if node.tag in {"div", "p"} and pre:
        body = f"{body}\n"
    return body


def node_text(node: Node, drop_first_heading: bool = False) -> str:
    text = html.unescape(_raw_text(node))
    lines = []
    blank = False
    for raw in text.replace("\r", "").split("\n"):
        line = re.sub(r"[ \t]+", " ", raw).strip()
        if line:
            lines.append(line)
            blank = False
        elif lines and not blank:
            lines.append("")
            blank = True
    while lines and not lines[-1]:
        lines.pop()
    if drop_first_heading and lines:
        lines = lines[1:]
        while lines and not lines[0]:
            lines.pop(0)
    return "\n".join(lines).strip()


def _section_body(section: Node) -> str:
    h2 = next((child for child in section.children if isinstance(child, Node) and child.tag in {"h2", "h3"}), None)
    clone = Node("section", children=[child for child in section.children if child is not h2])
    return node_text(clone)


def parse_luogu(page: str, source: str, problem_id: str, url: str) -> dict[str, Any]:
    parser = TreeParser()
    parser.feed(page)
    context_node = next(
        (
            node
            for node in parser.root.descendants()
            if node.tag == "script" and node.attrs.get("id") == "lentille-context"
        ),
        None,
    )
    if context_node is not None:
        raw_context = "".join(
            child for child in context_node.children if isinstance(child, str)
        ).strip()
        try:
            context = json.loads(raw_context)
            problem = context["data"]["problem"]
            content = problem.get("content") or problem.get("contenu") or {}
            samples = [
                {"input": str(pair[0]), "output": str(pair[1])}
                for pair in content.get("samples", problem.get("content", {}).get("samples", []))
                if isinstance(pair, list) and len(pair) >= 2
            ]
            # Current Luogu stores samples beside content rather than inside it.
            if not samples:
                samples = [
                    {"input": str(pair[0]), "output": str(pair[1])}
                    for pair in problem.get("content", {}).get("samples", [])
                    if isinstance(pair, list) and len(pair) >= 2
                ]
            if not samples:
                samples = [
                    {"input": str(pair[0]), "output": str(pair[1])}
                    for pair in problem.get("samples", [])
                    if isinstance(pair, list) and len(pair) >= 2
                ]
            pid = str(problem.get("pid") or problem_id).upper()
            return {
                "success": True,
                "source": source,
                "problem_id": problem_id,
                "detected_problem_id": pid,
                "title": f"{pid} {content.get('name') or problem.get('name') or ''}".strip(),
                "statement": str(content.get("description") or "").strip(),
                "input": str(content.get("formatI") or "").strip(),
                "output": str(content.get("formatO") or "").strip(),
                "constraints": str(content.get("hint") or "").strip(),
                "samples": samples,
                "url": url,
                "error": "",
            }
        except (KeyError, TypeError, json.JSONDecodeError):
            pass
    article = parser.root.find("article")
    if article is None:
        raise DatasetError("Luogu article not found")
    title_node = article.find("h1")
    title = node_text(title_node) if title_node else ""
    fields = {"statement": "", "input": "", "output": "", "constraints": ""}
    samples: list[dict[str, str]] = []
    for section in article.find_all("section"):
        heading_node = section.find("h2") or section.find("h3")
        heading = node_text(heading_node) if heading_node else ""
        body = _section_body(section)
        if "题目描述" in heading:
            fields["statement"] = body
        elif "输入格式" in heading:
            fields["input"] = body
        elif "输出格式" in heading:
            fields["output"] = body
        elif any(key in heading for key in ("说明", "提示", "数据范围", "约束")):
            fields["constraints"] = body
        elif "样例" in heading:
            pre_values = [node_text(node) for node in section.find_all("pre")]
            for index in range(0, len(pre_values), 2):
                samples.append(
                    {
                        "input": pre_values[index],
                        "output": pre_values[index + 1] if index + 1 < len(pre_values) else "",
                    }
                )
    return {
        "success": True,
        "source": source,
        "problem_id": problem_id,
        "detected_problem_id": title.split(maxsplit=1)[0].upper() if title else "",
        "title": title,
        **fields,
        "samples": samples,
        "url": url,
        "error": "",
    }


def _class_section(problem: Node, class_name: str) -> Node | None:
    return problem.find("div", class_name)


def parse_codeforces(page: str, source: str, problem_id: str, url: str) -> dict[str, Any]:
    parser = TreeParser()
    parser.feed(page)
    problem = parser.root.find("div", "problem-statement")
    if problem is None:
        raise DatasetError("Codeforces problem statement not found")
    title_node = problem.find("div", "title")
    title = node_text(title_node) if title_node else ""
    input_node = _class_section(problem, "input-specification")
    output_node = _class_section(problem, "output-specification")
    sample_node = _class_section(problem, "sample-tests")
    note_node = _class_section(problem, "note")
    description_parts = []
    for child in problem.children:
        if not isinstance(child, Node) or child.tag != "div":
            continue
        classes = child.classes()
        if classes & {"header", "input-specification", "output-specification", "sample-tests", "note"}:
            continue
        value = node_text(child)
        if value:
            description_parts.append(value)
    samples = []
    if sample_node:
        inputs = [node_text(node, drop_first_heading=True) for node in sample_node.find_all("div", "input")]
        outputs = [node_text(node, drop_first_heading=True) for node in sample_node.find_all("div", "output")]
        for index, input_value in enumerate(inputs):
            samples.append(
                {"input": input_value, "output": outputs[index] if index < len(outputs) else ""}
            )
    match = re.match(r"\s*([A-Z]\d?)\s*\.", title, re.IGNORECASE)
    contest = re.match(r"(\d+)", problem_id)
    detected_id = f"{contest.group(1)}{match.group(1).upper()}" if match and contest else ""
    return {
        "success": True,
        "source": source,
        "problem_id": problem_id,
        "detected_problem_id": detected_id,
        "title": title,
        "statement": "\n\n".join(description_parts),
        "input": node_text(input_node, drop_first_heading=True) if input_node else "",
        "output": node_text(output_node, drop_first_heading=True) if output_node else "",
        "constraints": node_text(note_node, drop_first_heading=True) if note_node else "",
        "samples": samples,
        "url": url,
        "error": "",
    }


def parse_statement_html(source: str, problem_id: str, url: str, page: str) -> dict[str, Any]:
    if source == "luogu":
        return parse_luogu(page, source, problem_id, url)
    if source == "codeforces":
        return parse_codeforces(page, source, problem_id, url)
    raise DatasetError(f"unsupported statement source: {source}")


def normalize_record(record: dict[str, Any]) -> str:
    sections = [
        f"# {record['title'].strip()}",
        "## 题目描述\n\n" + record["statement"].strip(),
        "## 输入格式\n\n" + record["input"].strip(),
        "## 输出格式\n\n" + record["output"].strip(),
    ]
    if str(record.get("constraints") or "").strip():
        sections.append("## 数据范围与说明\n\n" + record["constraints"].strip())
    for index, sample in enumerate(record.get("samples") or [], 1):
        sections.append(f"## 样例输入 #{index}\n\n```text\n{sample.get('input','').strip()}\n```")
        sections.append(f"## 样例输出 #{index}\n\n```text\n{sample.get('output','').strip()}\n```")
    return "\n\n".join(sections).strip() + "\n"


def normalize_manifest(manifest: Path) -> dict[str, int]:
    counts = {"attempted": 0, "normalized": 0, "missing_raw": 0, "failed_raw": 0}
    for item in read_jsonl(manifest):
        counts["attempted"] += 1
        source, problem_id = item["source"], item["problem_id"]
        raw_path = RAW_ROOT / source / f"{problem_id}.json"
        if not raw_path.is_file():
            counts["missing_raw"] += 1
            continue
        record = json.loads(raw_path.read_text(encoding="utf-8"))
        if not record.get("success"):
            counts["failed_raw"] += 1
            continue
        destination = NORMALIZED_ROOT / source / f"{problem_id}.md"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(normalize_record(record), encoding="utf-8")
        counts["normalized"] += 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    try:
        counts = normalize_manifest(args.manifest.resolve())
    except DatasetError as exc:
        parser.error(str(exc))
    print(json.dumps(counts, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

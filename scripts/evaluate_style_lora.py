#!/usr/bin/env python3
"""Compare the local base model with the first Style LoRA adapter."""

from __future__ import annotations

import gc
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path("/data/PersonalCoder/model")
ADAPTER_PATH = Path("/data/PersonalCoder/checkpoints/style_lora_v1/final_adapter")
OUTPUT_ROOT = ROOT / "outputs" / "style_eval"
REPORT_PATH = OUTPUT_ROOT / "report.json"

PROMPTS = [
    ("union_find", "实现并查集。请使用 C++ 实现，只输出完整代码。"),
    ("dijkstra", "实现 Dijkstra。请使用 C++ 实现，只输出完整代码。"),
    ("tarjan_scc", "实现 Tarjan SCC。请使用 C++ 实现，只输出完整代码。"),
    ("trie", "实现 Trie。请使用 C++ 实现，只输出完整代码。"),
    (
        "segment_tree",
        "实现线段树，支持区间加和区间和查询。请使用 C++ 实现，只输出完整代码。",
    ),
    (
        "heavy_light_decomposition",
        "实现树链剖分，支持路径查询。请使用 C++ 实现，只输出完整代码。",
    ),
    ("fenwick_tree", "实现树状数组。请使用 C++ 实现，只输出完整代码。"),
    ("bfs_shortest_path", "实现 BFS 最短路。请使用 C++ 实现，只输出完整代码。"),
]

GENERATION_CONFIG = {
    "do_sample": False,
    "temperature": 1.0,
    "top_p": 1.0,
    "max_new_tokens": 1024,
}

BOOLEAN_METRICS = [
    "bits_stdcpp",
    "using_namespace_std",
    "global_max_constant",
    "static_array",
    "vector",
    "int_main",
    "signed_main",
    "return_0",
    "ios_sync_with_stdio",
    "cin_tie",
    "extra_explanation",
    "code_only",
]


def fail(message: str) -> None:
    print(f"错误：{message}", file=sys.stderr)
    raise SystemExit(1)


def sanity_check() -> None:
    if not MODEL_PATH.is_dir():
        fail(f"本地模型目录不存在：{MODEL_PATH}")
    if not ADAPTER_PATH.is_dir():
        fail(f"LoRA adapter 目录不存在：{ADAPTER_PATH}")
    if not torch.cuda.is_available():
        fail("CUDA 不可用，无法执行 4-bit 模型评估")
    if subprocess.run(["g++", "--version"], capture_output=True).returncode != 0:
        fail("未找到可用的 g++ 编译器")


def extract_code(raw_text: str) -> tuple[str, bool, bool]:
    fenced = list(
        re.finditer(
            r"```(?:cpp|c\+\+|cc|cxx)?\s*\n?(.*?)```",
            raw_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
    )
    if fenced:
        selected = fenced[0]
        outside = (raw_text[: selected.start()] + raw_text[selected.end() :]).strip()
        return selected.group(1).strip(), bool(outside), not outside

    code = raw_text.strip()
    code_like_start = bool(
        re.match(
            r"\s*(?:#\s*include|#\s*define|using\s+namespace|//|/\*|"
            r"(?:int|signed|int32_t)\s+main\s*\()",
            code,
        )
    )
    return code, not code_like_start, code_like_start


def style_metrics(code: str, extra_explanation: bool, code_only: bool) -> dict[str, Any]:
    line_comments = len(re.findall(r"//[^\n]*", code))
    block_comments = len(re.findall(r"/\*.*?\*/", code, flags=re.DOTALL))
    global_max_pattern = re.compile(
        r"^\s*(?:(?:static|inline)\s+)?(?:const|constexpr)\s+"
        r"(?:unsigned\s+)?(?:int|long\s+long|size_t)\s+"
        r"(?:MAXN|MAXM|MAX_[A-Z0-9_]+|MAX[A-Z0-9_]*)\b",
        flags=re.MULTILINE,
    )
    static_array_pattern = re.compile(
        r"^\s*(?:(?:static|const|constexpr)\s+)*"
        r"(?:unsigned\s+)?(?:int|long\s+long|ll|char|bool|double|float)\s+"
        r"[A-Za-z_]\w*\s*\[[^\]]+\]",
        flags=re.MULTILINE,
    )
    return {
        "bits_stdcpp": bool(re.search(r"#\s*include\s*<bits/stdc\+\+\.h>", code)),
        "using_namespace_std": bool(re.search(r"\busing\s+namespace\s+std\s*;", code)),
        "global_max_constant": bool(global_max_pattern.search(code)),
        "static_array": bool(static_array_pattern.search(code)),
        "vector": bool(re.search(r"\b(?:std::)?vector\s*<", code)),
        "int_main": bool(re.search(r"\bint\s+main\s*\(", code)),
        "signed_main": bool(re.search(r"\bsigned\s+main\s*\(", code)),
        "return_0": bool(re.search(r"\breturn\s+0\s*;", code)),
        "ios_sync_with_stdio": bool(re.search(r"\bios::sync_with_stdio\s*\(", code)),
        "cin_tie": bool(re.search(r"\bcin\.tie\s*\(", code)),
        "line_comment_count": line_comments,
        "block_comment_count": block_comments,
        "extra_explanation": extra_explanation,
        "code_only": code_only,
    }


def compile_code(code: str, stem: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="personalcoder_style_eval_") as tmp:
        source = Path(tmp) / f"{stem}.cpp"
        source.write_text(code, encoding="utf-8")
        try:
            result = subprocess.run(
                ["g++", "-std=c++17", "-fsyntax-only", str(source)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stderr": result.stderr[-4000:],
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "returncode": None, "stderr": "g++ timeout (30s)"}


def load_model(adapter: bool):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        quantization_config=quantization_config,
        device_map={"": 0},
        dtype=torch.float16,
    )
    model = (
        PeftModel.from_pretrained(base, ADAPTER_PATH, is_trainable=False, local_files_only=True)
        if adapter
        else base
    )
    model.eval()
    return model, base


def generate_one(model, tokenizer, prompt: str, seed: int) -> str:
    messages = [{"role": "user", "content": prompt}]
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(rendered, return_tensors="pt").to("cuda")
    torch.manual_seed(seed)
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            **GENERATION_CONFIG,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_ids = output_ids[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    style: dict[str, Any] = {}
    for metric in BOOLEAN_METRICS:
        count = sum(bool(item["style"][metric]) for item in results)
        style[metric] = {"count": count, "rate": count / total if total else 0.0}
    style["line_comment_count"] = sum(
        item["style"]["line_comment_count"] for item in results
    )
    style["block_comment_count"] = sum(
        item["style"]["block_comment_count"] for item in results
    )
    compiled = sum(item["compilation"]["success"] for item in results)
    return {
        "sample_count": total,
        "style": style,
        "compile": {
            "success_count": compiled,
            "total": total,
            "rate": compiled / total if total else 0.0,
        },
    }


def evaluate_model(label: str, use_adapter: bool, tokenizer) -> dict[str, Any]:
    output_dir = OUTPUT_ROOT / label
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n加载 {label} 模型……", flush=True)
    model, base = load_model(use_adapter)
    results = []
    try:
        for index, (prompt_id, prompt) in enumerate(PROMPTS):
            print(f"[{label}] {index + 1}/{len(PROMPTS)} {prompt_id}", flush=True)
            raw_text = generate_one(model, tokenizer, prompt, seed=42 + index)
            output_path = output_dir / f"{prompt_id}.txt"
            output_path.write_text(raw_text + "\n", encoding="utf-8")
            code, extra_explanation, code_only = extract_code(raw_text)
            results.append(
                {
                    "id": prompt_id,
                    "prompt": prompt,
                    "output_file": str(output_path.relative_to(ROOT)),
                    "style": style_metrics(code, extra_explanation, code_only),
                    "compilation": compile_code(code, prompt_id),
                }
            )
    finally:
        del model
        del base
        gc.collect()
        torch.cuda.empty_cache()
    return {"summary": aggregate(results), "results": results}


def main() -> None:
    sanity_check()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

    models = {
        "base": evaluate_model("base", False, tokenizer),
        "lora512": evaluate_model("lora512", True, tokenizer),
    }
    comparison = {}
    for metric in BOOLEAN_METRICS:
        base_rate = models["base"]["summary"]["style"][metric]["rate"]
        lora_rate = models["lora512"]["summary"]["style"][metric]["rate"]
        comparison[metric] = {
            "base_rate": base_rate,
            "lora512_rate": lora_rate,
            "difference_percentage_points": (lora_rate - base_rate) * 100,
        }

    report = {
        "base_model": str(MODEL_PATH),
        "adapter": str(ADAPTER_PATH),
        "generation_config": GENERATION_CONFIG,
        "prompts": [{"id": item[0], "text": item[1]} for item in PROMPTS],
        "models": models,
        "style_rate_comparison": comparison,
    }
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("\n评估汇总")
    for label in ("base", "lora512"):
        summary = models[label]["summary"]
        compile_summary = summary["compile"]
        print(
            f"{label}: compile={compile_summary['success_count']}/"
            f"{compile_summary['total']} ({compile_summary['rate']:.1%})"
        )
        print(json.dumps(summary["style"], ensure_ascii=False, indent=2))
    print(f"报告已保存：{REPORT_PATH}")


if __name__ == "__main__":
    main()

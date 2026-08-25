#!/usr/bin/env python3
"""Offline, deterministic style comparison for base and Style-LoRA-1536."""

from __future__ import annotations

import gc
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path("/data/PersonalCoder/model")
ADAPTER_PATH = Path("/data/PersonalCoder/checkpoints/rtx4060/style_lora_1536_v1")
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "style_eval_1536"
MAX_NEW_TOKENS = 512
GENERATION_CONFIG = {
    "do_sample": False,
    "max_new_tokens": MAX_NEW_TOKENS,
    "temperature": None,
    "top_p": None,
}
TASKS = [
    ("01_dsu", "实现并查集。请使用 C++ 实现，只输出完整代码。"),
    ("02_dijkstra", "实现 Dijkstra。请使用 C++ 实现，只输出完整代码。"),
    ("03_tarjan_scc", "实现 Tarjan SCC。请使用 C++ 实现，只输出完整代码。"),
    ("04_trie", "实现 Trie。请使用 C++ 实现，只输出完整代码。"),
    ("05_segment_tree", "实现线段树，支持区间加和区间和查询。请使用 C++ 实现，只输出完整代码。"),
    ("06_hld", "实现树链剖分，支持路径查询。请使用 C++ 实现，只输出完整代码。"),
    ("07_fenwick", "实现树状数组。请使用 C++ 实现，只输出完整代码。"),
    ("08_bfs", "实现 BFS 最短路。请使用 C++ 实现，只输出完整代码。"),
]
METRIC_KEYS = (
    "include_bits_stdcpp",
    "using_namespace_std",
    "maxn_or_maxm",
    "static_array",
    "vector",
    "int_main",
    "signed_main",
    "return_0",
    "ios_sync_with_stdio",
    "cin_tie",
    "line_comment",
    "block_comment",
    "extra_natural_language",
    "code_only",
)


def extract_code(text: str) -> tuple[str, bool, str]:
    stripped = text.strip()
    fenced = re.search(r"```(?:cpp|c\+\+|cc|C\+\+)?\s*\n?(.*?)```", stripped, re.DOTALL)
    if fenced:
        code = fenced.group(1).strip()
        outside = (stripped[: fenced.start()] + stripped[fenced.end() :]).strip()
        return code, True, outside
    return stripped, False, ""


def style_metrics(raw_text: str, code: str, had_fence: bool, outside: str) -> dict[str, bool]:
    starts_like_code = bool(re.match(r"\s*(?:#include|using\s+namespace|typedef|struct|class|const\b)", code))
    trailing_text = code[code.rfind("}") + 1 :].strip() if "}" in code else ""
    extra_language = had_fence or bool(outside) or not starts_like_code or bool(trailing_text)
    return {
        "include_bits_stdcpp": bool(re.search(r"#\s*include\s*<bits/stdc\+\+\.h>", code)),
        "using_namespace_std": bool(re.search(r"\busing\s+namespace\s+std\s*;", code)),
        "maxn_or_maxm": bool(re.search(r"\bMAX[NM]\b", code)),
        "static_array": bool(re.search(r"\b(?:bool|char|short|int|long\s+long|float|double)\s+[A-Za-z_]\w*\s*\[[^\]]+\]", code)),
        "vector": bool(re.search(r"\bvector\s*<", code)),
        "int_main": bool(re.search(r"\bint\s+main\s*\(", code)),
        "signed_main": bool(re.search(r"\bsigned\s+main\s*\(", code)),
        "return_0": bool(re.search(r"\breturn\s+0\s*;", code)),
        "ios_sync_with_stdio": bool(re.search(r"\bios\s*::\s*sync_with_stdio\s*\(", code)),
        "cin_tie": bool(re.search(r"\bcin\s*\.\s*tie\s*\(", code)),
        "line_comment": bool(re.search(r"//", code)),
        "block_comment": bool(re.search(r"/\*.*?\*/", code, re.DOTALL)),
        "extra_natural_language": extra_language,
        "code_only": not extra_language and raw_text.strip() == code,
    }


def compile_code(code_path: Path) -> dict[str, object]:
    result = subprocess.run(
        ["g++", "-std=c++17", "-fsyntax-only", str(code_path)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
        check=False,
    )
    return {
        "success": result.returncode == 0,
        "returncode": result.returncode,
        "stderr": result.stderr[-4000:],
        "command": ["g++", "-std=c++17", "-fsyntax-only", str(code_path)],
    }


def generate_suite(model: object, tokenizer: object, label: str) -> list[dict[str, object]]:
    output_dir = OUTPUT_ROOT / label
    output_dir.mkdir(parents=True, exist_ok=False)
    records = []
    device = next(model.parameters()).device
    model.eval()
    for task_id, prompt in TASKS:
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = generated[0, inputs.input_ids.shape[1] :]
        raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        code, had_fence, outside = extract_code(raw_text)
        raw_path = output_dir / f"{task_id}.txt"
        code_path = output_dir / f"{task_id}.cpp"
        raw_path.write_text(raw_text + "\n", encoding="utf-8")
        code_path.write_text(code + "\n", encoding="utf-8")
        compile_result = compile_code(code_path)
        record = {
            "task": task_id,
            "prompt": prompt,
            "generated_tokens": int(new_tokens.numel()),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "raw_output_path": str(raw_path.relative_to(PROJECT_ROOT)),
            "compiled_code_path": str(code_path.relative_to(PROJECT_ROOT)),
            "style": style_metrics(raw_text, code, had_fence, outside),
            "compile": compile_result,
        }
        records.append(record)
        print(f"{label} {task_id}: compile={compile_result['success']}", flush=True)
    return records


def summarize(records: list[dict[str, object]]) -> dict[str, object]:
    count = len(records)
    metric_counts = {
        key: sum(bool(record["style"][key]) for record in records) for key in METRIC_KEYS
    }
    compiled = sum(bool(record["compile"]["success"]) for record in records)
    return {
        "samples": count,
        "metric_counts": metric_counts,
        "metric_rates": {key: round(value / count, 4) for key, value in metric_counts.items()},
        "compiled": compiled,
        "compile_rate": round(compiled / count, 4),
    }


def main() -> int:
    if OUTPUT_ROOT.exists():
        print(f"ERROR: Refusing to overwrite evaluation output: {OUTPUT_ROOT}", file=sys.stderr)
        return 1
    if not MODEL_PATH.is_dir() or not ADAPTER_PATH.is_dir():
        print("ERROR: Local model or adapter is missing", file=sys.stderr)
        return 1
    if not torch.cuda.is_available():
        print("ERROR: CUDA is unavailable", file=sys.stderr)
        return 1
    if subprocess.run(["g++", "--version"], stdout=subprocess.DEVNULL, check=False).returncode:
        print("ERROR: g++ is unavailable", file=sys.stderr)
        return 1

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        ),
        device_map={"": 0},
    )
    started = time.perf_counter()
    base_records = generate_suite(model, tokenizer, "base")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH, local_files_only=True)
    lora_records = generate_suite(model, tokenizer, "lora1536")
    torch.cuda.synchronize()
    report = {
        "model_path": str(MODEL_PATH),
        "adapter_path": str(ADAPTER_PATH),
        "generation": GENERATION_CONFIG,
        "compile_protocol": "g++ -std=c++17 -fsyntax-only on code extracted from an optional Markdown fence",
        "tasks": {"base": base_records, "lora1536": lora_records},
        "summary": {"base": summarize(base_records), "lora1536": summarize(lora_records)},
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

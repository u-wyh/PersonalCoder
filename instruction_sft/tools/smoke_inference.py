#!/usr/bin/env python3
"""Minimal offline compile smoke test for Instruction-SFT-v1 (not a benchmark)."""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE = Path("/data/PersonalCoder/model")
DEFAULT_ADAPTER = Path("/data/PersonalCoder/checkpoints/rtx4060/instruction_sft_v1/final_adapter")
DEFAULT_REPORT = PROJECT_ROOT / "instruction_sft" / "reports" / "smoke_v1.json"
PROMPTS = (
    (
        "SMOKE_LOCAL_001",
        "给定一个长度为 n 的整数序列，输出其中偶数元素的数量以及所有元素之和，两个答案用一个空格分隔。第一行输入 n，第二行输入 n 个整数。请使用 C++17 实现并只输出完整代码。",
    ),
    (
        "SMOKE_LOCAL_002",
        "输入一个只包含英文字母且不含空格的字符串 s，输出将 s 完全反转后的字符串。请使用 C++17 实现并只输出完整代码。",
    ),
    (
        "SMOKE_LOCAL_003",
        "输入两个正整数 a 和 b，输出它们的最大公约数与最小公倍数，使用一个空格分隔。保证结果在 64 位有符号整数范围内。请使用 C++17 实现并只输出完整代码。",
    ),
)


def extract_cpp(text: str) -> str:
    fenced = re.search(r"```(?:cpp|c\+\+|cc|cxx)?\s*(.*?)```", text, re.I | re.S)
    code = fenced.group(1) if fenced else text
    include = re.search(r"(?m)^\s*#\s*include\b", code)
    if include:
        code = code[include.start() :]
    return code.strip()


def compile_cpp(code: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="instruction_sft_smoke_") as temporary:
        source = Path(temporary) / "main.cpp"
        source.write_text(code + "\n", encoding="utf-8")
        result = subprocess.run(
            ["g++", "-std=c++17", "-O2", "-pipe", "-fsyntax-only", str(source)],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    return {"success": result.returncode == 0, "stderr": result.stderr[-2000:]}


def known_ids() -> tuple[set[str], set[str]]:
    train_ids = set()
    for split in ("train.jsonl", "val.jsonl"):
        path = PROJECT_ROOT / "instruction_sft" / "data" / "splits" / split
        train_ids.update(json.loads(line)["id"] for line in path.read_text(encoding="utf-8").splitlines())
    benchmark = {
        json.loads(line)["id"]
        for line in (PROJECT_ROOT / "benchmark" / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    }
    return train_ids, benchmark


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()
    if not args.base_model.is_dir() or not args.adapter.is_dir():
        parser.error("local base model or adapter is missing")
    train_ids, benchmark_ids = known_ids()
    smoke_ids = {item[0] for item in PROMPTS}
    if smoke_ids & (train_ids | benchmark_ids):
        raise ValueError("smoke ID overlaps training or benchmark")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        local_files_only=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        ),
        device_map={"": 0},
        dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(base, args.adapter, local_files_only=True, is_trainable=False)
    model.eval()
    records = []
    for smoke_id, prompt in PROMPTS:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(rendered, return_tensors="pt").to("cuda:0")
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(
            output_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()
        code = extract_cpp(generated)
        compilation = compile_cpp(code)
        records.append(
            {
                "id": smoke_id,
                "prompt": prompt,
                "generated_output": generated,
                "generated_tokens": int(output_ids.shape[1] - inputs["input_ids"].shape[1]),
                "has_cpp_main": bool(re.search(r"\b(?:int|signed)\s+main\s*\(", code)),
                "compile": compilation,
            }
        )
    report = {
        "base_model": str(args.base_model),
        "adapter": str(args.adapter),
        "model_loaded": True,
        "cuda": torch.cuda.is_available(),
        "training_id_overlap": False,
        "benchmark_id_overlap": False,
        "prompts": len(records),
        "compiled": sum(item["compile"]["success"] for item in records),
        "all_normal": all(item["has_cpp_main"] and item["compile"]["success"] for item in records),
        "records": records,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, ensure_ascii=False, indent=2))
    del model, base
    gc.collect()
    torch.cuda.empty_cache()
    return 0 if report["all_normal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

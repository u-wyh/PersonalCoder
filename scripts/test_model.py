#!/usr/bin/env python3
"""Run a local-only 4-bit smoke test for the configured code model."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
MAX_NEW_TOKENS = 512
PROMPT = "请用 C++ 实现两个整数相加，只输出完整代码。"


def memory_mib() -> str:
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    return f"allocated={allocated:.2f} MiB, reserved={reserved:.2f} MiB"


def main() -> int:
    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        print("ERROR: HF_HOME is not set. Set it to the existing Hugging Face cache directory.", file=sys.stderr)
        return 1
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available; this 4-bit smoke test requires a CUDA GPU.", file=sys.stderr)
        return 1

    hub_cache = Path(hf_home).expanduser() / "hub"
    print(f"HF_HOME: {Path(hf_home).expanduser()}")
    print(f"GPU memory before loading: {memory_mib()}")
    torch.cuda.reset_peak_memory_stats()

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            cache_dir=hub_cache,
            local_files_only=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            cache_dir=hub_cache,
            local_files_only=True,
            quantization_config=quantization_config,
            device_map={"": 0},
        )
    except (OSError, ValueError) as error:
        print(
            "ERROR: The model is missing or incomplete in the local Hugging Face cache. "
            "Wait for the manual download to finish; this script will not download it.\n"
            f"Details: {error}",
            file=sys.stderr,
        )
        return 1

    torch.cuda.synchronize()
    model_device = next(model.parameters()).device
    print(f"Model device: {model_device}")
    print(f"GPU memory after loading: {memory_mib()}")

    messages = [{"role": "user", "content": PROMPT}]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model_device)

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    new_tokens = generated[0, inputs.input_ids.shape[1]:]
    print("Generated result:")
    print(tokenizer.decode(new_tokens, skip_special_tokens=True))
    torch.cuda.synchronize()
    peak_mib = torch.cuda.max_memory_allocated() / 1024**2
    print(f"GPU peak allocated memory: {peak_mib:.2f} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

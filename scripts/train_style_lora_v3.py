#!/usr/bin/env python3
"""Train and verify the offline, time-weighted Style LoRA v3 adapter."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import re
import subprocess
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import bitsandbytes as bnb
import torch
import yaml
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "style_lora_v3.yaml"
EXPECTED_GPU_FRAGMENT = "RTX 4060 Laptop GPU"


class WeightedJsonlTextDataset(Dataset):
    def __init__(self, path: Path, require_weights: bool) -> None:
        self.records: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                try:
                    record = json.loads(line)
                    text = record["text"]
                    weight = record.get("time_weight", 1.0)
                except (json.JSONDecodeError, KeyError) as error:
                    raise ValueError(
                        f"Invalid sample in {path} at line {line_number}: {error}"
                    ) from error
                if not isinstance(text, str) or not text:
                    raise ValueError(f"Empty text in {path} at line {line_number}")
                if not isinstance(weight, (int, float)) or weight <= 0:
                    raise ValueError(f"Invalid time_weight in {path} at line {line_number}")
                if require_weights and "time_weight" not in record:
                    raise ValueError(f"Missing time_weight in {path} at line {line_number}")
                self.records.append(
                    {
                        "text": text,
                        "time_weight": float(weight),
                        "path": record.get("path"),
                    }
                )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]

    @property
    def weights(self) -> list[float]:
        return [record["time_weight"] for record in self.records]


class CausalCollator:
    def __init__(self, tokenizer: Any, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, records: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            [record["text"] for record in records],
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )
        labels = encoded["input_ids"].clone()
        labels[encoded["attention_mask"] == 0] = -100
        encoded["labels"] = labels
        encoded["sample_time_weights"] = torch.tensor(
            [record["time_weight"] for record in records], dtype=torch.float32
        )
        return encoded


def cosine_multiplier(step: int, total_steps: int, warmup_steps: int) -> float:
    if warmup_steps and step < warmup_steps:
        return float(step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def quantization_config(config: dict[str, Any]) -> BitsAndBytesConfig:
    if config["compute_dtype"] != "float16":
        raise ValueError("v3 requires float16 quantization compute dtype")
    return BitsAndBytesConfig(
        load_in_4bit=bool(config["load_in_4bit"]),
        bnb_4bit_quant_type=config["quant_type"],
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=bool(config["use_double_quant"]),
    )


def select_batch_size(training: dict[str, Any]) -> tuple[int, str]:
    preferred = int(training["per_device_train_batch_size"])
    total_mib = torch.cuda.get_device_properties(0).total_memory / 1024**2
    if not training.get("auto_select_batch_size", False):
        return preferred, "configured"
    if total_mib < 10 * 1024:
        return 1, "auto-selected batch size 1 for GPU memory below 10 GiB"
    return preferred, "preferred batch size fits detected GPU memory"


def extract_code(text: str) -> str:
    fenced = re.search(
        r"```(?:cpp|c\+\+|cc|cxx)?\s*\n?(.*?)```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return (fenced.group(1) if fenced else text).strip()


def compile_cpp(code: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="personalcoder_lora_v3_verify_") as tmp:
        source = Path(tmp) / "verify.cpp"
        source.write_text(code + "\n", encoding="utf-8")
        result = subprocess.run(
            ["g++", "-std=c++17", "-fsyntax-only", str(source)],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        return {
            "success": result.returncode == 0,
            "returncode": result.returncode,
            "stderr": result.stderr[-4000:],
        }


def fresh_adapter_validation(
    model_path: Path,
    adapter_path: Path,
    quantization: dict[str, Any],
    verification: dict[str, Any],
) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        quantization_config=quantization_config(quantization),
        device_map={"": 0},
        dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        is_trainable=False,
        local_files_only=True,
    )
    model.eval()
    model.config.use_cache = True
    prompt = "请使用 C++ 实现并查集，支持合并和连通性查询，只输出完整代码。"
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(rendered, return_tensors="pt").to("cuda:0")
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=int(verification["max_new_tokens"]),
            do_sample=bool(verification["do_sample"]),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated_ids = output_ids[0, inputs["input_ids"].shape[1] :]
    output = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    code = extract_code(output)
    compilation = compile_cpp(code)
    result = {
        "fresh_base_loaded": True,
        "adapter_loaded": True,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0),
        "prompt": prompt,
        "generated_tokens": int(generated_ids.numel()),
        "generated_output": output,
        "has_cpp_main": bool(re.search(r"\b(?:int|signed)\s+main\s*\(", code)),
        "compilation": compilation,
        "valid_code": bool(code) and compilation["success"],
    }
    del model, base_model, inputs, output_ids
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model_path = Path(config["model_path"])
    dataset_path = PROJECT_ROOT / config["dataset_path"]
    output_dir = Path(config["output_dir"])
    final_adapter_dir = output_dir / "final_adapter"
    report_path = PROJECT_ROOT / config["report_path"]
    training = config["training"]
    lora = config["lora"]
    quantization = config["quantization"]
    sampling = config["sampling"]

    required = [model_path, dataset_path / "train.jsonl", dataset_path / "validation.jsonl"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing local path(s): " + ", ".join(missing))
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint path: {output_dir}")
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite training report: {report_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    gpu_name = torch.cuda.get_device_name(0)
    if EXPECTED_GPU_FRAGMENT not in gpu_name:
        raise RuntimeError(f"Unexpected GPU: {gpu_name}")
    if subprocess.run(["g++", "--version"], capture_output=True, check=False).returncode:
        raise RuntimeError("g++ is unavailable for post-training validation")
    if int(training["num_train_epochs"]) != 1:
        raise ValueError("v3 requires exactly one epoch")
    if sampling["strategy"] != "weighted_random_sampler" or not sampling["replacement"]:
        raise ValueError("v3 requires WeightedRandomSampler with replacement")
    if list(lora["target_modules"]) != ["q_proj", "v_proj"]:
        raise ValueError("Unexpected LoRA target modules")

    seed = int(config["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_dataset = WeightedJsonlTextDataset(
        dataset_path / "train.jsonl", require_weights=True
    )
    validation_dataset = WeightedJsonlTextDataset(
        dataset_path / "validation.jsonl", require_weights=False
    )
    batch_size, batch_selection_reason = select_batch_size(training)
    collator = CausalCollator(tokenizer, int(training["max_seq_length"]))
    sampler_generator = torch.Generator().manual_seed(seed)
    train_sampler = WeightedRandomSampler(
        weights=torch.tensor(train_dataset.weights, dtype=torch.double),
        num_samples=len(train_dataset),
        replacement=True,
        generator=sampler_generator,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(training["per_device_eval_batch_size"]),
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )

    print(
        f"Preflight passed: GPU={gpu_name}, train={len(train_dataset)}, "
        f"validation={len(validation_dataset)}, batch={batch_size}",
        flush=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        quantization_config=quantization_config(quantization),
        device_map={"": 0},
        dtype=torch.float16,
    )
    model.config.use_cache = bool(training["use_cache"])
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=bool(training["gradient_checkpointing"]),
        gradient_checkpointing_kwargs={"use_reentrant": True},
    )
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(lora["r"]),
            lora_alpha=int(lora["alpha"]),
            lora_dropout=float(lora["dropout"]),
            target_modules=list(lora["target_modules"]),
            bias="none",
        ),
    )
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    optimizer = bnb.optim.PagedAdamW8bit(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(training["learning_rate"]),
    )
    accumulation = int(training["gradient_accumulation_steps"])
    total_optimizer_steps = math.ceil(len(train_loader) / accumulation)
    warmup_steps = math.ceil(total_optimizer_steps * float(training["warmup_ratio"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_multiplier(step, total_optimizer_steps, warmup_steps),
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    run_started = time.perf_counter()
    training_started = run_started
    model.train()
    optimizer.zero_grad(set_to_none=True)
    train_losses: list[float] = []
    sampled_weight_counts: Counter[str] = Counter()
    optimizer_steps = 0
    logging_steps = int(training["logging_optimizer_steps"])
    for micro_step, batch in enumerate(train_loader, start=1):
        drawn_weights = batch.pop("sample_time_weights")
        sampled_weight_counts.update(f"{float(weight):g}" for weight in drawn_weights)
        batch = {
            key: value.to("cuda:0", non_blocking=True) for key, value in batch.items()
        }
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=bool(training["fp16"])
        ):
            loss = model(**batch).loss
        train_losses.append(float(loss.detach().cpu()))
        (loss / accumulation).backward()
        if micro_step % accumulation == 0 or micro_step == len(train_loader):
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            if optimizer_steps == 1 or optimizer_steps % logging_steps == 0:
                print(
                    f"optimizer_step={optimizer_steps}/{total_optimizer_steps} "
                    f"micro_step={micro_step}/{len(train_loader)} loss={train_losses[-1]:.6f}",
                    flush=True,
                )
    torch.cuda.synchronize()
    training_seconds = time.perf_counter() - training_started

    model.eval()
    eval_losses: list[float] = []
    with torch.inference_mode():
        for batch in validation_loader:
            batch.pop("sample_time_weights")
            batch = {
                key: value.to("cuda:0", non_blocking=True) for key, value in batch.items()
            }
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=bool(training["fp16"])
            ):
                eval_losses.append(float(model(**batch).loss.detach().cpu()))
    torch.cuda.synchronize()
    peak_allocated_mib = round(torch.cuda.max_memory_allocated() / 1024**2, 2)
    peak_reserved_mib = round(torch.cuda.max_memory_reserved() / 1024**2, 2)

    model.save_pretrained(final_adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(final_adapter_dir)
    required_adapter_files = (
        final_adapter_dir / "adapter_config.json",
        final_adapter_dir / "adapter_model.safetensors",
        final_adapter_dir / "tokenizer_config.json",
    )
    missing_adapter_files = [str(path) for path in required_adapter_files if not path.is_file()]
    if missing_adapter_files:
        raise RuntimeError("Saved adapter is incomplete: " + ", ".join(missing_adapter_files))

    train_loss = sum(train_losses) / len(train_losses)
    eval_loss = sum(eval_losses) / len(eval_losses)
    del model, optimizer, scheduler, train_loader, validation_loader
    gc.collect()
    torch.cuda.empty_cache()
    inference_validation = fresh_adapter_validation(
        model_path, final_adapter_dir, quantization, config["verification"]
    )
    if not inference_validation["valid_code"]:
        raise RuntimeError("Fresh Base + LoRA-v3 did not generate compilable C++ code")

    report = {
        "status": "completed",
        "device": {
            "gpu": gpu_name,
            "cuda_available": True,
            "total_memory_mib": round(
                torch.cuda.get_device_properties(0).total_memory / 1024**2, 2
            ),
        },
        "configuration": config,
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "weighted_sampler": {
            "probability": "proportional to time_weight",
            "replacement": True,
            "draws_per_epoch": len(train_dataset),
            "actual_draw_counts_by_weight": dict(sorted(sampled_weight_counts.items())),
            "validation_weighted": False,
        },
        "selected_per_device_train_batch_size": batch_size,
        "batch_selection_reason": batch_selection_reason,
        "gradient_accumulation_steps": accumulation,
        "optimizer": training["optimizer"],
        "optimizer_steps": optimizer_steps,
        "expected_optimizer_steps": total_optimizer_steps,
        "epoch": 1,
        "train_loss": round(train_loss, 6),
        "eval_loss": round(eval_loss, 6),
        "peak_allocated_memory_mib": peak_allocated_mib,
        "peak_reserved_memory_mib": peak_reserved_mib,
        "training_time_seconds": round(training_seconds, 3),
        "total_run_time_seconds": round(time.perf_counter() - run_started, 3),
        "trainable_parameters": trainable_parameters,
        "adapter_path": str(final_adapter_dir),
        "adapter_files": [path.name for path in required_adapter_files],
        "inference_validation": inference_validation,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "optimizer_steps": optimizer_steps,
                "train_loss": report["train_loss"],
                "eval_loss": report["eval_loss"],
                "training_time_seconds": report["training_time_seconds"],
                "peak_allocated_memory_mib": peak_allocated_mib,
                "peak_reserved_memory_mib": peak_reserved_mib,
                "adapter_path": str(final_adapter_dir),
                "inference_validation": inference_validation,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

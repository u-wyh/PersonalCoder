#!/usr/bin/env python3
"""Run two-step QLoRA memory benchmarks without saving model state."""

from __future__ import annotations

import gc
import json
import statistics
import sys
import time
from pathlib import Path

import bitsandbytes as bnb
import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path("/data/PersonalCoder/model")
TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "style" / "train.jsonl"
REPORT_PATH = PROJECT_ROOT / "outputs" / "training_memory_report.json"
SEQUENCE_LENGTHS = (768, 1024)
TRAIN_STEPS = 2


def memory_snapshot() -> dict[str, float]:
    return {
        "allocated_mib": round(torch.cuda.memory_allocated() / 1024**2, 2),
        "reserved_mib": round(torch.cuda.memory_reserved() / 1024**2, 2),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
    }


def load_batches(tokenizer: object, sequence_length: int) -> list[dict[str, torch.Tensor]]:
    batches: list[dict[str, torch.Tensor]] = []
    with TRAIN_PATH.open(encoding="utf-8") as dataset:
        for line in dataset:
            text = json.loads(line)["text"]
            encoded = tokenizer(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=sequence_length,
                padding=False,
                return_tensors="pt",
            )
            if encoded.input_ids.shape[1] != sequence_length:
                continue
            batches.append({key: value for key, value in encoded.items()})
            if len(batches) == TRAIN_STEPS:
                return batches
    raise ValueError(f"Could not find {TRAIN_STEPS} samples with at least {sequence_length} tokens")


def is_cuda_oom(error: RuntimeError) -> bool:
    return isinstance(error, torch.OutOfMemoryError) or "out of memory" in str(error).lower()


def run_benchmark(tokenizer: object, sequence_length: int) -> dict[str, object]:
    model = None
    optimizer = None
    active_step_started: float | None = None
    losses: list[float] = []
    step_times: list[float] = []
    result: dict[str, object] = {
        "max_seq_length": sequence_length,
        "steps_requested": TRAIN_STEPS,
        "steps_completed": 0,
        "success": False,
        "cuda_oom": False,
    }

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    try:
        batches = load_batches(tokenizer, sequence_length)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
            quantization_config=quantization_config,
            device_map={"": 0},
        )
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        model = get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=8,
                lora_alpha=16,
                lora_dropout=0.05,
                target_modules=["q_proj", "v_proj"],
                bias="none",
            ),
        )
        model.train()
        trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        result["trainable_parameters"] = sum(parameter.numel() for parameter in trainable_parameters)
        optimizer = bnb.optim.PagedAdamW8bit(trainable_parameters, lr=2e-4)

        for batch in batches:
            optimizer.zero_grad(set_to_none=True)
            input_ids = batch["input_ids"].to("cuda:0")
            attention_mask = batch["attention_mask"].to("cuda:0")
            labels = input_ids.clone()
            torch.cuda.synchronize()
            active_step_started = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                ).loss
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            step_times.append(time.perf_counter() - active_step_started)
            active_step_started = None
            losses.append(float(loss.detach().cpu()))
            result["steps_completed"] = len(losses)

        result["success"] = True
    except RuntimeError as error:
        if not is_cuda_oom(error):
            raise
        result["cuda_oom"] = True
        result["error"] = str(error)
        if active_step_started is not None:
            result["failed_step_elapsed_seconds"] = round(
                time.perf_counter() - active_step_started,
                4,
            )
    finally:
        result.update(memory_snapshot())
        result["losses"] = [round(loss, 6) for loss in losses]
        result["step_times_seconds"] = [round(duration, 4) for duration in step_times]
        result["mean_step_time_seconds"] = (
            round(statistics.fmean(step_times), 4) if step_times else None
        )
        del optimizer, model
        gc.collect()
        torch.cuda.empty_cache()

    return result


def main() -> int:
    if not MODEL_PATH.is_dir():
        print(f"ERROR: Local model directory not found: {MODEL_PATH}", file=sys.stderr)
        return 1
    if not TRAIN_PATH.is_file():
        print(f"ERROR: Training dataset not found: {TRAIN_PATH}", file=sys.stderr)
        return 1
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available", file=sys.stderr)
        return 1

    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    report: dict[str, object] = {
        "model_path": str(MODEL_PATH),
        "train_path": TRAIN_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "configuration": {
            "quantization": "4-bit NF4",
            "compute_dtype": "float16",
            "double_quantization": True,
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "v_proj"],
            "batch_size": 1,
            "gradient_checkpointing": True,
            "optimizer": "PagedAdamW8bit",
            "learning_rate": 0.0002,
            "steps_per_length": TRAIN_STEPS,
        },
        "tests": {},
    }

    tests = report["tests"]
    for sequence_length in SEQUENCE_LENGTHS:
        print(f"\nTesting max_seq_length={sequence_length} ...")
        test_result = run_benchmark(tokenizer, sequence_length)
        tests[str(sequence_length)] = test_result
        print(json.dumps(test_result, ensure_ascii=False, indent=2))
        if sequence_length == 768 and test_result["cuda_oom"]:
            tests["1024"] = {
                "max_seq_length": 1024,
                "skipped": True,
                "reason": "Skipped because max_seq_length=768 caused CUDA OOM",
            }
            break

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nReport saved to: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

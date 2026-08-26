#!/usr/bin/env python3
"""Offline QLoRA training for PersonalCoder Instruction SFT v1."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import bitsandbytes as bnb
import torch
import yaml
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "instruction_sft" / "configs" / "instruction_sft_v1.yaml"
EXPECTED_GPU = "RTX 4060 Laptop GPU"


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, str]]:
    records = []
    with path.open(encoding="utf-8") as source:
        for line_number, raw in enumerate(source, 1):
            try:
                item = json.loads(raw)
                instruction, response = item["instruction"], item["response"]
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(f"invalid sample at {path}:{line_number}: {exc}") from exc
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError(f"empty instruction at {path}:{line_number}")
            if not isinstance(response, str) or not response.strip():
                raise ValueError(f"empty response at {path}:{line_number}")
            records.append({"instruction": instruction, "response": response})
            if limit and len(records) >= limit:
                break
    return records


class InstructionDataset(Dataset):
    def __init__(self, records: list[dict[str, str]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, str]:
        return self.records[index]


def token_ids(value: Any) -> list[int]:
    ids = value["input_ids"] if hasattr(value, "keys") else value
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


class AssistantOnlyCollator:
    """Apply the official chat template and mask every non-assistant token."""

    def __init__(self, tokenizer: Any, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    @staticmethod
    def trim_prompt(ids: list[int], budget: int) -> list[int]:
        if budget >= len(ids):
            return ids
        if budget <= 64:
            return ids[-budget:]
        tail = min(64, budget // 4)
        return ids[: budget - tail] + ids[-tail:]

    def encode(self, record: dict[str, str]) -> dict[str, Any]:
        prompt_text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": record["instruction"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = self.tokenizer.apply_chat_template(
            [
                {"role": "user", "content": record["instruction"]},
                {"role": "assistant", "content": record["response"]},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        if not full_text.startswith(prompt_text):
            raise ValueError("official rendered prompt is not a prefix of the full conversation")
        encoded = self.tokenizer(
            full_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        full = list(encoded["input_ids"])
        boundary = len(prompt_text)
        # Mask a rare token that straddles the prompt/response character
        # boundary; this is safer than assigning any prompt text to the loss.
        assistant_start = next(
            (index for index, (start, _end) in enumerate(encoded["offset_mapping"]) if start >= boundary),
            len(full),
        )
        prompt, response = full[:assistant_start], full[assistant_start:]
        if not response:
            raise ValueError("assistant response produced no tokens")
        original_length = len(full)
        if original_length <= self.max_length:
            kept_prompt, kept_response = prompt, response
        elif len(response) < self.max_length:
            kept_response = response
            kept_prompt = self.trim_prompt(prompt, self.max_length - len(response))
        else:
            prompt_budget = min(256, max(32, self.max_length // 8))
            kept_prompt = self.trim_prompt(prompt, prompt_budget)
            kept_response = response[: self.max_length - len(kept_prompt)]
        input_ids = kept_prompt + kept_response
        labels = [-100] * len(kept_prompt) + kept_response.copy()
        if not kept_response or any(label != -100 for label in labels[: len(kept_prompt)]):
            raise AssertionError("assistant-only loss mask is invalid")
        return {
            "input_ids": input_ids,
            "labels": labels,
            "assistant_start": len(kept_prompt),
            "original_length": original_length,
            "truncated": original_length > self.max_length,
        }

    def __call__(self, records: list[dict[str, str]]) -> dict[str, torch.Tensor]:
        examples = [self.encode(record) for record in records]
        max_batch_length = max(len(example["input_ids"]) for example in examples)
        pad_id = self.tokenizer.pad_token_id
        input_ids, labels, attention = [], [], []
        for example in examples:
            padding = max_batch_length - len(example["input_ids"])
            input_ids.append(example["input_ids"] + [pad_id] * padding)
            labels.append(example["labels"] + [-100] * padding)
            attention.append([1] * len(example["input_ids"]) + [0] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
            "original_lengths": torch.tensor([item["original_length"] for item in examples]),
            "truncated_samples": torch.tensor([item["truncated"] for item in examples]),
            "assistant_starts": torch.tensor([item["assistant_start"] for item in examples]),
        }


def quantization_config(config: dict[str, Any]) -> BitsAndBytesConfig:
    if config["compute_dtype"] != "float16":
        raise ValueError("Instruction SFT v1 requires float16 compute dtype")
    return BitsAndBytesConfig(
        load_in_4bit=bool(config["load_in_4bit"]),
        bnb_4bit_quant_type=str(config["quant_type"]),
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=bool(config["use_double_quant"]),
    )


def cosine_multiplier(step: int, total_steps: int, warmup_steps: int) -> float:
    if warmup_steps and step < warmup_steps:
        return float(step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def mask_audit(collator: AssistantOnlyCollator, records: list[dict[str, str]]) -> dict[str, Any]:
    examples = [collator.encode(record) for record in records[:16]]
    valid = all(
        all(label == -100 for label in example["labels"][: example["assistant_start"]])
        and all(label != -100 for label in example["labels"][example["assistant_start"] :])
        for example in examples
    )
    return {
        "checked_samples": len(examples),
        "prompt_labels_all_masked": valid,
        "assistant_labels_present": all(
            len(example["labels"]) > example["assistant_start"] for example in examples
        ),
    }


def fresh_adapter_reload(base_path: Path, adapter_path: Path, quantization: dict[str, Any]) -> dict[str, Any]:
    base = AutoModelForCausalLM.from_pretrained(
        base_path,
        local_files_only=True,
        quantization_config=quantization_config(quantization),
        device_map={"": 0},
        dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(base, adapter_path, local_files_only=True, is_trainable=False)
    result = {
        "fresh_base_loaded": True,
        "adapter_reloaded": True,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    del model, base
    gc.collect()
    torch.cuda.empty_cache()
    return result


def write_reports(json_path: Path, md_path: Path, report: dict[str, Any]) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rows = [
        "# Instruction SFT v1 Training Report",
        "",
        f"Mode: `{'sanity' if report['sanity'] else 'full'}`; completed: **{report['completed']}**.",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Train samples | {report['train_samples']} |",
        f"| Validation samples | {report['validation_samples']} |",
        f"| Max sequence length | {report['max_seq_length']} |",
        f"| Truncated train samples | {report['truncated_train_samples']} ({report['truncation_rate']:.2%}) |",
        f"| Optimizer steps | {report['optimizer_steps']} |",
        f"| Train loss | {report['train_loss']:.6f} |",
        f"| Eval loss | {report['eval_loss']:.6f} |",
        f"| Learning rate | {report['learning_rate']} |",
        f"| Training seconds | {report['training_time_seconds']:.2f} |",
        f"| Peak allocated VRAM MiB | {report['peak_allocated_memory_mib']:.2f} |",
        f"| Peak reserved VRAM MiB | {report['peak_reserved_memory_mib']:.2f} |",
        f"| Final adapter | `{report['final_adapter_path']}` |",
        "",
        f"Assistant-only mask audit: `{json.dumps(report['mask_audit'], ensure_ascii=False)}`",
        "",
        f"Fresh adapter reload: `{json.dumps(report['adapter_reload'], ensure_ascii=False)}`",
    ]
    md_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--sanity", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--report-markdown", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    training, lora, quantization = config["training"], config["lora"], config["quantization"]
    base_path = Path(config["base_model"])
    train_path = project_path(config["train_file"])
    val_path = project_path(config["validation_file"])
    if args.sanity:
        mode = config["sanity"]
        output_dir = args.output_dir or Path(mode["output_dir"])
        report_json = args.report_json or project_path(mode["report_json"])
        report_markdown = args.report_markdown or project_path(mode["report_markdown"])
        train_limit, val_limit = int(mode["train_samples"]), int(mode["validation_samples"])
        max_steps = int(mode["optimizer_steps"])
        sanity_accumulation = int(mode["gradient_accumulation_steps"])
    else:
        output_dir = args.output_dir or Path(config["output_dir"])
        report_json = args.report_json or project_path(config["report_json"])
        report_markdown = args.report_markdown or project_path(config["report_markdown"])
        train_limit = val_limit = max_steps = None
    required = [base_path, train_path, val_path]
    if missing := [str(path) for path in required if not path.exists()]:
        raise FileNotFoundError("missing local path(s): " + ", ".join(missing))
    if output_dir.exists() or report_json.exists() or report_markdown.exists():
        raise FileExistsError("refusing to overwrite an existing output/report path")
    if not torch.cuda.is_available() or EXPECTED_GPU not in torch.cuda.get_device_name(0):
        raise RuntimeError(f"required RTX 4060 GPU unavailable: {torch.cuda.is_available()}")
    if int(training["epochs"]) != 1 or list(lora["target_modules"]) != ["q_proj", "v_proj"]:
        raise ValueError("v1 requires one epoch and q_proj/v_proj targets")
    if int(training["per_device_train_batch_size"]) * int(training["gradient_accumulation_steps"]) != int(training["effective_batch_size"]):
        raise ValueError("configured effective batch size is inconsistent")

    seed = int(config["seed"])
    seed_everything(seed)
    tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_records = read_jsonl(train_path, train_limit)
    val_records = read_jsonl(val_path, val_limit)
    collator = AssistantOnlyCollator(tokenizer, int(training["max_seq_length"]))
    audit = mask_audit(collator, train_records)
    if not all(audit.values()):
        raise AssertionError(f"assistant mask audit failed: {audit}")
    precomputed = [collator.encode(record) for record in train_records]
    truncated_train = sum(example["truncated"] for example in precomputed)
    del precomputed

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        InstructionDataset(train_records),
        batch_size=int(training["per_device_train_batch_size"]),
        shuffle=True,
        generator=generator,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        InstructionDataset(val_records),
        batch_size=int(training["per_device_eval_batch_size"]),
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )
    accumulation = sanity_accumulation if args.sanity else int(training["gradient_accumulation_steps"])
    planned_steps = math.ceil(len(train_loader) / accumulation)
    total_steps = min(planned_steps, max_steps) if max_steps else planned_steps
    warmup_steps = math.ceil(total_steps * float(training["warmup_ratio"]))
    print(
        f"Preflight: mode={'sanity' if args.sanity else 'full'} GPU={torch.cuda.get_device_name(0)} "
        f"train={len(train_records)} val={len(val_records)} max_seq={training['max_seq_length']} "
        f"truncated={truncated_train} steps={total_steps}",
        flush=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        local_files_only=True,
        quantization_config=quantization_config(quantization),
        device_map={"": 0},
        dtype=torch.float16,
    )
    model.config.use_cache = False
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
            bias=str(lora["bias"]),
        ),
    )
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = bnb.optim.PagedAdamW8bit(
        [p for p in model.parameters() if p.requires_grad], lr=float(training["learning_rate"])
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: cosine_multiplier(step, total_steps, warmup_steps)
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    optimizer_steps = 0
    for micro_step, batch in enumerate(train_loader, 1):
        for key in ("original_lengths", "truncated_samples", "assistant_starts"):
            batch.pop(key)
        batch = {key: value.to("cuda:0", non_blocking=True) for key, value in batch.items()}
        with torch.autocast("cuda", dtype=torch.float16, enabled=bool(training["fp16"])):
            loss = model(**batch).loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at micro step {micro_step}: {loss}")
        losses.append(float(loss.detach().cpu()))
        (loss / accumulation).backward()
        if micro_step % accumulation == 0 or micro_step == len(train_loader):
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            if optimizer_steps == 1 or optimizer_steps % int(training["logging_optimizer_steps"]) == 0:
                recent = sum(losses[-accumulation:]) / min(accumulation, len(losses))
                print(f"step={optimizer_steps}/{total_steps} loss={recent:.6f} lr={scheduler.get_last_lr()[0]:.8g}", flush=True)
            if not args.sanity and optimizer_steps % int(training["save_optimizer_steps"]) == 0:
                model.save_pretrained(output_dir / f"checkpoint-{optimizer_steps}", safe_serialization=True)
            if optimizer_steps >= total_steps:
                break
    torch.cuda.synchronize()
    training_seconds = time.perf_counter() - started

    model.eval()
    eval_losses = []
    with torch.inference_mode():
        for batch in val_loader:
            for key in ("original_lengths", "truncated_samples", "assistant_starts"):
                batch.pop(key)
            batch = {key: value.to("cuda:0", non_blocking=True) for key, value in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16, enabled=bool(training["fp16"])):
                eval_loss = model(**batch).loss
            if not torch.isfinite(eval_loss):
                raise FloatingPointError(f"non-finite eval loss: {eval_loss}")
            eval_losses.append(float(eval_loss.cpu()))

    final_adapter = output_dir / "final_adapter"
    model.save_pretrained(final_adapter, safe_serialization=True)
    tokenizer.save_pretrained(final_adapter)
    peak_allocated = torch.cuda.max_memory_allocated() / 1024**2
    peak_reserved = torch.cuda.max_memory_reserved() / 1024**2
    del model, optimizer, scheduler, train_loader, val_loader
    gc.collect()
    torch.cuda.empty_cache()
    reload_result = fresh_adapter_reload(base_path, final_adapter, quantization)
    report = {
        "completed": True,
        "sanity": args.sanity,
        "base_model": str(base_path),
        "train_samples": len(train_records),
        "validation_samples": len(val_records),
        "max_seq_length": int(training["max_seq_length"]),
        "truncated_train_samples": truncated_train,
        "truncation_rate": round(truncated_train / len(train_records), 6),
        "epochs": 1,
        "optimizer_steps": optimizer_steps,
        "micro_steps": len(losses),
        "per_device_train_batch_size": int(training["per_device_train_batch_size"]),
        "gradient_accumulation_steps": accumulation,
        "effective_batch_size": int(training["per_device_train_batch_size"]) * accumulation,
        "learning_rate": float(training["learning_rate"]),
        "optimizer": training["optimizer"],
        "train_loss": sum(losses) / len(losses),
        "eval_loss": sum(eval_losses) / len(eval_losses),
        "training_time_seconds": round(training_seconds, 3),
        "peak_allocated_memory_mib": round(peak_allocated, 2),
        "peak_reserved_memory_mib": round(peak_reserved, 2),
        "trainable_parameters": trainable_parameters,
        "quantization": quantization,
        "lora": lora,
        "mask_audit": audit,
        "adapter_reload": reload_result,
        "final_adapter_path": str(final_adapter),
        "seed": seed,
    }
    write_reports(report_json, report_markdown, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

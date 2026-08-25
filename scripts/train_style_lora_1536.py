#!/usr/bin/env python3
"""Train and minimally verify the isolated 1536-token Style QLoRA adapter."""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import time
from pathlib import Path

import bitsandbytes as bnb
import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "style_lora_1536_v1.json"


class JsonlTextDataset(Dataset):
    def __init__(self, path: Path) -> None:
        self.records = []
        with path.open(encoding="utf-8") as source:
            for line in source:
                record = json.loads(line)
                self.records.append(record["text"])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> str:
        return self.records[index]


def gpu_used_mib() -> float | None:
    try:
        value = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "--id=0"],
            text=True,
        ).strip().splitlines()[0]
        return float(value)
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def cosine_multiplier(step: int, total_steps: int, warmup_steps: int) -> float:
    if warmup_steps and step < warmup_steps:
        return float(step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    training = config["training"]
    quant = config["quantization"]
    lora = config["lora"]
    model_path = Path(config["model_path"])
    dataset_path = PROJECT_ROOT / config["dataset_path"]
    output_dir = Path(config["output_dir"])

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    for split in ("train", "validation", "test"):
        if not (dataset_path / f"{split}.jsonl").is_file():
            raise FileNotFoundError(dataset_path / f"{split}.jsonl")
    if training["num_train_epochs"] != 1:
        raise ValueError("This v1 script requires exactly one epoch")

    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=config["local_files_only"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_length = training["max_seq_length"]

    def collate(texts: list[str]) -> dict[str, torch.Tensor]:
        encoded = tokenizer(
            texts,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        labels = encoded["input_ids"].clone()
        labels[encoded["attention_mask"] == 0] = -100
        encoded["labels"] = labels
        return encoded

    train_dataset = JsonlTextDataset(dataset_path / "train.jsonl")
    validation_dataset = JsonlTextDataset(dataset_path / "validation.jsonl")
    generator = torch.Generator().manual_seed(config["seed"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=training["batch_size"],
        shuffle=True,
        generator=generator,
        collate_fn=collate,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=training["batch_size"],
        shuffle=False,
        collate_fn=collate,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=config["local_files_only"],
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=quant["load_in_4bit"],
            bnb_4bit_quant_type=quant["quant_type"],
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=quant["use_double_quant"],
        ),
        device_map={"": 0},
    )
    model.config.use_cache = training["use_cache"]
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=training["gradient_checkpointing"]
    )
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora["r"],
            lora_alpha=lora["alpha"],
            lora_dropout=lora["dropout"],
            target_modules=lora["target_modules"],
            bias="none",
        ),
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = bnb.optim.PagedAdamW8bit(trainable, lr=training["learning_rate"])
    accumulation = training["gradient_accumulation_steps"]
    total_steps = math.ceil(len(train_loader) / accumulation)
    warmup_steps = math.ceil(total_steps * training["warmup_ratio"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: cosine_multiplier(step, total_steps, warmup_steps)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    run_started = time.perf_counter()
    train_started = run_started
    model.train()
    optimizer.zero_grad(set_to_none=True)
    train_losses: list[float] = []
    optimizer_steps = 0
    for micro_step, batch in enumerate(train_loader, start=1):
        batch = {key: value.to("cuda:0") for key, value in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=training["fp16"]):
            loss = model(**batch).loss
        train_losses.append(float(loss.detach().cpu()))
        (loss / accumulation).backward()
        if micro_step % accumulation == 0 or micro_step == len(train_loader):
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            print(
                f"step {optimizer_steps}/{total_steps} micro_step={micro_step}/{len(train_loader)} "
                f"loss={train_losses[-1]:.6f}",
                flush=True,
            )
    torch.cuda.synchronize()
    train_elapsed = time.perf_counter() - train_started

    model.eval()
    eval_losses: list[float] = []
    with torch.no_grad():
        for batch in validation_loader:
            batch = {key: value.to("cuda:0") for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=training["fp16"]):
                eval_losses.append(float(model(**batch).loss.detach().cpu()))
    torch.cuda.synchronize()

    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    prompt = "#include <bits/stdc++.h>\nusing namespace std;\nint main() {"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    inference_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    torch.cuda.synchronize()
    metrics = {
        "completed_epochs": 1,
        "train_micro_steps": len(train_loader),
        "total_optimizer_steps": optimizer_steps,
        "total_expected_optimizer_steps": total_steps,
        "training_elapsed_seconds": round(train_elapsed, 3),
        "total_run_elapsed_seconds": round(time.perf_counter() - run_started, 3),
        "train_loss": round(sum(train_losses) / len(train_losses), 6),
        "eval_loss": round(sum(eval_losses) / len(eval_losses), 6),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 1024**2, 2),
        "gpu_used_memory_at_end_mib": gpu_used_mib(),
        "adapter_path": str(output_dir),
        "inference_prompt": prompt,
        "inference_output": inference_text,
    }
    (output_dir / "training_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

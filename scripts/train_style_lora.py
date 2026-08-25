#!/usr/bin/env python3
"""Train the first formal PersonalCoder style LoRA adapter locally."""

from __future__ import annotations

import gc
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path("/data/PersonalCoder/model")
TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "style_chunks" / "train.jsonl"
VALIDATION_PATH = PROJECT_ROOT / "data" / "processed" / "style_chunks" / "validation.jsonl"
OUTPUT_DIR = Path("/data/PersonalCoder/checkpoints/style_lora_v1")
FINAL_ADAPTER_DIR = OUTPUT_DIR / "final_adapter"
SUMMARY_PATH = OUTPUT_DIR / "training_summary.json"
MAX_SEQ_LENGTH = 512
EXPECTED_GPU = "NVIDIA GeForce RTX 3050 Laptop GPU"


TRAINING_CONFIG: dict[str, Any] = {
    "model_path": str(MODEL_PATH),
    "train_dataset": str(TRAIN_PATH),
    "validation_dataset": str(VALIDATION_PATH),
    "output_dir": str(OUTPUT_DIR),
    "local_files_only": True,
    "quantization": {
        "load_in_4bit": True,
        "quantization_type": "nf4",
        "compute_dtype": "float16",
        "double_quant": True,
    },
    "lora": {
        "r": 8,
        "alpha": 16,
        "dropout": 0.05,
        "target_modules": ["q_proj", "v_proj"],
    },
    "max_seq_length": MAX_SEQ_LENGTH,
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "gradient_checkpointing": True,
    "optimizer": "paged_adamw_8bit",
    "learning_rate": 2e-4,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
    "num_train_epochs": 1,
    "fp16": True,
    "logging_steps": 10,
    "eval_strategy": "steps",
    "eval_steps": 100,
    "save_strategy": "steps",
    "save_steps": 100,
    "save_total_limit": 2,
    "seed": 42,
}


class JsonlTextDataset(Dataset):
    def __init__(self, path: Path) -> None:
        self.texts: list[str] = []
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                try:
                    text = json.loads(line)["text"]
                except (json.JSONDecodeError, KeyError) as error:
                    raise ValueError(f"Invalid sample in {path} at line {line_number}: {error}") from error
                if not isinstance(text, str) or not text:
                    raise ValueError(f"Invalid text in {path} at line {line_number}")
                self.texts.append(text)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, str]:
        return {"text": self.texts[index]}


class CausalLanguageModelingCollator:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, str]]) -> dict[str, torch.Tensor]:
        batch = self.tokenizer(
            [feature["text"] for feature in features],
            add_special_tokens=False,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=True,
            return_tensors="pt",
        )
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch


class StepTimingCallback(TrainerCallback):
    def __init__(self, logging_steps: int) -> None:
        self.logging_steps = logging_steps
        self.started_at: float | None = None
        self.step_timings: list[dict[str, float | int]] = []

    def on_step_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self.started_at = time.perf_counter()

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        if self.started_at is None:
            return
        duration = time.perf_counter() - self.started_at
        self.step_timings.append({"step": state.global_step, "seconds": round(duration, 4)})
        if state.global_step % self.logging_steps == 0:
            print(f"step={state.global_step} optimizer_step_seconds={duration:.4f}", flush=True)
        self.started_at = None


def memory_metrics() -> dict[str, float]:
    return {
        "allocated_mib": round(torch.cuda.memory_allocated() / 1024**2, 2),
        "reserved_mib": round(torch.cuda.memory_reserved() / 1024**2, 2),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 1024**2, 2),
    }


def write_summary(summary: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sanity_check() -> None:
    missing = [path for path in (MODEL_PATH, TRAIN_PATH, VALIDATION_PATH) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Required local path missing: {', '.join(map(str, missing))}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    gpu_name = torch.cuda.get_device_name(0)
    if gpu_name != EXPECTED_GPU:
        raise RuntimeError(f"Unexpected GPU: {gpu_name!r}; expected {EXPECTED_GPU!r}")
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {OUTPUT_DIR}")
    print(f"Sanity check passed: GPU={gpu_name}")


def quantization_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def validate_saved_adapter(tokenizer: Any) -> dict[str, Any]:
    gc.collect()
    torch.cuda.empty_cache()
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        quantization_config=quantization_config(),
        device_map={"": 0},
    )
    model = PeftModel.from_pretrained(base_model, FINAL_ADAPTER_DIR, is_trainable=False)
    model.eval()
    prompt = "请用 C++ 实现并查集，只输出完整代码。"
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda:0")
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    new_tokens = generated[0, inputs.input_ids.shape[1]:]
    output = tokenizer.decode(new_tokens, skip_special_tokens=True)
    del model, base_model, inputs, generated
    gc.collect()
    torch.cuda.empty_cache()
    return {"success": True, "prompt": prompt, "output": output}


def main() -> int:
    try:
        sanity_check()
    except (FileNotFoundError, RuntimeError) as error:
        print(f"ERROR: Sanity check failed: {error}", file=sys.stderr)
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "training_config.json").write_text(
        json.dumps(TRAINING_CONFIG, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    torch.manual_seed(42)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    summary: dict[str, Any] = {"status": "running", "configuration": TRAINING_CONFIG}

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        train_dataset = JsonlTextDataset(TRAIN_PATH)
        validation_dataset = JsonlTextDataset(VALIDATION_PATH)
        print(f"Loaded train={len(train_dataset)} validation={len(validation_dataset)} samples")

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
            quantization_config=quantization_config(),
            device_map={"": 0},
        )
        model.config.use_cache = False
        checkpointing_kwargs = {"use_reentrant": True}
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs=checkpointing_kwargs,
        )
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
        trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        print(f"Trainable parameters: {trainable_parameters}")

        timing_callback = StepTimingCallback(logging_steps=10)
        training_args = TrainingArguments(
            output_dir=str(OUTPUT_DIR),
            do_train=True,
            do_eval=True,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=8,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs=checkpointing_kwargs,
            learning_rate=2e-4,
            warmup_steps=0.03,
            lr_scheduler_type="cosine",
            num_train_epochs=1,
            fp16=True,
            optim="paged_adamw_8bit",
            logging_strategy="steps",
            logging_steps=10,
            logging_first_step=True,
            eval_strategy="steps",
            eval_steps=100,
            prediction_loss_only=True,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=2,
            seed=42,
            data_seed=42,
            report_to="none",
            remove_unused_columns=False,
            use_cache=False,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            data_collator=CausalLanguageModelingCollator(tokenizer),
            processing_class=tokenizer,
            callbacks=[timing_callback],
        )

        started = time.perf_counter()
        train_result = trainer.train()
        training_elapsed = time.perf_counter() - started
        final_eval_metrics = trainer.evaluate()
        trainer.save_model(str(FINAL_ADAPTER_DIR))
        tokenizer.save_pretrained(FINAL_ADAPTER_DIR)
        trainer.save_state()

        summary.update(
            {
                "status": "trained",
                "train_samples": len(train_dataset),
                "validation_samples": len(validation_dataset),
                "trainable_parameters": trainable_parameters,
                "global_steps": trainer.state.global_step,
                "completed_epochs": trainer.state.epoch,
                "training_elapsed_seconds": round(training_elapsed, 2),
                "train_metrics": train_result.metrics,
                "final_eval_metrics": final_eval_metrics,
                "final_train_loss": train_result.metrics.get("train_loss"),
                "final_eval_loss": final_eval_metrics.get("eval_loss"),
                "memory": memory_metrics(),
                "step_timings": timing_callback.step_timings,
                "log_history": trainer.state.log_history,
                "adapter_path": str(FINAL_ADAPTER_DIR),
            }
        )
        write_summary(summary)

        del trainer, model, train_result
        gc.collect()
        torch.cuda.empty_cache()
        inference_result = validate_saved_adapter(tokenizer)
        summary["inference_validation"] = inference_result
        summary["status"] = "completed"
        write_summary(summary)
        print(json.dumps({key: summary[key] for key in (
            "status", "global_steps", "completed_epochs", "training_elapsed_seconds",
            "final_train_loss", "final_eval_loss", "memory", "adapter_path",
            "inference_validation",
        )}, ensure_ascii=False, indent=2))
        return 0
    except RuntimeError as error:
        if "out of memory" not in str(error).lower():
            raise
        error_trace = traceback.format_exc()
        summary.update(
            {
                "status": "cuda_oom",
                "error": str(error),
                "traceback": error_trace,
                "memory": memory_metrics(),
            }
        )
        write_summary(summary)
        print(error_trace, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

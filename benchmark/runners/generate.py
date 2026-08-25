#!/usr/bin/env python3
"""Generate C++ submissions from one local Base or LoRA model."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


RUNNERS_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = RUNNERS_DIR.parent
PROJECT_ROOT = BENCHMARK_DIR.parent
PROMPT_PATH = BENCHMARK_DIR / "prompts" / "cpp_only.txt"
MAX_NEW_TOKENS = 1024
SEED = 42
GENERATION_CONFIG: dict[str, Any] = {
    "max_new_tokens": MAX_NEW_TOKENS,
    "do_sample": False,
    "temperature": 0.0,
    "top_p": 1.0,
    "num_beams": 1,
    "repetition_penalty": 1.0,
    "use_cache": True,
}

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.evaluate import EvaluationError, Problem, load_manifest  # noqa: E402


FENCE_PATTERN = re.compile(
    r"```[ \t]*([^\r\n`]*)\r?\n(.*?)```", re.IGNORECASE | re.DOTALL
)
CODE_START_PATTERN = re.compile(
    r"(?m)^[ \t]*(?:#[ \t]*include\b|using[ \t]+namespace[ \t]+std\b|"
    r"(?:int|signed)[ \t]+main[ \t]*\()"
)
CPP_FENCE_LANGUAGES = {"cpp", "c++", "cc", "cxx"}


class GenerationError(RuntimeError):
    """Raised when local generation inputs or runtime are invalid."""


@dataclass(frozen=True)
class ModelPaths:
    base: Path
    adapter: Path | None


def extract_cpp(text: str) -> str:
    """Extract the most likely complete C++ source from model output."""
    stripped = text.strip()
    fenced_blocks = [
        (match.group(1).strip().lower(), match.group(2).strip())
        for match in FENCE_PATTERN.finditer(stripped)
    ]
    if fenced_blocks:
        cpp_blocks = [
            code for language, code in fenced_blocks if language in CPP_FENCE_LANGUAGES
        ]
        candidates = cpp_blocks or [code for _, code in fenced_blocks]
        return max(candidates, key=len).strip()

    code_start = CODE_START_PATTERN.search(stripped)
    if code_start:
        return stripped[code_start.start() :].strip()
    return stripped


def _safe_model_name(model: str) -> str:
    if not model or model in {".", ".."} or Path(model).name != model:
        raise GenerationError(f"invalid model directory name: {model!r}")
    return model


def _existing_directory(path: str | Path, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise GenerationError(f"{label} directory not found: {resolved}")
    return resolved


def resolve_model_paths(
    model_path: str | Path | None,
    base_model: str | Path | None,
    adapter_path: str | Path | None,
) -> ModelPaths:
    """Validate the mutually exclusive Base and Base+LoRA CLI forms."""
    if adapter_path is None:
        if model_path is None:
            raise GenerationError("Base generation requires --model-path")
        if base_model is not None:
            raise GenerationError("use --model-path, not --base-model, without LoRA")
        return ModelPaths(
            base=_existing_directory(model_path, "model"), adapter=None
        )

    if base_model is None:
        raise GenerationError("LoRA generation requires --base-model")
    if model_path is not None:
        raise GenerationError("do not combine --model-path with --adapter-path")
    return ModelPaths(
        base=_existing_directory(base_model, "base model"),
        adapter=_existing_directory(adapter_path, "adapter"),
    )


def load_prompt_template(path: Path = PROMPT_PATH) -> str:
    if not path.is_file():
        raise GenerationError(f"prompt template not found: {path}")
    template = path.read_text(encoding="utf-8")
    if template.count("{problem}") != 1:
        raise GenerationError("prompt template must contain exactly one {problem}")
    return template


def load_statements(problems: Sequence[Problem]) -> dict[str, str]:
    statements: dict[str, str] = {}
    for problem in problems:
        statement_path = problem.directory / "statement.md"
        if not statement_path.is_file():
            raise GenerationError(f"statement not found: {statement_path}")
        statements[problem.problem_id] = statement_path.read_text(encoding="utf-8")
    return statements


def load_local_model(paths: ModelPaths) -> tuple[Any, Any, Any]:
    """Load one tokenizer and one 4-bit Base or Base+LoRA model locally."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise GenerationError("CUDA is required for benchmark generation")

    tokenizer = AutoTokenizer.from_pretrained(paths.base, local_files_only=True)
    if not tokenizer.chat_template:
        raise GenerationError("local tokenizer does not define a chat template")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        paths.base,
        local_files_only=True,
        quantization_config=quantization,
        device_map={"": 0},
        dtype=torch.float16,
    )
    if paths.adapter is not None:
        model = PeftModel.from_pretrained(
            model,
            paths.adapter,
            is_trainable=False,
            local_files_only=True,
        )
    model.eval()
    model.config.use_cache = True
    return model, tokenizer, torch


def generate_one(
    model: Any,
    tokenizer: Any,
    torch: Any,
    prompt: str,
) -> tuple[str, int]:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(rendered, return_tensors="pt").to("cuda:0")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            **GENERATION_CONFIG,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = generated[0, inputs["input_ids"].shape[1] :]
    raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return raw_text, int(new_tokens.numel())


def save_text(path: Path, text: str, ensure_final_newline: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if ensure_final_newline and text and not text.endswith("\n"):
        text += "\n"
    path.write_text(text, encoding="utf-8")


def run_generation(
    model_name: str,
    manifest_path: str | Path,
    paths: ModelPaths,
    output_dir: str | Path = BENCHMARK_DIR / "outputs",
    raw_output_dir: str | Path = BENCHMARK_DIR / "raw_outputs",
) -> dict[str, Any]:
    """Generate and save raw/extracted outputs for all manifest problems."""
    model_name = _safe_model_name(model_name)
    try:
        problems = load_manifest(manifest_path)
    except EvaluationError as exc:
        raise GenerationError(str(exc)) from exc
    statements = load_statements(problems)
    prompt_template = load_prompt_template()
    extracted_root = Path(output_dir).resolve() / model_name
    raw_root = Path(raw_output_dir).resolve() / model_name

    model, tokenizer, torch = load_local_model(paths)
    records: list[dict[str, Any]] = []
    for index, problem in enumerate(problems, start=1):
        print(
            f"[{model_name}] generating {problem.problem_id} ({index}/{len(problems)})",
            file=sys.stderr,
            flush=True,
        )
        prompt = prompt_template.replace("{problem}", statements[problem.problem_id])
        raw_text, token_count = generate_one(model, tokenizer, torch, prompt)
        code = extract_cpp(raw_text)
        raw_path = raw_root / f"{problem.problem_id}.txt"
        code_path = extracted_root / f"{problem.problem_id}.cpp"
        save_text(raw_path, raw_text)
        save_text(code_path, code, ensure_final_newline=True)
        records.append(
            {
                "problem": problem.problem_id,
                "generated_tokens": token_count,
                "raw_output": str(raw_path),
                "code_output": str(code_path),
            }
        )

    return {
        "model": model_name,
        "base_model": str(paths.base),
        "adapter": str(paths.adapter) if paths.adapter else None,
        "problems": len(problems),
        "generation_config": GENERATION_CONFIG,
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="output model label")
    parser.add_argument("--manifest", type=Path, required=True, help="JSONL manifest")
    parser.add_argument("--model-path", type=Path, help="local Base model path")
    parser.add_argument("--base-model", type=Path, help="local Base path for LoRA")
    parser.add_argument("--adapter-path", type=Path, help="local LoRA adapter path")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BENCHMARK_DIR / "outputs",
        help="directory for extracted <model>/<problem>.cpp files",
    )
    parser.add_argument(
        "--raw-output-dir",
        type=Path,
        default=BENCHMARK_DIR / "raw_outputs",
        help="directory for raw <model>/<problem>.txt files",
    )
    args = parser.parse_args()

    try:
        paths = resolve_model_paths(args.model_path, args.base_model, args.adapter_path)
        result = run_generation(
            model_name=args.model,
            manifest_path=args.manifest,
            paths=paths,
            output_dir=args.output_dir,
            raw_output_dir=args.raw_output_dir,
        )
    except GenerationError as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

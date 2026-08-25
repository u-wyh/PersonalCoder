#!/usr/bin/env python3
"""Offline, deterministic three-way Style LoRA evaluation on one GPU."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL_PATH = Path("/data/PersonalCoder/model")
LORA_512_PATH = Path(
    "/data/PersonalCoder/checkpoints/rtx4060/style_lora_512_v1/final_adapter"
)
LORA_1536_PATH = Path(
    "/data/PersonalCoder/checkpoints/rtx4060/style_lora_1536_v1"
)
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "unified_style_eval"
REPORT_PATH = OUTPUT_ROOT / "report.json"

COMMON_INSTRUCTION = "请使用 C++ 实现，只输出完整代码。"
TASKS = [
    ("01_union_find", "实现并查集，支持合并两个集合并查询两个元素是否属于同一集合。"),
    ("02_dijkstra", "实现使用邻接表和优先队列的 Dijkstra 单源最短路。"),
    ("03_floyd", "实现 Floyd 算法求有向带权图的任意两点最短路。"),
    ("04_bfs", "实现 BFS 求无权图中从 1 号点到其他点的最短距离。"),
    ("05_dfs", "实现 DFS 统计无向图的连通块数量。"),
    ("06_tarjan_scc", "实现 Tarjan 算法求有向图的强连通分量。"),
    ("07_articulation_points", "实现 Tarjan 算法求无向图的所有割点。"),
    ("08_bridges", "实现 Tarjan 算法求无向图的所有桥。"),
    ("09_trie", "实现仅含小写字母字符串的 Trie，支持插入和查询出现次数。"),
    ("10_kmp", "实现 KMP 字符串匹配，输出模式串在文本串中的所有匹配起点。"),
    ("11_fenwick_tree", "实现树状数组，支持单点加和区间和查询。"),
    ("12_segment_tree", "实现普通线段树，支持单点修改和区间最大值查询。"),
    ("13_lazy_segment_tree", "实现带 lazy propagation 的线段树，支持区间加和区间和查询。"),
    ("14_sparse_table", "实现 ST 表，回答静态数组的区间最大值查询。"),
    ("15_lca", "实现倍增 LCA，回答树上多次最近公共祖先查询。"),
    ("16_heavy_light_decomposition", "实现树链剖分，支持树上路径加和路径和查询。"),
    ("17_topological_sort", "实现 Kahn 拓扑排序，并判断有向图是否存在环。"),
    ("18_binary_search", "实现二分查找，回答有序数组中第一个大于等于给定值的位置。"),
    ("19_monotonic_stack", "实现单调栈，求每个元素右侧第一个严格更大元素的位置。"),
    ("20_monotonic_queue", "实现单调队列，求长度为 k 的每个滑动窗口最大值。"),
    ("21_zero_one_knapsack", "实现 01 背包，求容量限制下可获得的最大价值。"),
    ("22_complete_knapsack", "实现完全背包，求容量限制下可获得的最大价值。"),
    ("23_bitmask_dp", "实现状态压缩 DP 求解从 0 号点出发访问所有点一次的最短路径。"),
    ("24_bellman_ford", "实现 Bellman-Ford 单源最短路并检测可达负环。"),
    ("25_kruskal_mst", "实现 Kruskal 算法求无向带权图的最小生成树。"),
    ("26_dinic_max_flow", "实现 Dinic 算法求有向网络的最大流。"),
    ("27_bipartite_check", "使用 BFS 判断无向图是否为二分图。"),
    ("28_directed_cycle_dfs", "使用 DFS 三色标记判断有向图是否存在环。"),
    ("29_heap", "实现小根堆，支持插入、查询最小值和删除最小值。"),
    ("30_prefix_sum", "实现二维前缀和，回答矩阵子矩形元素和查询。"),
]
PROMPTS = [(task_id, statement + COMMON_INSTRUCTION) for task_id, statement in TASKS]

GENERATION_CONFIG: dict[str, Any] = {
    "do_sample": False,
    "temperature": 1.0,
    "top_p": 1.0,
    "max_new_tokens": 1024,
    "num_beams": 1,
    "repetition_penalty": 1.0,
    "use_cache": True,
}
MODEL_PATHS = {
    "base": None,
    "lora512": LORA_512_PATH,
    "lora1536": LORA_1536_PATH,
}
BOOLEAN_STYLE_METRICS = (
    "bits_stdcpp",
    "using_namespace_std",
    "global_max_constant",
    "static_array",
    "vector",
    "map",
    "unordered_map",
    "set",
    "associative_container",
    "int_main",
    "signed_main",
    "return_0",
    "ios_sync_with_stdio",
    "cin_tie",
    "extra_natural_language",
    "strict_code_only",
)

FENCE_PATTERN = re.compile(
    r"```(?:cpp|c\+\+|cc|cxx|C\+\+)?\s*\n?(.*?)```",
    flags=re.IGNORECASE | re.DOTALL,
)
CODE_START_PATTERN = re.compile(
    r"(?m)^\s*(?:#\s*include|#\s*define|using\s+namespace|typedef\b|"
    r"template\s*<|(?:const|constexpr|static)\b|struct\b|class\b|enum\b|"
    r"(?:int|signed|int32_t)\s+main\s*\(|//|/\*)"
)


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def preflight() -> None:
    required = {
        "base model": BASE_MODEL_PATH,
        "LoRA-512": LORA_512_PATH,
        "LoRA-1536": LORA_1536_PATH,
    }
    missing = [f"{label}: {path}" for label, path in required.items() if not path.is_dir()]
    if missing:
        fail("missing local model path(s): " + "; ".join(missing))
    if OUTPUT_ROOT.exists():
        fail(f"refusing to overwrite existing output: {OUTPUT_ROOT}")
    if not torch.cuda.is_available():
        fail("CUDA is unavailable")
    compiler = subprocess.run(
        ["g++", "--version"], capture_output=True, text=True, check=False
    )
    if compiler.returncode != 0:
        fail("g++ is unavailable")


def extract_code(raw_text: str) -> tuple[str, dict[str, bool]]:
    stripped = raw_text.strip()
    fences = list(FENCE_PATTERN.finditer(stripped))
    if fences:
        selected = max(fences, key=lambda match: len(match.group(1)))
        outside = FENCE_PATTERN.sub("", stripped).strip()
        code = selected.group(1).strip()
        return code, {
            "had_markdown_fence": True,
            "extra_natural_language": bool(outside),
            "strict_code_only": False,
        }

    start = CODE_START_PATTERN.search(stripped)
    if not start:
        return stripped, {
            "had_markdown_fence": False,
            "extra_natural_language": bool(stripped),
            "strict_code_only": False,
        }

    code_start = start.start()
    prefix = stripped[:code_start].strip()
    candidate = stripped[code_start:].strip()
    last_brace = candidate.rfind("}")
    suffix = ""
    if last_brace >= 0:
        suffix = candidate[last_brace + 1 :].strip()
        code = candidate[: last_brace + 1].strip()
    else:
        code = candidate
    extra = bool(prefix or suffix)
    return code, {
        "had_markdown_fence": False,
        "extra_natural_language": extra,
        "strict_code_only": not extra and stripped == code,
    }


def style_metrics(code: str, extraction: dict[str, bool]) -> dict[str, Any]:
    global_max_pattern = re.compile(
        r"(?m)^\s*(?:(?:inline|static)\s+)?(?:const|constexpr)\s+"
        r"(?:unsigned\s+)?(?:int|long\s+long|size_t|auto)\s+"
        r"(?:MAXN|MAXM|MAX_[A-Z0-9_]+|MAX[A-Z0-9_]*)\b"
    )
    static_array_pattern = re.compile(
        r"(?m)^\s*(?:(?:static|const|constexpr)\s+)*"
        r"(?:unsigned\s+)?(?:bool|char|short|int|long\s+long|ll|float|double)\s+"
        r"[A-Za-z_]\w*\s*\[[^\]]+\]"
    )
    has_map = bool(re.search(r"\b(?:std::)?map\s*<", code))
    has_unordered_map = bool(re.search(r"\b(?:std::)?unordered_map\s*<", code))
    has_set = bool(re.search(r"\b(?:std::)?set\s*<", code))
    return {
        "bits_stdcpp": bool(
            re.search(r"#\s*include\s*<\s*bits/stdc\+\+\.h\s*>", code)
        ),
        "using_namespace_std": bool(re.search(r"\busing\s+namespace\s+std\s*;", code)),
        "global_max_constant": bool(global_max_pattern.search(code)),
        "static_array": bool(static_array_pattern.search(code)),
        "vector": bool(re.search(r"\b(?:std::)?vector\s*<", code)),
        "map": has_map,
        "unordered_map": has_unordered_map,
        "set": has_set,
        "associative_container": has_map or has_unordered_map or has_set,
        "int_main": bool(re.search(r"\bint\s+main\s*\(", code)),
        "signed_main": bool(re.search(r"\bsigned\s+main\s*\(", code)),
        "return_0": bool(re.search(r"\breturn\s+0\s*;", code)),
        "ios_sync_with_stdio": bool(
            re.search(r"\bios\s*::\s*sync_with_stdio\s*\(", code)
        ),
        "cin_tie": bool(re.search(r"\bcin\s*\.\s*tie\s*\(", code)),
        "line_comment_count": len(re.findall(r"//[^\n]*", code)),
        "block_comment_count": len(re.findall(r"/\*.*?\*/", code, re.DOTALL)),
        **extraction,
    }


def compile_source(path: Path, label: str) -> dict[str, Any]:
    command = ["g++", "-std=c++17", "-x", "c++", "-fsyntax-only", str(path)]
    started = time.perf_counter()
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        return {
            "success": result.returncode == 0,
            "returncode": result.returncode,
            "stderr": result.stderr[-4000:],
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "source": label,
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "returncode": None,
            "stderr": "g++ timeout after 60 seconds",
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "source": label,
        }


def load_model(adapter_path: Path | None) -> Any:
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
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
    if adapter_path is not None:
        model = PeftModel.from_pretrained(
            model,
            adapter_path,
            is_trainable=False,
            local_files_only=True,
        )
    model.eval()
    return model


def generate_one(model: Any, tokenizer: Any, rendered_prompt: str, seed: int) -> tuple[str, int]:
    inputs = tokenizer(rendered_prompt, return_tensors="pt").to("cuda")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            **GENERATION_CONFIG,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = generated[0, inputs["input_ids"].shape[1] :]
    raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return raw_text, int(new_tokens.numel())


def evaluate_model(
    label: str,
    adapter_path: Path | None,
    tokenizer: Any,
    rendered_prompts: dict[str, str],
) -> list[dict[str, Any]]:
    output_dir = OUTPUT_ROOT / label
    output_dir.mkdir(exist_ok=False)
    print(f"Loading {label}...", flush=True)
    model = load_model(adapter_path)
    records: list[dict[str, Any]] = []
    try:
        for index, (task_id, prompt) in enumerate(PROMPTS, start=1):
            started = time.perf_counter()
            raw_text, generated_tokens = generate_one(
                model, tokenizer, rendered_prompts[task_id], seed=20260825 + index
            )
            code, extraction = extract_code(raw_text)
            raw_path = output_dir / f"{task_id}.raw.txt"
            code_path = output_dir / f"{task_id}.extracted.cpp"
            raw_path.write_text(raw_text + "\n", encoding="utf-8", errors="strict")
            code_path.write_text(code + "\n", encoding="utf-8", errors="strict")
            raw_compile = compile_source(raw_path, "raw_output")
            extracted_compile = compile_source(code_path, "extracted_code")
            style = style_metrics(code, extraction)
            record = {
                "id": task_id,
                "prompt": prompt,
                "raw_output_path": str(raw_path.relative_to(PROJECT_ROOT)),
                "extracted_code_path": str(code_path.relative_to(PROJECT_ROOT)),
                "generated_tokens": generated_tokens,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "style": style,
                "compilation": {
                    "raw": raw_compile,
                    "extracted": extracted_compile,
                },
                "instruction_following": {
                    "no_extra_natural_language": not style["extra_natural_language"],
                    "strict_code_only": style["strict_code_only"],
                },
                "complete_program": bool(
                    (style["int_main"] or style["signed_main"])
                    and extracted_compile["success"]
                ),
            }
            records.append(record)
            print(
                f"[{label}] {index:02d}/{len(PROMPTS)} {task_id}: "
                f"raw={raw_compile['success']} extracted={extracted_compile['success']}",
                flush=True,
            )
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return records


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    style: dict[str, Any] = {}
    for metric in BOOLEAN_STYLE_METRICS:
        count = sum(bool(record["style"][metric]) for record in records)
        style[metric] = {"count": count, "rate": round(count / total, 6)}
    for metric in ("line_comment_count", "block_comment_count"):
        values = [int(record["style"][metric]) for record in records]
        style[metric] = {
            "total": sum(values),
            "average_per_output": round(sum(values) / total, 6),
            "outputs_with_comments": sum(value > 0 for value in values),
        }
    raw_count = sum(record["compilation"]["raw"]["success"] for record in records)
    extracted_count = sum(
        record["compilation"]["extracted"]["success"] for record in records
    )
    complete_count = sum(record["complete_program"] for record in records)
    return {
        "sample_count": total,
        "style": style,
        "compile": {
            "raw": {"success_count": raw_count, "rate": round(raw_count / total, 6)},
            "extracted": {
                "success_count": extracted_count,
                "rate": round(extracted_count / total, 6),
            },
        },
        "complete_program": {
            "count": complete_count,
            "rate": round(complete_count / total, 6),
        },
    }


def style_comparison(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    comparison: dict[str, Any] = {}
    for metric in BOOLEAN_STYLE_METRICS:
        rates = {label: summaries[label]["style"][metric]["rate"] for label in MODEL_PATHS}
        comparison[metric] = {
            "rates": rates,
            "lora512_minus_base_pp": round((rates["lora512"] - rates["base"]) * 100, 4),
            "lora1536_minus_base_pp": round((rates["lora1536"] - rates["base"]) * 100, 4),
            "lora1536_minus_lora512_pp": round(
                (rates["lora1536"] - rates["lora512"]) * 100, 4
            ),
        }
    return comparison


def main() -> None:
    preflight()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rendered_prompts = {
        task_id: tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for task_id, prompt in PROMPTS
    }
    chat_template = tokenizer.chat_template or ""

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    model_records: dict[str, list[dict[str, Any]]] = {}
    for label, adapter_path in MODEL_PATHS.items():
        model_records[label] = evaluate_model(
            label, adapter_path, tokenizer, rendered_prompts
        )

    summaries = {label: summarize(records) for label, records in model_records.items()}
    tasks = []
    for index, (task_id, prompt) in enumerate(PROMPTS):
        tasks.append(
            {
                "id": task_id,
                "prompt": prompt,
                "models": {
                    label: model_records[label][index] for label in MODEL_PATHS
                },
            }
        )
    report = {
        "protocol": {
            "offline": True,
            "device": torch.cuda.get_device_name(0),
            "base_model_path": str(BASE_MODEL_PATH),
            "adapter_paths": {
                "lora512": str(LORA_512_PATH),
                "lora1536": str(LORA_1536_PATH),
            },
            "tokenizer_path": str(BASE_MODEL_PATH),
            "chat_template_sha256": hashlib.sha256(chat_template.encode()).hexdigest(),
            "generation_config": GENERATION_CONFIG,
            "common_instruction": COMMON_INSTRUCTION,
            "prompt_count": len(PROMPTS),
            "compile_command": "g++ -std=c++17 -x c++ -fsyntax-only <source>",
            "algorithm_correctness_evaluated": False,
        },
        "summary": summaries,
        "style_metric_comparison": style_comparison(summaries),
        "tasks": tasks,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        errors="strict",
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2), flush=True)
    print(f"Report: {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run the final offline four-model PersonalCoder style evaluation."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import evaluate_style_models as core


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL_PATH = Path("/data/PersonalCoder/model")
LORA_512_PATH = Path(
    "/data/PersonalCoder/checkpoints/rtx4060/style_lora_512_v1/final_adapter"
)
LORA_1536_PATH = Path(
    "/data/PersonalCoder/checkpoints/rtx4060/style_lora_1536_v1"
)
LORA_V3_PATH = Path(
    "/data/PersonalCoder/checkpoints/rtx4060/style_lora_v3/final_adapter"
)
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "final_style_eval"
REPORT_PATH = OUTPUT_ROOT / "report.json"
COMMON_INSTRUCTION = "请使用 C++ 实现，只输出完整代码。"
MODEL_PATHS = {
    "base": None,
    "lora512": LORA_512_PATH,
    "lora1536": LORA_1536_PATH,
    "lora_v3": LORA_V3_PATH,
}
TASKS = [
    ("01_union_find", "实现并查集，支持合并两个集合并查询连通性。"),
    ("02_bfs", "实现 BFS 求无权图从 1 号点出发的最短距离。"),
    ("03_dfs", "实现 DFS 统计无向图连通块数量。"),
    ("04_dijkstra", "实现邻接表和优先队列优化的 Dijkstra 单源最短路。"),
    ("05_floyd", "实现 Floyd 算法求有向带权图任意两点最短路。"),
    ("06_tarjan_scc", "实现 Tarjan 算法求有向图强连通分量。"),
    ("07_articulation_points", "实现 Tarjan 算法求无向图所有割点。"),
    ("08_bridges", "实现 Tarjan 算法求无向图所有桥。"),
    ("09_trie", "实现小写字母 Trie，支持插入和查询出现次数。"),
    ("10_kmp", "实现 KMP，输出模式串在文本串中的所有匹配起点。"),
    ("11_fenwick_tree", "实现树状数组，支持单点加和区间和查询。"),
    ("12_segment_tree", "实现普通线段树，支持单点修改和区间最大值查询。"),
    ("13_lazy_segment_tree", "实现 lazy propagation 线段树，支持区间加和区间和查询。"),
    ("14_sparse_table", "实现 ST 表，回答静态数组区间最大值查询。"),
    ("15_lca", "实现倍增 LCA，回答树上最近公共祖先查询。"),
    ("16_heavy_light_decomposition", "实现树链剖分，支持路径加和路径和查询。"),
    ("17_topological_sort", "实现 Kahn 拓扑排序并判断有向图是否有环。"),
    ("18_binary_search", "实现二分查找有序数组中第一个大于等于给定值的位置。"),
    ("19_monotonic_stack", "实现单调栈求每个元素右侧第一个严格更大元素的位置。"),
    ("20_monotonic_queue", "实现单调队列求长度为 k 的每个滑动窗口最大值。"),
    ("21_zero_one_knapsack", "实现 01 背包求容量限制下最大价值。"),
    ("22_complete_knapsack", "实现完全背包求容量限制下最大价值。"),
    ("23_bitmask_dp", "实现状态压缩 DP 求从 0 号点出发访问所有点一次的最短路径。"),
    ("24_bellman_ford", "实现 Bellman-Ford 单源最短路并检测可达负环。"),
    ("25_kruskal_mst", "实现 Kruskal 求无向带权图最小生成树。"),
    ("26_dinic_max_flow", "实现 Dinic 求有向网络最大流。"),
    ("27_bipartite_check", "使用 BFS 判断无向图是否为二分图。"),
    ("28_directed_cycle_dfs", "使用 DFS 三色标记判断有向图是否存在环。"),
    ("29_heap", "实现小根堆，支持插入、查询和删除最小值。"),
    ("30_prefix_sum_2d", "实现二维前缀和回答矩阵子矩形元素和查询。"),
    ("31_merge_sort", "实现归并排序并统计逆序对数量。"),
    ("32_quick_sort", "实现随机化快速排序。"),
    ("33_heap_sort", "实现原地堆排序。"),
    ("34_prefix_xor", "实现前缀异或回答静态数组区间异或查询。"),
    ("35_difference_array", "实现差分数组，支持多次区间加后输出最终数组。"),
    ("36_zero_one_bfs", "实现 01 BFS 求边权仅为 0 或 1 的单源最短路。"),
    ("37_spfa", "实现 SPFA 单源最短路并检测负环。"),
    ("38_prim_mst", "实现优先队列优化的 Prim 最小生成树。"),
    ("39_bipartite_matching", "实现匈牙利算法求二分图最大匹配。"),
    ("40_euler_sieve", "实现欧拉筛求不超过 n 的所有质数。"),
    ("41_fast_power", "实现快速幂计算 a 的 b 次方对模数 p 取模。"),
    ("42_extended_gcd", "实现扩展欧几里得算法求 gcd 及一组贝祖系数。"),
    ("43_modular_inverse", "实现扩展欧几里得求模逆，并处理逆元不存在的情况。"),
    ("44_combinations", "预处理阶乘和逆阶乘，回答多次组合数取模查询。"),
    ("45_matrix_exponentiation", "实现矩阵快速幂求斐波那契数列第 n 项。"),
    ("46_digit_dp", "实现数位 DP 统计区间内不含数字 4 的整数数量。"),
    ("47_interval_dp", "实现区间 DP 求石子合并的最小代价。"),
    ("48_lis", "实现 O(n log n) 的最长严格递增子序列长度算法。"),
    ("49_lcs", "实现动态规划求两个字符串的最长公共子序列长度。"),
    ("50_aho_corasick", "实现 AC 自动机统计多个模式串在文本中的出现次数。"),
]
PROMPTS = [(task_id, statement + COMMON_INSTRUCTION) for task_id, statement in TASKS]
CURRENT_STYLE_METRICS = (
    "bits_stdcpp",
    "using_namespace_std",
    "global_max_constant",
    "static_array",
    "ios_sync_with_stdio",
    "cin_tie",
)


def preflight() -> None:
    required = {
        "base": BASE_MODEL_PATH / "config.json",
        "lora512": LORA_512_PATH / "adapter_config.json",
        "lora1536": LORA_1536_PATH / "adapter_config.json",
        "lora_v3": LORA_V3_PATH / "adapter_config.json",
    }
    missing = [f"{label}: {path}" for label, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing local model path(s): " + "; ".join(missing))
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"Refusing to overwrite final evaluation: {OUTPUT_ROOT}")
    if not core.torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if subprocess.run(["g++", "--version"], capture_output=True, check=False).returncode:
        raise RuntimeError("g++ is unavailable")
    if len(PROMPTS) < 50 or any(not prompt.endswith(COMMON_INSTRUCTION) for _, prompt in PROMPTS):
        raise ValueError("The fixed prompt suite is incomplete or inconsistent")
    if core.GENERATION_CONFIG["do_sample"] is not False:
        raise ValueError("Final evaluation must be deterministic")
    if core.GENERATION_CONFIG["max_new_tokens"] != 1024:
        raise ValueError("Final evaluation requires max_new_tokens=1024")


def rate(summary: dict[str, Any], metric: str) -> float:
    return float(summary["style"][metric]["rate"])


def best_labels(values: dict[str, float]) -> list[str]:
    best = max(values.values())
    return [label for label, value in values.items() if value == best]


def build_analysis(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    style_scores = {
        label: round(
            sum(rate(summary, metric) for metric in CURRENT_STYLE_METRICS)
            / len(CURRENT_STYLE_METRICS),
            6,
        )
        for label, summary in summaries.items()
    }
    completeness = {
        label: float(summary["complete_program"]["rate"])
        for label, summary in summaries.items()
    }
    extracted_compile = {
        label: float(summary["compile"]["extracted"]["rate"])
        for label, summary in summaries.items()
    }
    unweighted_labels = ("lora512", "lora1536")
    best_unweighted_style = max(style_scores[label] for label in unweighted_labels)
    best_unweighted_compile = max(extracted_compile[label] for label in unweighted_labels)
    previous_pattern_consistent = (
        style_scores["lora512"] >= style_scores["lora1536"]
        and extracted_compile["lora1536"] >= extracted_compile["lora512"]
    )
    return {
        "current_style_alignment_score": style_scores,
        "most_similar_to_current_style": best_labels(style_scores),
        "code_completeness_rate": completeness,
        "highest_code_completeness": best_labels(completeness),
        "extracted_compile_rate": extracted_compile,
        "highest_compile_rate": best_labels(extracted_compile),
        "time_weighted_v3_vs_unweighted": {
            "v3_style_score": style_scores["lora_v3"],
            "best_unweighted_style_score": best_unweighted_style,
            "v3_extracted_compile_rate": extracted_compile["lora_v3"],
            "best_unweighted_extracted_compile_rate": best_unweighted_compile,
            "style_improved": style_scores["lora_v3"] > best_unweighted_style,
            "compile_improved": extracted_compile["lora_v3"] > best_unweighted_compile,
        },
        "lora512_vs_lora1536_matches_previous_experiment": previous_pattern_consistent,
    }


def main() -> None:
    preflight()
    core.OUTPUT_ROOT = OUTPUT_ROOT
    core.REPORT_PATH = REPORT_PATH
    core.PROMPTS = PROMPTS
    core.MODEL_PATHS = MODEL_PATHS
    started = time.perf_counter()
    tokenizer = core.AutoTokenizer.from_pretrained(BASE_MODEL_PATH, local_files_only=True)
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
        model_records[label] = core.evaluate_model(
            label, adapter_path, tokenizer, rendered_prompts
        )

    summaries = {label: core.summarize(records) for label, records in model_records.items()}
    style_table = {
        metric: {
            label: summaries[label]["style"][metric] for label in MODEL_PATHS
        }
        for metric in core.BOOLEAN_STYLE_METRICS
    }
    compile_table = {
        label: summaries[label]["compile"] for label in MODEL_PATHS
    }
    improvements = {
        label: {
            "style_rate_delta_pp": {
                metric: round(
                    (rate(summaries[label], metric) - rate(summaries["base"], metric))
                    * 100,
                    4,
                )
                for metric in core.BOOLEAN_STYLE_METRICS
            },
            "raw_compile_delta_pp": round(
                (
                    summaries[label]["compile"]["raw"]["rate"]
                    - summaries["base"]["compile"]["raw"]["rate"]
                )
                * 100,
                4,
            ),
            "extracted_compile_delta_pp": round(
                (
                    summaries[label]["compile"]["extracted"]["rate"]
                    - summaries["base"]["compile"]["extracted"]["rate"]
                )
                * 100,
                4,
            ),
        }
        for label in MODEL_PATHS
        if label != "base"
    }
    tasks = [
        {
            "id": task_id,
            "prompt": prompt,
            "models": {
                label: model_records[label][index] for label in MODEL_PATHS
            },
        }
        for index, (task_id, prompt) in enumerate(PROMPTS)
    ]
    report = {
        "protocol": {
            "offline": True,
            "device": core.torch.cuda.get_device_name(0),
            "base_model_path": str(BASE_MODEL_PATH),
            "adapter_paths": {
                label: str(path) for label, path in MODEL_PATHS.items() if path is not None
            },
            "tokenizer_path": str(BASE_MODEL_PATH),
            "chat_template_sha256": hashlib.sha256(chat_template.encode()).hexdigest(),
            "generation_config": core.GENERATION_CONFIG,
            "common_instruction": COMMON_INSTRUCTION,
            "prompt_count": len(PROMPTS),
            "compile_command": "g++ -std=c++17 -x c++ -fsyntax-only <source>",
            "algorithm_correctness_evaluated": False,
        },
        "summary": summaries,
        "style_metric_table": style_table,
        "compile_rate_table": compile_table,
        "base_vs_lora_improvements": improvements,
        "final_analysis": build_analysis(summaries),
        "tasks": tasks,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    with REPORT_PATH.open("x", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(json.dumps({
        "summary": summaries,
        "final_analysis": report["final_analysis"],
        "report": str(REPORT_PATH),
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)

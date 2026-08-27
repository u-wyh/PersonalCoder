# Phase 3.5 Official Sample Verification

Every response is recompiled unchanged and run against cached official samples. Output comparison is whitespace-token based.

| Metric | Count |
| --- | ---: |
| total_pairs | 1811 |
| compiled | 1811 |
| pairs_with_executable_samples | 1808 |
| pairs_without_executable_samples | 3 |
| all_samples_passed | 1623 |
| partial_sample_failure | 27 |
| all_samples_failed | 158 |
| pairs_with_runtime_error | 37 |
| pairs_with_time_limit | 3 |
| pairs_with_output_limit | 0 |
| pairs_with_unparseable_sample | 0 |

All-sample pass rate among verifiable pairs: **89.77%**.

## Deterministic random manual review

Reviewed 20 failures; categories: `{"code_error": 10, "interactive_problem": 1, "multiple_testcase_format": 1, "problem_id_mapping_error": 3, "special_checker": 5}`.
- **codeforces_626E** — `code_error`: The algorithm emits a prefix-sum debug line before the required answer, so every official sample is rejected despite the later construction matching.
- **codeforces_631D** — `problem_id_mapping_error`: The response is a virtual-tree/LCA solution and is unrelated to the compressed-string matching statement; sample execution is killed.
- **codeforces_679A** — `interactive_problem`: This is an interactive query problem. A static stdin/stdout sample transcript cannot verify the otherwise matching interactive implementation.
- **luogu_P11226** — `special_checker`: The task accepts any valid alphabet permutation per team. The generated permutations differ from the sample and require semantic validation, not token equality.
- **luogu_P11277** — `special_checker`: The task accepts any constructed sequence satisfying the pair-count constraint; comparison with one sample construction is not a validity checker.
- **luogu_P11888** — `problem_id_mapping_error`: The response solves an unrelated digit-cost dynamic program and its input contract does not match the Fibonacci-LCM problem.
- **luogu_P12009** — `code_error`: The implementation's DSU/hash update returns Haru for the third query where the two strings become equivalent; this is a concrete algorithm/implementation error.
- **luogu_P1277** — `special_checker`: The actual 4x4 matrix differs from the sample but satisfies every row, column, diagonal, and fixed-cell constraint; any valid matrix is accepted.
- **luogu_P2208** — `code_error`: The graph construction and gravity transitions are unfinished and main returns without printing an answer.
- **luogu_P2323** — `special_checker`: The response outputs a different valid minimum-bottleneck spanning-tree construction with the required number of type-1 roads; exact sample matching is invalid.
- **luogu_P2375** — `code_error`: The required product computation is commented out and the program prints string lengths and prefix-function debug arrays.
- **luogu_P2472** — `code_error`: Grid vertex IDs use n as the row stride instead of m and the escape-boundary test is off by one, producing a wrong max-flow network.
- **luogu_P2783** — `code_error`: The program reads queries using the edge count, builds/traverses the wrong adjacency structure after contraction, and prints decimal rather than binary answers.
- **luogu_P3848** — `code_error`: The DFS walks to adjacent zero cells instead of jumping across one or more occupied cells to the next zero cell, solving a different movement rule.
- **luogu_P5676** — `multiple_testcase_format`: Each test case contains only N followed by two arrays, but the program reads an extra m and shifts the entire remaining input, causing a crash.
- **luogu_P6306** — `code_error`: The submitted main function is empty and produces no answer.
- **luogu_P7245** — `problem_id_mapping_error`: The response is an unrelated airport/min-cost-flow program rather than the modular expectation calculation in the statement.
- **luogu_P7687** — `special_checker`: The reported bridge set is identical to the sample up to edge orientation and line order, both explicitly allowed by the statement.
- **luogu_P8096** — `code_error`: The dynamic-programming loops are incomplete and the program exits without computing or printing the requested count.
- **luogu_P8765** — `code_error`: The segment-tree implementation has invalid child-index aggregation and missing return paths; the official sample crashes with SIGSEGV.

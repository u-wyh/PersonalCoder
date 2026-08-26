# PersonalCoder Instruction SFT Dataset v1

Local-only dataset audit; no statement was generated from code and no website was crawled.

| Metric | Count |
| --- | ---: |
| Deduplicated historical code | 3261 |
| Codes with problem ID | 2652 |
| Unique identified problems | 2127 |
| Reliable local statements | 0 |
| Benchmark-contaminated code excluded | 14 |
| C++17 compile pass | 2070 |
| C++17 compile fail | 51 |
| C++17 compile timeout | 0 |
| Final instruction-response pairs | 0 |
| Train / validation | 0 / 0 |

## Distributions

- Sources: {}
- Age buckets: {}
- Response tokens: {'count': 0, 'min': 0, 'median': 0, 'p90': 0, 'max': 0, 'mean': 0.0}

## Selection and leakage policy

- One compile-passed response per `(source, problem_id)`, ranked by recency, current style, then completeness.
- The audited 30-problem Benchmark is excluded by ID, SHA256, path-derived ID, code similarity, and statement similarity.
- Splits are deterministic by problem ID; no problem ID can cross train/validation.
- `verified=true` means local C++17 compilation only; no offline tests or official OJ AC status were available.

## Decision

- Threshold met: False
- Recommendation: `do_not_train_collect_real_statements_and_problem_mappings`

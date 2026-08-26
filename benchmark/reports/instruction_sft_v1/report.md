# Phase 3.4 Instruction-SFT-v1 Benchmark

Frozen audited 30-problem / 138-test benchmark; P0; identical tokenizer, chat template, quantization, greedy generation (`max_new_tokens=1024`) and Judge. Old P0 outputs are reused.

## Overall

| Model | Compile | Compile Rate | Offline AC | AC Rate |
| --- | ---: | ---: | ---: | ---: |
| Base | 23/30 | 76.67% | 5/30 | 16.67% |
| LoRA-512 | 23/30 | 76.67% | 8/30 | 26.67% |
| LoRA-1536 | 25/30 | 83.33% | 7/30 | 23.33% |
| Instruction-SFT-v1 | 27/30 | 90.00% | 6/30 | 20.00% |

## Difficulty

| Model | Easy AC | Medium AC | Hard AC |
| --- | ---: | ---: | ---: |
| Base | 3/12 | 2/12 | 0/6 |
| LoRA-512 | 6/12 | 2/12 | 0/6 |
| LoRA-1536 | 5/12 | 2/12 | 0/6 |
| Instruction-SFT-v1 | 4/12 | 2/12 | 0/6 |

## Source AC

Instruction-SFT-v1: Luogu 4/10, Codeforces 2/10, ICPC 0/10.

## Failure diagnosis

Instruction-SFT-v1 submission status: AC 6, CE 3, RE 4, WA 17, TLE 0, OLE 0.
CE categories: `{"missing_include": 0, "undeclared_identifier": 1, "syntax_error": 1, "type_error": 0, "template_error": 1, "other": 0}`.
The smoke missing-header issue is not systematic: none of the 30 formal outputs fails solely from a missing include. The three CEs are nonexistent `std::split` (p011), streaming a vector (p017), and a 1024-token truncation (p030).

## Paired AC

- Base → Instruction: gained ['p008', 'p014'], lost ['p019'], net +1.
- LoRA-512 → Instruction: gained ['p014'], lost ['p005', 'p012', 'p019'], net -2.
- LoRA-1536 → Instruction: gained ['p014'], lost ['p005', 'p019'], net -1.
- The unique all-old-fail recovery is p014; p019 is the only Base AC regression.

## Instruction following

All four models use Markdown fences for 30/30 outputs; strict unfenced code-only is 0/30, extra prose is 0/30, and extraction succeeds 30/30. Instruction-SFT-v1 therefore does not improve literal P0 code-only adherence. One Instruction output (p030) reaches the 1024-token cap.

## Style side effect

Instruction-SFT-v1 partially learns personal style: `using namespace std` 46.67% vs Base 23.33%; MAX constants 23.33% vs 6.67%; fixed arrays 16.67% vs 0.00%. It does not increase bits/stdc++.h, fast IO, or long long in this benchmark.

## Answers and decision

1. **Compile is higher than Base:** 90.00% vs 76.67% (+4 submissions).
2. **Offline AC is only slightly higher than Base:** 20.00% vs 16.67% (+1 net AC).
3. **It is below Style-LoRA-512:** 6 vs 8 AC.
4. **It is below Style-LoRA-1536:** 6 vs 7 AC, despite higher compile rate.
5. **Easy changes most:** 4 vs Base 3; Medium stays 2 and Hard stays 0.
6. **Error change:** CE and RE each fall by 4 vs Base, while WA rises by 7; semantic correctness is the bottleneck.
7. **Smoke missing declarations are not systemic** in the formal set; API/type hallucinations and one truncation remain.
8. **Instruction following does not improve** under the literal unfenced-code criterion; all models fence every output.
9. **Personal style is learned partially**, especially namespace/MAX/fixed-array features.
10. **The experiment does not prove Instruction SFT is more effective than pure Style LoRA:** compile improves, but AC remains below both Style adapters.

Decision: this is case C. Do not start Style+Instruction yet. Phase 3.5 should diagnose training targets and semantic data quality, stratify by problem difficulty/response truncation, audit erroneous historical solutions, and test whether 1.5B model capacity limits algorithm reasoning—without changing this frozen result.

Generation completeness: 30/30; configuration `{"max_new_tokens": 1024, "do_sample": false, "temperature": 0.0, "top_p": 1.0, "num_beams": 1, "repetition_penalty": 1.0, "use_cache": true}`.

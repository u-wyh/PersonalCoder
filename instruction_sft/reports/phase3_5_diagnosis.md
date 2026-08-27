# Phase 3.5 Instruction-SFT Diagnosis v1

## Executive conclusion

Instruction-SFT-v1 improves local code regularity and therefore compilation, but its targets are not uniformly AC-grounded and are heavily concentrated in high-difficulty, long historical solutions. The evidence does **not** support truncation or loss masking as the primary cause. The first corrective action is a semantic-clean dataset, not another training run or a larger model.

## Evidence summary

| Area | Result | Interpretation |
| --- | --- | --- |
| Actual truncation | 202 overlength train pairs; only 12 assistant responses lose tokens/EOS | H1 is real but small (0.74% of train), not the main explanation |
| Response-preserving simulation | 1618/1630 responses fully retained; 12 responses themselves exceed 2048 | Phase 3.3 already used response-preserving truncation for ordinary cases |
| Official samples | 1808 verifiable; 1623 pass all samples (89.77%) | Most pairs have sample-level support, but samples are weak AC evidence |
| Semantic confidence | A 1623, B 3, C 136, D 49 | At least 49 clear bad targets exist; another 136 remain uncertain/special |
| Manual failures | 20 reviewed: code error 10, mapping error 3, input format 1, special checker 5, interactive 1 | Failures are a mix of true contamination and verifier limitations |
| Loss mask | 20/20 valid (10 truncated, 10 untruncated) | User/padding tokens are masked; assistant code receives loss |
| Difficulty | Easy 49, Medium 570, Hard 1188, Unknown 4 | The corpus is not too easy; it is strongly skewed toward hard problems |
| Length | ≤512: 765; 513–1024: 713; >1024: 333 | The corpus is not dominated solely by tiny implementations |
| Benchmark outcome | compile 27/30, AC 6/30; CE/RE fall while WA rises to 17 | SFT learned form and executable structure more readily than semantics |

## H1 — response truncation

Weakly supported, not primary. The formal collator is not ordinary whole-dialog right truncation. It preserves the response and trims the instruction when possible. Of 202 overlength train pairs, 190 retain a complete assistant response and 12 lose response tails and EOS. Removing or separately handling those 12 is worthwhile in a future rebuild, but it cannot plausibly explain the broad 17-WA pattern by itself.

## H2 — semantic data quality

Supported and highest-priority. Compilation was the original response-quality gate, but compilation is not correctness. Official-sample execution finds 185 non-all-pass pairs. Some are false negatives caused by interaction or non-unique output; however, manual review also finds unrelated solutions, empty/incomplete programs, debug output, incorrect input contracts, crashes, and concrete algorithm mistakes. Conservative labelling still assigns 49 pairs to D and 136 to C. Conversely, A means only that cached samples pass—not that the code passes hidden OJ tests. Thus semantic noise is certainly present and the measured A rate is an upper bound on confidence, not an AC rate.

## H3 — difficulty and algorithm distribution

The proposed “data is too easy” hypothesis is rejected. Official metadata maps 65.60% of pairs to hard and only 2.71% to easy, while the frozen benchmark is 40% easy, 40% medium, and 20% hard. The gap is substantial but reversed: training is much harder and longer (median response 591 tokens versus benchmark reference median 99). Such a corpus can be inefficient supervision for a 1.5B model: it exposes complex finished code without intermediate reasoning or reliable semantic validation.

Algorithm balance cannot be fully established from local authoritative metadata. The 1731 Luogu pages expose numeric tag IDs without a local name dictionary, so they remain unknown instead of being guessed. Named Codeforces tags cover 79 pairs; within that subset, data structures, graph, search, DP, and math are represented. This limited tag coverage prevents a claim that any algorithm family dominates all 1811 pairs.

## H4 — what the SFT loss teaches

Strongly supported. The assistant-only mask is correct, so loss directly imitates every target code token. Token-level next-token loss provides dense supervision for includes, declarations, control-flow syntax, variable patterns, and complete program structure. Algorithm selection and hidden-case correctness provide no direct signal: a semantically wrong but fluent target is rewarded identically. Phase 3.4 matches this mechanism—compile rises from 23/30 to 27/30, but the error mass moves toward WA (17) and AC reaches only 6/30.

## H5 — 1.5B capacity

Plausible secondary limit, not yet established as the primary bottleneck. Medium remains 2/12 and Hard 0/6, which is consistent with limited reasoning capacity. But semantic contamination, weak sample-only validation, an over-hard/long corpus, and one-shot answer imitation are unresolved confounders. The evidence order therefore does not justify changing model size now.

## Train–Benchmark relationship

There is a clear distribution gap, but the Benchmark is not demonstrably harder by official difficulty. Training has far more hard problems and much longer statements/solutions, whereas the Benchmark contains concise, deliberately balanced tasks and stronger fixed tests. The key mismatch is that training targets are historical final code with uneven semantic confidence, while evaluation demands hidden-case generalization.

## Phase 3.6 decision

Build a **semantic-clean Instruction dataset** as the single next direction. Keep A pairs, separately review B/C, exclude confirmed D from the candidate training split, handle interactive/special-checker tasks explicitly, and preserve the existing Benchmark boundary. Do not train v2 until the resulting dataset card reports auditable semantic gates. This decision addresses the earliest demonstrated failure in the required evidence order and leaves model-capacity testing for later.

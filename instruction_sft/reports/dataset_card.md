# PersonalCoder Instruction SFT Dataset v1

Real public problem statements paired with the user's compile-passed historical solutions. No statement is synthesized or rewritten.

| Metric | Count |
| --- | ---: |
| Eligible selected codes | 1849 |
| Locatable statements | 1848 |
| Fetch success | 1847 / 1848 |
| Verified statements | 1811 |
| Manual review required / excluded | 37 |
| Benchmark contamination excluded | 0 |
| Final instruction-response pairs | 1811 |
| Train / validation | 1630 / 181 |

## Provenance and validation

- Statements were acquired from public Codeforces/Codeforces Gym and Luogu problem pages with a low-rate, resumable cache.
- A deterministic 100-problem pilot passed the expansion gate; ten pilot pairs were manually checked against their code.
- Failed, short, incomplete, ID-mismatched, login/challenge, or error pages are excluded rather than repaired synthetically.
- Every response was already SHA256-deduplicated and passed local `g++ -std=c++17 -O2 -pipe -fsyntax-only` selection in Phase 3.1.
- Held-out benchmark contamination is checked by source/problem ID, SHA256, code similarity, and statement similarity.
- Train/validation is a deterministic 90/10 problem-level split with seed 42 and no problem-ID overlap.
- Raw page caches and normalized per-problem Markdown are local ignored artifacts; the committed JSONL preserves the full verified instruction text.

## Distribution

- Source: `{"codeforces": 80, "luogu": 1731}`
- Statement characters: `{"count": 1811, "min": 284, "median": 977, "p90": 2116, "max": 22766, "mean": 1190.65}`
- Response tokens: `{"count": 1811, "min": 73, "median": 591, "p90": 1239, "max": 10096, "mean": 690.03}`

## Training decision

Threshold `>=500` met: **true**. Recommendation: `ready_for_phase3_3_instruction_sft`. This phase does not train a model.

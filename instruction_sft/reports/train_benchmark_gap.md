# Train–Benchmark Gap

## Difficulty

| Split | Easy | Medium | Hard | Unknown |
| --- | ---: | ---: | ---: | ---: |
| Instruction-SFT (1811) | 49 | 570 | 1188 | 4 |
| Benchmark (30) | 12 | 12 | 6 | 0 |

The training corpus is historical accepted/compiled-looking code paired with statements, not curated reasoning supervision. Its official difficulty mix is measurable, but named algorithm coverage is reliable only for 79/1811 Codeforces-tagged pairs. The benchmark intentionally fixes a 12/12/6 easy/medium/hard mix and tests unseen full solutions under stronger tests.

## Length and task shape

- Training response tokens: `{'min': 73, 'mean': 690.03, 'p50': 591, 'p90': 1239, 'p95': 1452, 'max': 10096}`.
- Benchmark reference-code tokens: `{'min': 39, 'mean': 158.8, 'p50': 99, 'p90': 388, 'p95': 396, 'max': 455}`.
- Training instruction tokens: `{'min': 180, 'mean': 672.06, 'p50': 586, 'p90': 1065, 'p95': 1309, 'max': 20368}`.
- Benchmark statement tokens: `{'min': 27, 'mean': 52.77, 'p50': 51, 'p90': 68, 'p95': 69, 'max': 86}`.

The dominant gap is supervision quality rather than proof of a pure difficulty mismatch: passing a few official samples does not establish AC, and the corpus includes demonstrably wrong/mismatched programs. Medium/Hard benchmark performance may also expose model-capacity limits, but capacity cannot be isolated before semantic cleaning.

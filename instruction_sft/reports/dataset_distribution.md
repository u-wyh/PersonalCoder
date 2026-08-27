# Instruction-SFT Dataset Distribution

Pairs: **1811**; sources: `{'luogu': 1731, 'codeforces': 80}`.

## Difficulty

| Band | Count | Rate |
| --- | ---: | ---: |
| easy | 49 | 2.71% |
| medium | 570 | 31.47% |
| hard | 1188 | 65.60% |
| unknown | 4 | 0.22% |

Difficulty mapping: Luogu 1–2 easy, 3–4 medium, 5–7 hard; Codeforces ≤1200 easy, 1300–1900 medium, ≥2000 hard. Missing metadata stays unknown.

## Response structure

- Response tokens: `{'min': 73, 'mean': 690.03, 'p50': 591, 'p90': 1239, 'p95': 1452, 'max': 10096}`
- Code lines: `{'min': 13, 'mean': 102.7, 'p50': 87, 'p90': 194, 'p95': 224, 'max': 514}`
- Functions: `{'min': 0, 'mean': 4.94, 'p50': 4, 'p90': 10, 'p95': 13, 'max': 26}`
- Template declarations: `{'min': 0, 'mean': 0.0, 'p50': 0, 'p90': 0, 'p95': 0, 'max': 3}`
- Length bands: `{'short_<=512': 765, 'medium_513_1024': 713, 'long_>1024': 333}`

## Algorithm metadata

Reliable named tags cover **79/1811** pairs. Luogu cached pages expose numeric tag IDs without a local name dictionary, so they are deliberately not guessed.

Canonical categories (multi-label; untagged pairs count as other/unknown): `{'other/unknown': 1732, 'search': 26, 'math': 22, 'graph': 38, 'greedy': 10, 'data_structure': 44, 'implementation': 17, 'string': 6, 'dp': 25}`
Codeforces tags: `{'data structures': 44, 'trees': 26, 'dp': 25, 'binary search': 17, 'dfs and similar': 17, 'divide and conquer': 16, 'graphs': 15, 'math': 13, 'brute force': 10, 'greedy': 10, 'implementation': 9, 'dsu': 8, 'combinatorics': 8, 'bitmasks': 7, 'sortings': 7, 'hashing': 6, 'shortest paths': 5, 'constructive algorithms': 4, 'number theory': 4, 'probabilities': 3, 'strings': 3, '2-sat': 3, 'games': 3, 'meet-in-the-middle': 3, 'interactive': 2, 'string suffix structures': 2, 'ternary search': 1, 'fft': 1, 'geometry': 1}`

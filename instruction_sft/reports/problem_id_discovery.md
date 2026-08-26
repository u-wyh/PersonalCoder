# Instruction SFT Problem ID Discovery

This audit is local-only. IDs are accepted only from an unambiguous filename, path, header, Git subject, or explicit ICPC event path.

| Metric | Count |
| --- | ---: |
| Deduplicated C++ samples | 3261 |
| Identified samples | 2652 |
| Unique problems | 2127 |
| IDs with multiple code versions | 414 |
| Unknown samples | 609 |

## Source distribution

- luogu: 2522
- codeforces: 130
- icpc: 0
- unknown: 609

## Time evidence

- SHA-matched filesystem timestamps: 3261
- Local Git timestamps: 3260

Ambiguous identifiers are retained in the index as metadata but are not promoted to formal problem mappings.

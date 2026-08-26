# Instruction SFT v1 Token Lengths

Tokenizer: `/data/PersonalCoder/model`; train samples: 1630; percentiles use nearest-rank.

| Component | P50 | P75 | P90 | P95 | P99 | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| instruction_tokens | 586 | 790 | 1074 | 1319 | 1868 | 20368 |
| response_tokens | 586 | 893 | 1234 | 1442 | 1991 | 10096 |
| total_chat_tokens | 1270 | 1669 | 2199 | 2491 | 3330 | 20912 |

| Max sequence length | Truncated | Truncation rate | Coverage |
| ---: | ---: | ---: | ---: |
| 1024 | 1107 | 67.91% | 32.09% |
| 1536 | 527 | 32.33% | 67.67% |
| 2048 | 202 | 12.39% | 87.61% |
| 3072 | 30 | 1.84% | 98.16% |
| 4096 | 7 | 0.43% | 99.57% |

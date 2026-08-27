# Phase 3.5 Truncation Analysis

Train samples: 1630; max sequence length: 2048.

## Actual Phase 3.3 policy

response-preserving: keep the full assistant response when response_tokens < max_seq_length; trim the instruction head/tail to fit; if response alone is too long, retain up to 256 prompt tokens and the response prefix

- A untruncated: 1428
- B instruction-only truncation, complete response: 190
- C response truncated: 12
- D current response retained <=10%: 0
- E assistant EOS truncated: 12
- Removed response tokens: total 23634, mean 1969.5, P50/P90/P95 695/8306/8306

## Counterfactual ordinary right truncation

It would truncate 202 responses; 13 would retain at most 10% of the response.

## Ideal response-first simulation

- Fully preserved responses: 1618 / 1630
- Response >2048: 12; max response: 10098
- Mean retained instruction tokens among originally truncated samples: 719.32

The training pipeline was already response-preserving for ordinary overlength pairs; only responses that themselves exceed the budget lose code tails/EOS.

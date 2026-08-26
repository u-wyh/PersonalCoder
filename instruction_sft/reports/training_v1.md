# Instruction SFT v1 Training Report

Mode: `full`; completed: **True**.

| Metric | Value |
| --- | ---: |
| Train samples | 1630 |
| Validation samples | 181 |
| Max sequence length | 2048 |
| Truncated train samples | 202 (12.39%) |
| Optimizer steps | 204 |
| Train loss | 0.602804 |
| Eval loss | 0.595376 |
| Learning rate | 0.0001 |
| Training seconds | 2161.78 |
| Peak allocated VRAM MiB | 5937.33 |
| Peak reserved VRAM MiB | 15208.00 |
| Final adapter | `/data/PersonalCoder/checkpoints/rtx4060/instruction_sft_v1/final_adapter` |

Assistant-only mask audit: `{"checked_samples": 16, "prompt_labels_all_masked": true, "assistant_labels_present": true}`

Fresh adapter reload: `{"fresh_base_loaded": true, "adapter_reloaded": true, "trainable_parameters": 0}`

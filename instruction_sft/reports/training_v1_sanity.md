# Instruction SFT v1 Training Report

Mode: `sanity`; completed: **True**.

| Metric | Value |
| --- | ---: |
| Train samples | 16 |
| Validation samples | 8 |
| Max sequence length | 2048 |
| Truncated train samples | 4 (25.00%) |
| Optimizer steps | 8 |
| Train loss | 0.614175 |
| Eval loss | 0.795493 |
| Learning rate | 0.0001 |
| Training seconds | 16.91 |
| Peak allocated VRAM MiB | 5934.45 |
| Peak reserved VRAM MiB | 9374.00 |
| Final adapter | `/data/PersonalCoder/checkpoints/rtx4060/instruction_sft_v1_sanity/final_adapter` |

Assistant-only mask audit: `{"checked_samples": 16, "prompt_labels_all_masked": true, "assistant_labels_present": true}`

Fresh adapter reload: `{"fresh_base_loaded": true, "adapter_reloaded": true, "trainable_parameters": 0}`

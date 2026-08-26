# PersonalCoder Pilot Benchmark Results

## Overall

| Model | Compile Rate | Offline AC Rate |
| --- | ---: | ---: |
| Base | 23/30 (76.67%) | 5/30 (16.67%) |
| LoRA-512 | 23/30 (76.67%) | 8/30 (26.67%) |
| LoRA-1536 | 25/30 (83.33%) | 7/30 (23.33%) |

## AC by Difficulty

| Model | Easy | Medium | Hard |
| --- | ---: | ---: | ---: |
| Base | 3/12 (25.00%) | 2/12 (16.67%) | 0/6 (0.00%) |
| LoRA-512 | 6/12 (50.00%) | 2/12 (16.67%) | 0/6 (0.00%) |
| LoRA-1536 | 5/12 (41.67%) | 2/12 (16.67%) | 0/6 (0.00%) |

## Full vs Clean Offline AC

| Model | Full AC | Clean AC |
| --- | ---: | ---: |
| Base | 5/30 (16.67%) | 4/24 (16.67%) |
| LoRA-512 | 8/30 (26.67%) | 7/24 (29.17%) |
| LoRA-1536 | 7/30 (23.33%) | 6/24 (25.00%) |

Contaminated-record subset retained in Full and excluded from Clean: p013, p015, p016, p017, p025, p026.

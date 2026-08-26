# PersonalCoder Pilot Benchmark Results

## Overall

| Model | Compile Rate | Offline AC Rate |
| --- | ---: | ---: |
| Base | 23/30 (76.67%) | 5/30 (16.67%) |
| LoRA-512 | 23/30 (76.67%) | 6/30 (20.00%) |
| LoRA-1536 | 25/30 (83.33%) | 5/30 (16.67%) |

## AC by Difficulty

| Model | Easy | Medium | Hard |
| --- | ---: | ---: | ---: |
| Base | 3/12 (25.00%) | 2/12 (16.67%) | 0/6 (0.00%) |
| LoRA-512 | 5/12 (41.67%) | 1/12 (8.33%) | 0/6 (0.00%) |
| LoRA-1536 | 4/12 (33.33%) | 1/12 (8.33%) | 0/6 (0.00%) |

## Full vs Clean Offline AC

| Model | Full AC | Clean AC |
| --- | ---: | ---: |
| Base | 5/30 (16.67%) | 5/24 (20.83%) |
| LoRA-512 | 6/30 (20.00%) | 6/24 (25.00%) |
| LoRA-1536 | 5/30 (16.67%) | 5/24 (20.83%) |

Contaminated-record subset retained in Full and excluded from Clean: p013, p015, p016, p017, p025, p026.

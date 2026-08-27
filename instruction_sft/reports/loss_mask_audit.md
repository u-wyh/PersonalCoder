# Phase 3.5 Loss Mask Audit

Seed: 42; 10 truncated + 10 untruncated formal train samples.

Overall: **PASS**. Prompt labels all masked: True; assistant labels active: True; padding labels masked: True.

| ID | Truncated | Input tokens | Assistant start | Masked | Valid loss | Prompt masked | Assistant active |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| luogu_P8475 | True | 2048 | 985 | 985 | 1063 | True | True |
| luogu_P8875 | True | 2048 | 1262 | 1262 | 786 | True | True |
| luogu_P9120 | True | 2048 | 615 | 615 | 1433 | True | True |
| luogu_P14420 | True | 2048 | 1210 | 1210 | 838 | True | True |
| luogu_P9869 | True | 2048 | 1501 | 1501 | 547 | True | True |
| luogu_P9352 | True | 2048 | 1149 | 1149 | 899 | True | True |
| luogu_P8310 | True | 2048 | 848 | 848 | 1200 | True | True |
| luogu_P4012 | True | 2048 | 667 | 667 | 1381 | True | True |
| luogu_P4899 | True | 2048 | 256 | 256 | 1792 | True | True |
| luogu_P8818 | True | 2048 | 1013 | 1013 | 1035 | True | True |
| luogu_P6086 | False | 1446 | 764 | 764 | 682 | True | True |
| codeforces_293E | False | 1362 | 486 | 486 | 876 | True | True |
| luogu_P11562 | False | 1338 | 874 | 874 | 464 | True | True |
| luogu_P9149 | False | 1988 | 1018 | 1018 | 970 | True | True |
| luogu_P1621 | False | 732 | 385 | 385 | 347 | True | True |
| luogu_P2057 | False | 1167 | 417 | 417 | 750 | True | True |
| luogu_P1948 | False | 1063 | 557 | 557 | 506 | True | True |
| luogu_P2523 | False | 1055 | 617 | 617 | 438 | True | True |
| luogu_P3181 | False | 1742 | 209 | 209 | 1533 | True | True |
| luogu_P6035 | False | 1765 | 858 | 858 | 907 | True | True |

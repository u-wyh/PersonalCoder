# PersonalCoder Prompt Ablation

## Overall

| Prompt | Model | Compile | Offline AC |
| --- | --- | ---: | ---: |
| P0 | Base | 23/30 (76.67%) | 5/30 (16.67%) |
| P0 | LoRA-512 | 23/30 (76.67%) | 8/30 (26.67%) |
| P0 | LoRA-1536 | 25/30 (83.33%) | 7/30 (23.33%) |
| P1 | Base | 21/30 (70.00%) | 4/30 (13.33%) |
| P1 | LoRA-512 | 22/30 (73.33%) | 6/30 (20.00%) |
| P1 | LoRA-1536 | 24/30 (80.00%) | 7/30 (23.33%) |
| P2 | Base | 24/30 (80.00%) | 4/30 (13.33%) |
| P2 | LoRA-512 | 24/30 (80.00%) | 7/30 (23.33%) |
| P2 | LoRA-1536 | 25/30 (83.33%) | 7/30 (23.33%) |

## AC by Difficulty

| Prompt | Model | Easy | Medium | Hard |
| --- | --- | ---: | ---: | ---: |
| P0 | Base | 3/12 | 2/12 | 0/6 |
| P0 | LoRA-512 | 6/12 | 2/12 | 0/6 |
| P0 | LoRA-1536 | 5/12 | 2/12 | 0/6 |
| P1 | Base | 3/12 | 1/12 | 0/6 |
| P1 | LoRA-512 | 4/12 | 2/12 | 0/6 |
| P1 | LoRA-1536 | 4/12 | 3/12 | 0/6 |
| P2 | Base | 2/12 | 2/12 | 0/6 |
| P2 | LoRA-512 | 5/12 | 2/12 | 0/6 |
| P2 | LoRA-1536 | 5/12 | 2/12 | 0/6 |

## Failure distribution

| Prompt | Model | CE | RE | TLE | OLE | WA | AC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P0 | Base | 7 | 3 | 0 | 0 | 15 | 5 |
| P0 | LoRA-512 | 7 | 1 | 0 | 0 | 14 | 8 |
| P0 | LoRA-1536 | 5 | 1 | 0 | 0 | 17 | 7 |
| P1 | Base | 9 | 1 | 0 | 0 | 16 | 4 |
| P1 | LoRA-512 | 8 | 1 | 0 | 0 | 15 | 6 |
| P1 | LoRA-1536 | 6 | 3 | 0 | 0 | 14 | 7 |
| P2 | Base | 6 | 1 | 0 | 1 | 18 | 4 |
| P2 | LoRA-512 | 6 | 1 | 0 | 0 | 16 | 7 |
| P2 | LoRA-1536 | 5 | 2 | 0 | 0 | 16 | 7 |

Aggregate across the three models:

- P0: {'AC': 20, 'CE': 19, 'RE': 5, 'WA': 46}
- P1: {'AC': 17, 'CE': 23, 'RE': 5, 'WA': 45}
- P2: {'AC': 18, 'CE': 17, 'OLE': 1, 'RE': 4, 'WA': 50}

## Paired problem-level changes

### Base

- P0 → P1: gained=['p008']; lost=['p003', 'p019']; net=-1; status changes={'AC->WA': 2, 'CE->RE': 1, 'CE->WA': 3, 'RE->AC': 1, 'RE->CE': 2, 'WA->CE': 4}
- P0 → P2: gained=none; lost=['p003']; net=-1; status changes={'AC->WA': 1, 'CE->WA': 4, 'RE->CE': 2, 'WA->CE': 1, 'WA->OLE': 1}
- P1 → P2: gained=['p019']; lost=['p008']; net=+0; status changes={'AC->RE': 1, 'CE->WA': 4, 'RE->WA': 1, 'WA->AC': 1, 'WA->CE': 1, 'WA->OLE': 1}

### LoRA-512

- P0 → P1: gained=none; lost=['p008', 'p012']; net=-2; status changes={'AC->WA': 2, 'CE->WA': 3, 'WA->CE': 4}
- P0 → P2: gained=none; lost=['p008']; net=-1; status changes={'AC->WA': 1, 'CE->WA': 2, 'WA->CE': 1}
- P1 → P2: gained=['p012']; lost=none; net=+1; status changes={'CE->WA': 3, 'WA->AC': 1, 'WA->CE': 1}

### LoRA-1536

- P0 → P1: gained=['p009', 'p014']; lost=['p005', 'p008']; net=+0; status changes={'AC->RE': 1, 'AC->WA': 1, 'CE->AC': 1, 'CE->RE': 1, 'CE->WA': 1, 'WA->AC': 1, 'WA->CE': 4}
- P0 → P2: gained=['p012']; lost=['p008']; net=+0; status changes={'AC->WA': 1, 'RE->WA': 1, 'WA->AC': 1, 'WA->RE': 2}
- P1 → P2: gained=['p005', 'p012']; lost=['p009', 'p014']; net=+0; status changes={'AC->CE': 1, 'AC->WA': 1, 'CE->RE': 1, 'CE->WA': 3, 'RE->AC': 1, 'RE->CE': 1, 'RE->WA': 1, 'WA->AC': 1, 'WA->CE': 1, 'WA->RE': 1}

## Previously common P0 failures recovered

P0 common failures (22): ['p002', 'p004', 'p006', 'p007', 'p009', 'p011', 'p014', 'p015', 'p016', 'p017', 'p018', 'p020', 'p021', 'p022', 'p023', 'p024', 'p025', 'p026', 'p027', 'p028', 'p029', 'p030']
- P1/Base: none
- P1/LoRA-512: none
- P1/LoRA-1536: ['p009', 'p014']
- P2/Base: none
- P2/LoRA-512: none
- P2/LoRA-1536: none

## Key code-level observations

- Base/P1/p008 changes a hard-coded assertion harness into the required stdin/stdout solution and becomes AC; this is a genuine instruction-following repair.
- Base/P1/p003 replaces the required abbreviation count with asterisks, while Base/P1/p019 replaces the input-driven solution with a hard-coded self-check. Both were P0 AC and become WA, showing that a longer checklist can distract generation.
- LoRA-1536/P1/p009 removes unsolicited Chinese input prompts and becomes AC; p014 replaces the conflicting `rank` identifier with a parent vector and changes CE to AC. These are output/implementation repairs rather than new algorithms.
- LoRA-512 and LoRA-1536 both lose p008 under enhanced prompts after replacing the correct direct loop with incorrect digit-vector simulations.
- LoRA-1536/P2/p012 restores pair-by-pair `long long` absolute differences and becomes AC. Base/P2/p028 misparses the multi-case input and floods output until OLE, so stronger wording did not ensure constraint compliance.

## Auxiliary style check

Rates below are Style-All and are not optimization targets.

| Prompt | Model | using namespace std | MAX constant | static array |
| --- | --- | ---: | ---: | ---: |
| P0 | Base | 7/30 | 2/30 | 0/30 |
| P0 | LoRA-512 | 17/30 | 6/30 | 6/30 |
| P0 | LoRA-1536 | 10/30 | 5/30 | 5/30 |
| P1 | Base | 8/30 | 4/30 | 1/30 |
| P1 | LoRA-512 | 25/30 | 6/30 | 6/30 |
| P1 | LoRA-1536 | 17/30 | 5/30 | 5/30 |
| P2 | Base | 12/30 | 2/30 | 0/30 |
| P2 | LoRA-512 | 24/30 | 6/30 | 5/30 |
| P2 | LoRA-1536 | 16/30 | 6/30 | 6/30 |

## Required answers

1. **P1 is not better than P0.** AC deltas are Base -1, LoRA-512 -2, LoRA-1536 0; aggregate AC falls from 20/90 to 17/90.
2. **P2 is not better than P0.** AC deltas are Base -1, LoRA-512 -1, LoRA-1536 0; aggregate AC falls to 18/90.
3. **P0 has the highest aggregate AC:** 20/90; totals={'P0': 20, 'P1': 17, 'P2': 18}.
4. **Base sensitivity is small in total but material per problem:** AC is 5/4/4 (range 1), with P1 gaining p008 but losing p003 and p019.
5. **LoRA-512 is the most prompt-sensitive:** AC is 8/6/7 (range 2); P1 loses p008 and p012, while P2 still loses p008.
6. **LoRA-1536 is aggregate-stable but not problem-stable:** AC remains 7/7/7, while P1 gains p009/p014 and loses p005/p008; P2 gains p012 and loses p008.
7. **Both LoRAs exceed Base under every prompt:** True. This is stable within this audited 30-problem sample, not proof of a general capability gain.
8. **Enhanced prompts do not reduce semantic failures.** P1 raises aggregate CE from 19 to 23 and lowers AC by 3. P2 lowers CE/RE to 17/4 from 19/5, but WA rises from 46 to 50 (plus one OLE) and AC falls by 2; its gains are compile-side, not correctness-side.
9. **There are cross-model trade-offs.** P1 recovers two common failures only for LoRA-1536, yet lowers Base and LoRA-512 AC; enhanced prompts also exchange different AC identities even when LoRA-1536's total is unchanged.
10. **The bottleneck is not Prompt alone.** The evidence points mainly to base algorithmic/semantic capability and the instruction-training data form; Style LoRA changes code priors but does not reliably repair reasoning. Stop prompt tuning, validate the persistent LoRA advantage on a larger audited benchmark, and prioritize a controlled Instruction SFT if training is resumed.

# Instruction SFT v1 Paired Comparison

同一 audited 30题、P0、生成参数与 Judge 的逐题比较。

| Problem | Difficulty | Base | LoRA-512 | LoRA-1536 | Instruction-SFT-v1 |
| --- | --- | ---: | ---: | ---: | ---: |
| p001 | easy | AC | AC | AC | AC |
| p002 | easy | WA | CE | WA | WA |
| p003 | easy | AC | AC | AC | AC |
| p004 | easy | CE | WA | WA | WA |
| p005 | easy | CE | AC | AC | WA |
| p006 | easy | CE | WA | WA | WA |
| p007 | easy | CE | WA | WA | WA |
| p008 | easy | RE | AC | AC | AC |
| p009 | easy | WA | WA | WA | WA |
| p010 | easy | AC | AC | AC | AC |
| p011 | easy | CE | CE | CE | CE |
| p012 | easy | WA | AC | WA | WA |
| p013 | medium | AC | AC | AC | AC |
| p014 | medium | CE | CE | CE | AC |
| p015 | medium | RE | CE | CE | WA |
| p016 | medium | RE | OLE | OLE | RE |
| p017 | medium | RE | WA | WA | CE |
| p018 | medium | WA | WA | WA | WA |
| p019 | medium | AC | AC | AC | WA |
| p020 | medium | WA | WA | WA | WA |
| p021 | medium | WA | WA | WA | WA |
| p022 | medium | WA | WA | WA | WA |
| p023 | medium | WA | WA | WA | WA |
| p024 | medium | WA | WA | RE | WA |
| p025 | hard | RE | CE | CE | RE |
| p026 | hard | RE | CE | RE | WA |
| p027 | hard | CE | CE | CE | RE |
| p028 | hard | RE | RE | WA | RE |
| p029 | hard | WA | WA | WA | WA |
| p030 | hard | RE | WA | WA | CE |

## Required paired sets

- Base FAIL → Instruction AC：p008, p014
- Base AC → Instruction FAIL：p019
- LoRA-512 FAIL → Instruction AC：p014
- LoRA-512 AC → Instruction FAIL：p005, p012, p019
- LoRA-1536 FAIL → Instruction AC：p014
- LoRA-1536 AC → Instruction FAIL：p005, p019
- 三种旧模型全部 FAIL → Instruction AC：p014
- 四模型全部 FAIL（21）：p002, p004, p006, p007, p009, p011, p015, p016, p017, p018, p020, p021, p022, p023, p024, p025, p026, p027, p028, p029, p030

## Actual code findings

- **p005**：Instruction 将 n 题误读为固定三组各三个数，未按 n 行逐题计数；因此仅过 2/5。两个 Style LoRA 使用逐行三数求和并 AC。
- **p008**：Instruction 正确读取 n,k 并逐次执行末位减一/除十，修复 Base 运行自测断言且不读取输入的问题；三个 LoRA 均 AC。
- **p012**：Instruction 延续 Base 的错误：把两列分别排序后配对，且使用 int 处理 1e18；仅 LoRA-512 按 EOF 使用 long long 原对计算并 AC。
- **p014**：三个旧模型分别因类成员缺失、rank 名称歧义或数组 rank 冲突而 CE；Instruction 采用 MAXN 静态父数组与路径压缩，4/4 AC，是唯一旧模型全失败后的新增 AC。
- **p019**：Instruction 用 ans 记录当前非递减段长度，下降时直接重置，却未保存历史最大值；Base 与两个 Style LoRA 均维护 maxLength 并 AC。
- **p011**：Instruction 调用不存在于 C++17 标准库的 std::split；这是 API/符号幻觉，不是单纯缺少头文件。
- **p017**：Instruction 直接输出 vector<int>，触发 operator<< 模板类型错误。
- **p030**：Instruction 命中 1024-token 上限，输出在 bfs 函数中截断，造成缺分号/右花括号的语法错误。

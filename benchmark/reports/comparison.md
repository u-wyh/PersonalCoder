# Pilot Per-problem Comparison

状态来自对已有生成代码按原 Judge 配置的复评；AC 指固定离线测试集通过。

| Problem | Difficulty | Base | LoRA-512 | LoRA-1536 |
| --- | --- | ---: | ---: | ---: |
| p001 | easy | AC | AC | AC |
| p002 | easy | AC | CE | WA |
| p003 | easy | AC | AC | AC |
| p004 | easy | CE | WA | WA |
| p005 | easy | CE | AC | AC |
| p006 | easy | CE | WA | WA |
| p007 | easy | CE | WA | WA |
| p008 | easy | RE | AC | AC |
| p009 | easy | WA | WA | WA |
| p010 | easy | WA | WA | WA |
| p011 | easy | CE | CE | CE |
| p012 | easy | WA | AC | WA |
| p013 | medium | WA | WA | WA |
| p014 | medium | CE | CE | CE |
| p015 | medium | WA | CE | CE |
| p016 | medium | WA | WA | WA |
| p017 | medium | WA | WA | WA |
| p018 | medium | WA | WA | WA |
| p019 | medium | AC | AC | AC |
| p020 | medium | WA | WA | WA |
| p021 | medium | AC | WA | WA |
| p022 | medium | WA | WA | WA |
| p023 | medium | WA | WA | WA |
| p024 | medium | WA | WA | RE |
| p025 | hard | RE | CE | CE |
| p026 | hard | RE | CE | RE |
| p027 | hard | CE | CE | CE |
| p028 | hard | RE | RE | WA |
| p029 | hard | WA | WA | WA |
| p030 | hard | RE | WA | WA |

## 重点结果组

- Base FAIL → 至少一个 LoRA AC：p005, p008, p012
- Base AC → 至少一个 LoRA FAIL：p002, p021
- 三模型全部 FAIL（22）：p004, p006, p007, p009, p010, p011, p013, p014, p015, p016, p017, p018, p020, p022, p023, p024, p025, p026, p027, p028, p029, p030
- LoRA-512 与 LoRA-1536 状态不同：p002, p012, p024, p026, p028

## 关键代码差异核查

- **p002**：Base 与 LoRA-1536 都把条件误写为可被 4 整除；Base 仅因固定测试未暴露反例且带换行而得到 Offline AC。LoRA-1536 还缺少末尾换行，在严格字节 diff 下 WA；LoRA-512 对 sqrt(w) 使用取模，直接 CE。
- **p005**：Base 调用 std::accumulate 却未包含 <numeric>，CE；两个 LoRA 都改用显式三项求和并 AC。
- **p008**：Base 输出了自测函数和错误断言，未读取题目输入并运行时中止；两个 LoRA 都生成了正确的输入、循环和输出。
- **p012**：Base 与 LoRA-1536 使用 int，并把两列分别排序后再求差，改变了输入配对；LoRA-512 按 EOF 流式读取 long long，唯一 AC。
- **p021**：Base 每次插入后排序并立即输出，固定测试全部通过；LoRA-512 读完后插入额外的 0，LoRA-1536 只整体排序一次且偶数分支可能访问 i+1 越界，二者均 WA。

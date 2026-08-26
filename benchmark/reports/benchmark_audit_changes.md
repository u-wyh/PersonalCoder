# Benchmark Audit v1.1 Changes

## Overall

| Model | v1 Compile | v1.1 Compile | v1 Offline AC | v1.1 Offline AC | v1 Clean AC | v1.1 Clean AC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 23/30 | 23/30 | 5/30 | 5/30 | 5/24 | 4/24 |
| LoRA-512 | 23/30 | 23/30 | 6/30 | 8/30 | 6/24 | 7/24 |
| LoRA-1536 | 25/30 | 25/30 | 5/30 | 7/30 | 5/24 | 6/24 |

旧结果已原样保存在 `benchmark/reports/pilot_v1/`；v1.1 位于 `benchmark/reports/pilot_v1_1/`。没有重新生成模型代码。

## Per-problem changes

所有题目的 checker 都从隐式 `exact` 改为显式 `token`。这只忽略 whitespace 布局，token 内容仍必须完全一致。

| Problem | Tests | Checker | Base | LoRA-512 | LoRA-1536 | Reason |
| --- | ---: | --- | --- | --- | --- | --- |
| p001 | 5→5 | exact→token | AC | AC | AC | 输出为普通数值 token；原测试已含正负边界 |
| p002 | 5→8 | exact→token | **AC→WA** | CE | WA | 加入 4、6、98；6/98 击穿错误的 `%4==0` 条件 |
| p003 | 5→5 | exact→token | AC | AC | AC | 已覆盖长度 10/11、多单词 |
| p004 | 5→5 | exact→token | CE | non-AC | non-AC | 已覆盖零分、并列和 k 边界 |
| p005 | 5→5 | exact→token | CE | AC | AC | 已覆盖 0/1/2/3 人会解组合 |
| p006 | 5→5 | exact→token | CE | non-AC | non-AC | 四种增减语句均已覆盖 |
| p007 | 5→5 | exact→token | CE | non-AC | non-AC | 单数字、重复值、混排已覆盖 |
| p008 | 5→5 | exact→token | RE | AC | AC | 已覆盖末位 0、最大 n、连续操作 |
| p009 | 5→5 | exact→token | non-AC | non-AC | non-AC | 已覆盖阈值等于、全可摘和全不可摘 |
| p010 | 5→5 | exact→token | **WA→AC** | **WA→AC** | **WA→AC** | 原答案仅有空白布局差异；token 语义下正确 |
| p011 | 5→5 | exact→token | CE | CE | CE | 单片段、多片段和长名称已覆盖 |
| p012 | 5→6 | exact→token | non-AC | AC | non-AC | 新增负数、±10^18 和多行 EOF，验证 64 位差值 |
| p013 | 3→4 | exact→token | **WA→AC** | **WA→AC** | **WA→AC** | 排序输出是 token-based；新增重复值与 ±10^9 |
| p014 | 3→4 | exact→token | CE | CE | CE | 新增自查询、重复合并和集合桥接 |
| p015 | 3→4 | exact→token | non-AC | CE | CE | 新增 3×10^9 路径与不可达点，覆盖距离溢出 |
| p016 | 3→4 | exact→token | non-AC | non-AC | non-AC | 新增 3×10^9 区间和及负更新 |
| p017 | 3→4 | exact→token | non-AC | non-AC | non-AC | 新增单字符模式和完全重叠匹配 |
| p018 | 3→4 | exact→token | non-AC | non-AC | non-AC | 新增 k=n；既有测试已覆盖 k=0 与重复边界无解 |
| p019 | 3→5 | exact→token | AC | AC | AC | 新增严格递减和全相等序列 |
| p020 | 3→4 | exact→token | non-AC | non-AC | non-AC | 新增 3×10^9 前缀和与首尾查询 |
| p021 | 3→5 | exact→token | **AC→WA** | non-AC | non-AC | 新增 INT 边界与负重复值，击穿 Base 的 int 中位数溢出 |
| p022 | 3→4 | exact→token | non-AC | non-AC | non-AC | 新增 k=n，强制连续选择同侧关闭 |
| p023 | 3→4 | exact→token | non-AC | non-AC | non-AC | 新增全水地图；既有环形陆地覆盖湖泊不计岸线 |
| p024 | 3→4 | exact→token | non-AC | non-AC | non-AC | 新增单面值系统；既有 canonical/non-canonical 对照 |
| p025 | 3→4 | exact→token | non-AC | CE | CE | 新增多 SCC 分支 DAG，验证缩点后最大路径 |
| p026 | 3→4 | exact→token | non-AC | CE | non-AC | 新增非 1 根、星形加链及同点查询 |
| p027 | 3→4 | exact→token | CE | CE | CE | 新增两条并列最短路，要求全部删除后求次短路 |
| p028 | 3→4 | exact→token | non-AC | non-AC | non-AC | 新增全异值及单点/全区间查询 |
| p029 | 3→4 | exact→token | non-AC | non-AC | non-AC | 新增跨行同重哑铃和嵌套配对阈值 |
| p030 | 3→4 | exact→token | non-AC | non-AC | non-AC | 新增单终端及多终端绕墙 MST |

## Required conclusions

- **旧 AC 被推翻**：有。Base/p002 因 `%4` 错误被 regression case 推翻；Base/p021 因 `int` 求偶数中位数溢出被边界测试推翻。
- **p002**：Base 从 AC 变 WA；LoRA-512 仍 CE；LoRA-1536 仍 WA。正确 reference 通过 8/8。
- **p013**：原题为普通 token-based 排序输出，行尾空格和换行布局不应改变判定。三模型均从 WA 变 AC，并通过新增重复/极值测试。
- **额外发现 p010**：逆序数字同样属于 token 输出；三模型的旧 WA 都是空白格式误判，v1.1 均 AC。
- **总体结论**：核心结论不变。LoRA-512/1536 的 v1.1 Offline AC 分别高于 Base 3/2 题，但增量仍集中于 Easy、实现遵循和空白判定修正；Hard 仍为 0/6，未证明 Style LoRA 普遍提升算法能力。LoRA-1536 的 Compile 优势仍成立。

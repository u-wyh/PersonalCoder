# PersonalCoder

PersonalCoder 是一个基于小型代码大模型和 QLoRA 的个性化算法代码生成模型，目标是学习个人 C++ 竞赛代码风格和算法实现习惯。

## 环境检查

安装核心依赖后，运行：

```bash
python scripts/check_env.py
```

脚本会输出 Python、PyTorch、CUDA、GPU 以及训练相关核心包的版本和状态。若 CUDA 不可用，脚本会清晰报错并以非零状态退出，但不会修改任何系统配置。

## Model smoke test

模型固定从 `/data/PersonalCoder/model` 加载。确认本地模型文件完整后，运行 4-bit NF4 推理测试：

```bash
source .venv/bin/activate
python scripts/test_model.py
```

测试设置了 `local_files_only=True`，不依赖 Hugging Face 在线下载。

## RTX 4060：1536-token Style QLoRA

该实验使用独立数据目录和 checkpoint 目录：

```bash
source .venv/bin/activate
python scripts/build_style_chunks_1536.py
python scripts/train_style_lora_1536.py --config configs/style_lora_1536_v1.json
```

训练脚本在目标目录非空时会直接拒绝运行，避免覆盖已有 adapter。

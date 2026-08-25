# PersonalCoder

PersonalCoder 是一个基于小型代码大模型和 QLoRA 的个性化算法代码生成模型，目标是学习个人 C++ 竞赛代码风格和算法实现习惯。

## 环境检查

安装核心依赖后，运行：

```bash
python scripts/check_env.py
```

脚本会输出 Python、PyTorch、CUDA、GPU 以及训练相关核心包的版本和状态。若 CUDA 不可用，脚本会清晰报错并以非零状态退出，但不会修改任何系统配置。

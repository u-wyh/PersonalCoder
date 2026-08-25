# 项目协作规则

## Python 环境

- 执行任何 Python、PyTorch、训练或评测命令前，必须先运行：

  ```bash
  source .venv/bin/activate
  ```

## 本机训练环境

- 操作系统：WSL2 Ubuntu 24.04
- GPU：RTX 4060 Laptop GPU 8GB
- 项目路径：`/home/wyh15/PersonalCoder`
- Base Model：`/data/PersonalCoder/model`
- Checkpoint 根目录：`/data/PersonalCoder/checkpoints`

## 模型访问

- 不允许自动访问网络下载模型；模型必须使用本地路径。

## Git 工作流

- 完成代码任务后，必须依次运行：

  ```bash
  git status
  git add .
  git commit -m "<与任务对应的提交信息>"
  git push <当前工作分支>
  ```

- 禁止使用 `git push --force`。

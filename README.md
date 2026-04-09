# Tic-Tac-Toe
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-2.x-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![tqdm](https://img.shields.io/badge/tqdm-progress-76B900?logo=python&logoColor=white)](https://tqdm.github.io/)
[![uv](https://img.shields.io/badge/uv-package%20manager-DE5FE9)](https://docs.astral.sh/uv/)

[中文（默认）](./README.md) | [English](./README.en.md)

这是一个基于 Python 的命令行井字棋（无 GUI）项目，内置了强化学习（Reinforcement Learning）AI。

AI 采用蒙特卡洛方法进行自我对弈学习，并随着模拟对局增加不断优化落子策略。

# 架构
- `src/engine.py`：游戏规则与状态管理（棋盘、落子、胜负判断）
- `src/ai.py`：蒙特卡洛策略与模型持久化（训练、决策、保存/加载）
- `src/cli.py`：终端交互层（输入输出与对局循环）
- `src/game.py`：轻量入口，负责组合 CLI 与 AI
- `src/model.pkl`：AI 使用的预训练价值表

# 安装
## 1. 克隆仓库
``` sh
git clone https://github.com/ezhpry/tictactoe.git
```

## 2. 初始化环境
请先安装 `uv`。

执行 `uv sync` 会自动安装依赖、创建虚拟环境并匹配 Python 版本。
``` sh
cd tictactoe
uv sync
```

## 3. 运行游戏
``` sh
python3 ./src/game.py
```

# 可选：重新训练 AI
编辑 `src/game.py`，取消以下注释：
```py
# game.model.train()
```
然后再次运行程序，即可生成新的 `src/model.pkl`。

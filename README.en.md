# Tic-Tac-Toe
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-2.x-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![tqdm](https://img.shields.io/badge/tqdm-progress-76B900?logo=python&logoColor=white)](https://tqdm.github.io/)
[![uv](https://img.shields.io/badge/uv-package%20manager-DE5FE9)](https://docs.astral.sh/uv/)

[中文（默认）](./README.md) | [English](./README.en.md)

A simple Tic-Tac-Toe game in the terminal (no GUI) written in Python with a reinforcement learning-based AI.

The AI uses Monte Carlo methods to learn from simulated games and improve its moves over time.

# Architecture
- `src/engine.py`: game rules and state management (board, moves, winner detection)
- `src/ai.py`: Monte Carlo policy and model persistence (train, choose move, save/load)
- `src/cli.py`: terminal interaction layer (input/output loop)
- `src/game.py`: lightweight entrypoint that wires CLI + AI together
- `src/model.pkl`: pre-trained value table used by the AI

# Installation
## 1. Clone the repository
``` sh
git clone https://github.com/ezhpry/tictactoe.git
```

## 2. Setup environment
Make sure you have installed uv.

`uv sync` will automatically install dependencies, set up a virtual environment, and ensure the correct Python version.
``` sh
cd tictactoe
uv sync
```

## 3. Run the game
``` sh
python3 ./src/game.py
```

# Optional: Re-train the AI
Edit `src/game.py` and uncomment:
```py
# game.model.train()
```
Then run the game again to generate a new `src/model.pkl`.

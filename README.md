# Tic-Tac-Toe
A simple Tic-Tac-Toe game in the terminal (no GUI) written in Python with a reinforcement learning-based AI.

The AI uses Monte Carlo methods to learn from simulated games and improve its moves over time.

# Architecture
- `src/engine.py`: game rules and state management (board, moves, winner detection)
- `src/ai.py`: Monte Carlo policy and model persistence (train, choose move, save/load)
- `src/cli.py`: terminal interaction layer (input/output loop)
- `src/game.py`: lightweight entrypoint that wires CLI + AI together

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










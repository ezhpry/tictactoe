from __future__ import annotations

"""蒙特卡洛策略与训练模块。

这个模块只关心“如何评估局面并选择落子”，
不处理终端交互；规则判断交给 engine。
"""

import os
import pickle
import random

import numpy as np
from tqdm import tqdm

from engine import Move, StateKey, TicTacToe, board_key, legal_moves, status_to_reward

dirname = os.path.dirname(os.path.abspath(__file__))


class MonteCarloModel:
    """基于状态访问计数的简易蒙特卡洛模型。"""

    def __init__(self, epsilon: float = 0.7, count: int = 10000, model_path: str | None = None):
        self.epsilon = epsilon
        self.count = count
        self.model_path = model_path or os.path.join(dirname, "model.pkl")
        # table[state] = [black_win_count, draw_count, white_win_count]
        self.table: dict[StateKey, np.ndarray] = self.load()

    def save(self) -> None:
        """持久化模型到本地文件。"""

        with open(self.model_path, "wb") as file:
            file.write(pickle.dumps(self.table))

    def load(self) -> dict[StateKey, np.ndarray]:
        """从本地加载模型；若不存在则返回空表。"""

        if not os.path.exists(self.model_path):
            return {}
        with open(self.model_path, "rb") as file:
            return pickle.loads(file.read())

    def choose_move(self, board: np.ndarray, turn: int) -> Move:
        """对外决策接口：默认走利用策略。"""

        return self._exploitation(board, turn)

    def act(self, board: np.ndarray, turn: int) -> Move:
        # Backward-compatible name with the old interface.
        return self.choose_move(board, turn)

    def _exploration(self, board: np.ndarray) -> Move:
        """探索：在合法位置中随机选一步。"""

        moves = legal_moves(board)
        if not moves:
            raise ValueError("No legal move on the board.")
        return random.choice(moves)

    def _exploitation(self, board: np.ndarray, turn: int) -> Move:
        """利用：遍历所有候选落子，选择估值最高的一步。"""

        moves = legal_moves(board)
        if not moves:
            raise ValueError("No legal move on the board.")

        scored_moves: list[tuple[Move, float]] = []
        for move in moves:
            next_board = board.copy()
            next_board[move] = turn
            key = board_key(next_board)
            if key not in self.table:
                continue

            black, _, white = self.table[key]
            total = int(np.sum(self.table[key]))
            if total <= 0:
                continue

            # black-white 越大对黑方越好；乘以 turn 后统一成“当前执子方收益”
            score = (black - white) / total * turn
            scored_moves.append((move, score))

        if not scored_moves:
            return self._exploration(board)
        return max(scored_moves, key=lambda item: item[1])[0]

    def _run_episode(self) -> None:
        """自我对弈一局，并把终局奖励回灌到访问过的状态。"""

        game = TicTacToe()
        chain: list[StateKey] = []

        while not game.is_over:
            if random.random() < self.epsilon:
                move = self._exploration(game.board)
            else:
                move = self._exploitation(game.board, game.turn)

            game.play(move)
            chain.append(board_key(game.board))

        reward = status_to_reward(game.status)
        for key in chain:
            self.table.setdefault(key, np.array([0, 0, 0], dtype=np.int64))
            self.table[key] += reward

    def train(self, episodes: int | None = None, show_progress: bool = True) -> None:
        """训练多局并保存模型。"""

        total_episodes = episodes or self.count
        iterator = tqdm(range(total_episodes)) if show_progress else range(total_episodes)

        for _ in iterator:
            self._run_episode()

        self.save()

from __future__ import annotations

"""井字棋规则引擎。

这个模块只关心“规则与状态”，不依赖 AI 或终端输入输出：
- 棋盘与回合定义
- 合法落子计算
- 终局判断（黑胜/白胜/平局/进行中）
"""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

EMPTY = 0
BLACK = 1
WHITE = -1

Move = tuple[int, int]
StateKey = tuple[int, ...]


class GameStatus(Enum):
    """对局状态。"""

    ONGOING = "ongoing"
    BLACK_WIN = "black_win"
    DRAW = "draw"
    WHITE_WIN = "white_win"


def new_board() -> np.ndarray:
    """创建空棋盘。"""

    return np.zeros((3, 3), dtype=np.int8)


def board_key(board: np.ndarray) -> StateKey:
    """将棋盘压平成可哈希 key，用于价值表索引。"""

    return tuple(int(v) for v in board.reshape(9))


def legal_moves(board: np.ndarray) -> list[Move]:
    """返回当前所有可落子位置。"""

    return [tuple(pos) for pos in np.argwhere(board == EMPTY)]


def evaluate_board(board: np.ndarray) -> GameStatus:
    """判断当前棋盘状态是否终局。"""

    # 井字棋胜负只取决于 8 条线：3 行 + 3 列 + 2 条对角线
    col_sum = list(np.sum(board, axis=0))
    row_sum = list(np.sum(board, axis=1))
    diagonal_sum = [np.trace(board)]
    anti_diagonal_sum = [np.trace(np.flip(board, axis=1))]
    lines = col_sum + row_sum + diagonal_sum + anti_diagonal_sum

    if 3 in lines:
        return GameStatus.BLACK_WIN
    if -3 in lines:
        return GameStatus.WHITE_WIN
    if not np.any(board == EMPTY):
        return GameStatus.DRAW
    return GameStatus.ONGOING


def status_to_reward(status: GameStatus) -> np.ndarray:
    """将终局状态映射为训练奖励向量 [black, draw, white]。"""

    if status is GameStatus.BLACK_WIN:
        return np.array([1, 0, 0], dtype=np.int64)
    if status is GameStatus.DRAW:
        return np.array([0, 1, 0], dtype=np.int64)
    if status is GameStatus.WHITE_WIN:
        return np.array([0, 0, 1], dtype=np.int64)
    raise ValueError("Cannot build reward for non-terminal status.")


@dataclass
class TicTacToe:
    """可变对局对象：保存棋盘和当前执子方。"""

    board: np.ndarray = field(default_factory=new_board)
    turn: int = BLACK

    def copy(self) -> "TicTacToe":
        """深拷贝当前对局，避免外部修改互相影响。"""

        return TicTacToe(board=self.board.copy(), turn=self.turn)

    def reset(self) -> None:
        """重置到初始局面。"""

        self.board = new_board()
        self.turn = BLACK

    def available_moves(self) -> list[Move]:
        """查询当前可落子位置。"""

        return legal_moves(self.board)

    def play(self, move: Move) -> None:
        """执行一步落子，并自动切换回合。"""

        row, col = move
        if row not in (0, 1, 2) or col not in (0, 1, 2):
            raise ValueError(f"Move out of board: {move}")
        if self.board[move] != EMPTY:
            raise ValueError(f"Cell already occupied: {move}")
        self.board[move] = self.turn
        self.turn *= -1

    @property
    def status(self) -> GameStatus:
        """实时计算当前局面状态。"""

        return evaluate_board(self.board)

    @property
    def is_over(self) -> bool:
        """是否已结束。"""

        return self.status is not GameStatus.ONGOING

from __future__ import annotations

"""终端交互层。

职责是把用户输入输出和 engine/ai 连接起来，
不直接实现规则细节或训练逻辑。
"""

from ai import MonteCarloModel
from engine import EMPTY, BLACK, GameStatus, Move, TicTacToe


class TerminalGame:
    """人机对战的命令行运行器。"""

    def __init__(self, model: MonteCarloModel | None = None):
        self.game = TicTacToe()
        self.model = model or MonteCarloModel(epsilon=0.7, count=100000)
        self.human_side = BLACK

    def _input_move(self) -> Move:
        """读取并校验用户输入（1~9 对应九宫格）。"""

        while True:
            index = input("Please input action ( 1 ~ 9 ) :").strip()
            if not index.isdigit() or len(index) != 1 or index == "0":
                print(f"invalid action {index}")
                continue

            offset = int(index) - 1
            move = (offset // 3, offset % 3)
            if self.game.board[move] != EMPTY:
                print(f"{index} is not empty")
                continue
            return move

    def _print_result(self) -> None:
        """输出终局棋盘和胜负结果。"""

        print(self.game.board)
        status = self.game.status
        if status is GameStatus.BLACK_WIN:
            print("black win!")
        elif status is GameStatus.DRAW:
            print("draw!")
        elif status is GameStatus.WHITE_WIN:
            print("white win!")

    def run(self) -> None:
        """启动对局主循环。"""

        # 打印模型已学习到的状态数量，便于观察模型规模。
        print(len(self.model.table))

        while not self.game.is_over:
            print(self.game.board)

            # 人类回合读取输入；AI 回合调用策略模块决策。
            if self.game.turn == self.human_side:
                move = self._input_move()
            else:
                move = self.model.choose_move(self.game.board, self.game.turn)

            self.game.play(move)

        self._print_result()

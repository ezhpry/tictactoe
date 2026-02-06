import numpy as np
import re

EMPTY = 0
BLACK = 1
WHITE = -1

def end_game(state: np.ndarray) -> np.ndarray | None:
    r0 = list(np.sum(state, axis=0))  # 按列求和
    r1 = list(np.sum(state, axis=1))  # 按行求和
    r2 = [np.trace(state)]  # 主对角线求和
    r3 = [np.trace(np.flip(state, axis=1))]  # 副对角线求和
    r = r0+r1+r2+r3

    # 三个数字 分别表示 黑 平局 白
    if 3 in r:
        return np.array([1, 0, 0])

    if -3 in r:
        return np.array([0, 0, 1])
    if len(np.argwhere(state == 0)) == 0:
        return np.array([0, 1, 0])
    return None


class Game(object):
    def __init__(self):
        self.state = np.zeros((3, 3), dtype=np.int8)
        self.turn = BLACK

    def input_function(self):
        while True:
            index = input("Please input action ( 1 ~ 9 ) :")
            if not re.match('^[1-9]$', index):
                print(f"invalid action {index}")
                continue
            index = int(index)-1
            where = (index//3, index % 3)
            if self.state[where] != EMPTY:
                print(f"{index+1} is not empty")
                continue
            return where

    def action(self, where: tuple[int, int]):
        assert (self.state[where] == 0)
        self.state[where] = self.turn
        self.turn *= -1

    def check(self):
        r = end_game(self.state)
        if r is None:
            return False
        print(self.state)
        black, draw, white = r
        if black:
            print("black win!")
        if draw:
            print("draw win!")
        if white:
            print("white win")
        return True

    def start(self):
        while True:
            print(self.state)
            where = self.input_function()
            self.action(where)
            if self.check():
                break


def main():
    game = Game()
    game.start()


if __name__ == '__main__':
    main()

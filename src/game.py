import numpy as np
import random
from tqdm import tqdm
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

    # BLACK DRAW WHITE
    if 3 in r:
        return np.array([1, 0, 0])
    if -3 in r:
        return np.array([0, 0, 1])
    if len(np.argwhere(state == 0)) == 0:
        return np.array([0, 1, 0])
    return None

def hash(state: np.ndarray):
    return tuple(state.reshape(9))

class Model(object):
    def __init__(self, epsilon=0.7, count=10000):

        self.table: dict[tuple, np.ndarray] = {}
        self.epsilon = epsilon
        self.count = count

    def act(self, state: np.ndarray, turn: int):
        # wheres = np.argwhere(state == EMPTY)
        # where = random.choice(wheres)
        # return tuple(where)
        return self.exploitation(state, turn)

    def exploration(self, state: np.ndarray):
        wheres = np.argwhere(state == EMPTY)
        where = random.choice(wheres)
        return tuple(where)

    def exploitation(self, state: np.ndarray, turn: int):
        wheres = np.argwhere(state == EMPTY)
        assert (len(wheres) > 0)

        results = []
        for where in wheres:
            where = tuple(where)
            s = state.copy()
            s[where] = turn

            key = hash(s)
            if key not in self.table:
                continue

            black, draw, white = self.table[key]
            p = (black-white)/sum(self.table[key])*turn
            results.append((where, p))
        if not results:
            return self.exploration(state)

        result = sorted(results, key=lambda e: e[1])[-1]
        return result[0]

    def step(self, state: np.ndarray, turn: int, chain: list):
        if random.random() < self.epsilon:
            where = self.exploration(state)
        else:
            where = self.exploitation(state, turn)

        state[where] = turn
        chain.append(hash(state))
        end = end_game(state)
        if end is None:
            return self.step(state, turn*-1, chain)
        for key in chain:
            self.table.setdefault(key, np.array([0, 0, 0]))
            self.table[key] += end
        return

    def train(self):
        state = np.zeros((3, 3), dtype=np.int8)
        turn = BLACK
        for _ in tqdm(range(self.count)):
            self.step(state.copy(), turn, [])


class Game(object):
    def __init__(self):
        self.state = np.zeros((3, 3), dtype=np.int8)
        self.turn = BLACK
        self.model = Model(epsilon=0.7, count=10000)
        self.model.train()
        print(len(self.model.table))

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
            print("draw!")
        if white:
            print("white win!")
        return True

    def start(self):
        while True:
            print(self.state)
            where = self.input_function()
            self.action(where)
            if self.check():
                break
            where = self.model.act(self.state, self.turn)
            self.action(where)
            if self.check():
                break


def main():
    game = Game()
    game.start()


if __name__ == '__main__':
    main()

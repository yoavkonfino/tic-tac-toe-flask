import random

# a state is a 9-character string, "." = empty cell, X always moves first
EMPTY = "."

LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),   # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),   # columns
    (0, 4, 8), (2, 4, 6),              # diagonals
]

# rewards for the agent
WIN_REWARD = 1.0
LOSS_REWARD = -1.0
DRAW_REWARD = 0.0
STEP_REWARD = 0.0   # a move that does not end the game


class Board:
    def __init__(self):
        self.cells = [EMPTY] * 9

    def available_moves(self):
        return [i for i, c in enumerate(self.cells) if c == EMPTY]

    def place(self, index, mark):
        if self.cells[index] != EMPTY:
            raise ValueError("cell %d is already taken" % index)
        self.cells[index] = mark

    def winner(self):
        # "X", "O" or None
        for a, b, c in LINES:
            if self.cells[a] != EMPTY and self.cells[a] == self.cells[b] == self.cells[c]:
                return self.cells[a]
        return None

    def is_full(self):
        return EMPTY not in self.cells

    def state(self):
        return "".join(self.cells)


class RandomPlayer:
    def __init__(self, mark="O"):
        self.mark = mark

    def choose_move(self, board):
        return random.choice(board.available_moves())


class Environment:
    # the opponent's reply happens inside step(), so the agent always gets
    # back a state where it is its turn again (or the game is over)
    def __init__(self, opponent=None):
        self.opponent = opponent if opponent is not None else RandomPlayer()
        self.board = Board()
        self.agent_mark = "X"
        self.opponent_mark = "O"
        self._result = None

    def reset(self, agent_starts=True):
        # the first player is X; if the opponent starts, it moves here
        self.board = Board()
        self._result = None
        self.agent_mark = "X" if agent_starts else "O"
        self.opponent_mark = "O" if agent_starts else "X"
        self.opponent.mark = self.opponent_mark
        if not agent_starts:
            self._opponent_move()
        return self.board.state()

    def step(self, action):
        # returns (next_state, reward, done)
        if self._result is not None:
            raise RuntimeError("the game is over; call reset()")
        self.board.place(action, self.agent_mark)
        if self.board.winner() == self.agent_mark:
            return self._finish("win", WIN_REWARD)
        if self.board.is_full():
            return self._finish("draw", DRAW_REWARD)

        self._opponent_move()
        if self.board.winner() == self.opponent_mark:
            return self._finish("loss", LOSS_REWARD)
        if self.board.is_full():
            return self._finish("draw", DRAW_REWARD)
        return self.board.state(), STEP_REWARD, False

    def get_result(self):
        # "win", "loss" or "draw"; None while the game is on
        return self._result

    def _opponent_move(self):
        move = self.opponent.choose_move(self.board)
        self.board.place(move, self.opponent_mark)

    def _finish(self, result, reward):
        self._result = result
        return self.board.state(), reward, True
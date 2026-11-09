import json
import random

# a state is a 9-character string, "." = empty cell
EMPTY = "."


class Agent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=1.0,
                 epsilon_min=0.05, epsilon_decay=0.999):
        # state -> list of 9 values (None where the cell is occupied)
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def get_q_values(self, state):
        # add a new row first if the state is unseen
        if state not in self.q_table:
            self.q_table[state] = [0.0 if c == EMPTY else None for c in state]
        return self.q_table[state]

    def legal_actions(self, state):
        return [i for i, c in enumerate(state) if c == EMPTY]

    def best_action(self, state):
        legal = self.legal_actions(state)
        if not legal:
            raise ValueError("no legal actions in state " + repr(state))
        q = self.get_q_values(state)
        best = max(q[a] for a in legal)
        # random among ties
        return random.choice([a for a in legal if q[a] == best])

    def choose_action(self, state):
        legal = self.legal_actions(state)
        if not legal:
            raise ValueError("no legal actions in state " + repr(state))
        # explore with probability epsilon, otherwise exploit
        if random.random() < self.epsilon:
            return random.choice(legal)
        return self.best_action(state)

    def update(self, s, a, r, s_next, done):
        q = self.get_q_values(s)
        if done:
            target = r
        else:
            q_next = self.get_q_values(s_next)
            legal_next = self.legal_actions(s_next)
            # best Q among the legal moves of the next state
            future = max(q_next[i] for i in legal_next) if legal_next else 0.0
            target = r + self.gamma * future
        q[a] += self.alpha * (target - q[a])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        # None is stored as null
        with open(path, "w") as f:
            json.dump(self.q_table, f)

    def load(self, path):
        with open(path) as f:
            data = json.load(f)
        # accept {"meta": ..., "q_table": ...} as well as a bare Q-table
        self.q_table = data["q_table"] if "q_table" in data else data
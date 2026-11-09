import random

from Agent import Agent
from Environment import Environment


class Game:
    def __init__(self, agent, env, num_episodes=50000, report_every=5000,
                 save_path="q_table.json"):
        self.agent = agent
        self.env = env
        self.num_episodes = num_episodes
        self.report_every = report_every
        self.save_path = save_path

        # counts for the current block
        self.wins = self.losses = self.draws = 0
        # counts for all episodes
        self.total_wins = self.total_losses = self.total_draws = 0

    def play_episode(self):
        state = self.env.reset(agent_starts=random.choice([True, False]))
        done = False
        while not done:
            action = self.agent.choose_action(state)
            next_state, reward, done = self.env.step(action)
            self.agent.update(state, action, reward, next_state, done)
            state = next_state
        return self.env.get_result()

    def record(self, result):
        if result == "win":
            self.wins += 1
            self.total_wins += 1
        elif result == "loss":
            self.losses += 1
            self.total_losses += 1
        elif result == "draw":
            self.draws += 1
            self.total_draws += 1
        else:
            raise ValueError("unknown result: %r" % (result,))

    def report(self, episode):
        games = self.wins + self.losses + self.draws
        win_pct = 100.0 * self.wins / games if games else 0.0
        print("Episode %6d | wins %5d  losses %5d  draws %5d | win %5.1f%% | epsilon %.3f"
              % (episode, self.wins, self.losses, self.draws, win_pct, self.agent.epsilon))
        # start a new block
        self.wins = self.losses = self.draws = 0

    def train(self):
        for episode in range(1, self.num_episodes + 1):
            self.record(self.play_episode())
            self.agent.decay_epsilon()
            if episode % self.report_every == 0:
                self.report(episode)
        # a last, shorter block if the episodes don't divide evenly
        if self.wins + self.losses + self.draws > 0:
            self.report(self.num_episodes)

        total = self.total_wins + self.total_losses + self.total_draws
        if total:
            print("Total: %d wins, %d losses, %d draws (%.1f%% wins)"
                  % (self.total_wins, self.total_losses, self.total_draws,
                     100.0 * self.total_wins / total))
        self.agent.save(self.save_path)
        print("Q-table saved to", self.save_path)


if __name__ == "__main__":
    agent = Agent(alpha=0.3, gamma=0.9, epsilon=1.0,
                  epsilon_min=0.05, epsilon_decay=0.9998)
    Game(agent, Environment(), num_episodes=50000, report_every=5000).train()
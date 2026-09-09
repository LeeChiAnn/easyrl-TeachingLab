import numpy as np
from .base import BaseAgent


class SARSAAgent(BaseAgent):
    """表格 SARSA（on-policy）。与 Q-Learning 的区别仅在于更新时用 next_action。"""

    def __init__(self, env, learning_rate=0.5, gamma=0.99,
                 epsilon_start=0.5, epsilon_min=0.0, epsilon_schedule="linear"):
        super().__init__(env)
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.schedule = epsilon_schedule
        self.epsilon = epsilon_start
        self.q = np.random.uniform(0, 1, (env.num_states, env.num_actions))

    def _eps(self, episode):
        if self.schedule == "linear":
            return max(self.epsilon_min, self.epsilon_start / (episode + 1))
        decay = getattr(self, "epsilon_decay", 0.995)
        return max(self.epsilon_min, self.epsilon_start * (decay ** episode))

    def act(self, obs, episode):
        eps = self._eps(episode)
        if np.random.rand() < eps:
            return np.random.randint(self.env.num_actions)
        s = self.env.digitize_state(obs)
        return int(np.argmax(self.q[s]))

    def learn(self, obs, action, reward, next_obs, next_action, done):
        s = self.env.digitize_state(obs)
        s2 = self.env.digitize_state(next_obs)
        q = self.q[s, action]
        if done:
            target = reward
        else:
            target = reward + self.gamma * self.q[s2, int(next_action)]
        td = target - q
        self.q[s, action] += self.lr * td
        return abs(td)

    def best_action(self, obs):
        s = self.env.digitize_state(obs)
        return int(np.argmax(self.q[s]))

    def end_episode(self, episode):
        self.epsilon = self._eps(episode)
        return None

    def save(self, path):
        np.save(path, self.q)

    def load(self, path):
        self.q = np.load(path)

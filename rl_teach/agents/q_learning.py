import numpy as np
from .base import BaseAgent


class QLearningAgent(BaseAgent):
    """表格 Q-Learning（off-policy）。复用了已测试通过的更新逻辑与超参默认值。"""

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
        target = reward if done else reward + self.gamma * np.max(self.q[s2])
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

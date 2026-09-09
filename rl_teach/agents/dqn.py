import numpy as np
from .base import BaseAgent


class DQNAgent(BaseAgent):
    """最简洁的可演示 DQN：单隐层网络 + 经验回放 + 目标网络。"""

    def __init__(self, env, gamma=0.99, lr=0.01, epsilon_start=1.0,
                 epsilon_min=0.05, epsilon_decay=0.995, hidden=16,
                 batch=32, memory=2000, target_update=50):
        super().__init__(env)
        self.n_in = env.obs_dim
        self.n_out = env.num_actions
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.decay = epsilon_decay
        self.hidden = hidden
        self.batch = batch
        self.target_update = target_update
        # 在线网络
        self.W1 = np.random.randn(self.n_in, hidden) * 0.1
        self.b1 = np.zeros(hidden)
        self.W2 = np.random.randn(hidden, self.n_out) * 0.1
        self.b2 = np.zeros(self.n_out)
        # 目标网络（周期同步）
        self._sync_target()
        self.memory = []
        self.mem_max = memory
        self.step_count = 0

    def _forward(self, x, W1, b1, W2, b2):
        h = np.tanh(x.dot(W1) + b1)
        q = h.dot(W2) + b2
        return h, q

    def _sync_target(self):
        self.tW1, self.tb1, self.tW2, self.tb2 = (
            self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy())

    def act(self, obs, episode):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_out)
        x = np.asarray(obs, dtype=float)
        _, q = self._forward(x, self.W1, self.b1, self.W2, self.b2)
        return int(np.argmax(q))

    def best_action(self, obs):
        x = np.asarray(obs, dtype=float)
        _, q = self._forward(x, self.W1, self.b1, self.W2, self.b2)
        return int(np.argmax(q))

    def learn(self, obs, action, reward, next_obs, next_action, done):
        self.memory.append((np.asarray(obs, float),
                            int(action), float(reward),
                            np.asarray(next_obs, float), bool(done)))
        if len(self.memory) > self.mem_max:
            self.memory.pop(0)

        loss = 0.0
        if len(self.memory) >= self.batch:
            idx = np.random.choice(len(self.memory), self.batch, replace=False)
            X = np.stack([self.memory[i][0] for i in idx])
            A = np.array([self.memory[i][1] for i in idx])
            R = np.array([self.memory[i][2] for i in idx])
            N = np.stack([self.memory[i][3] for i in idx])
            D = np.array([self.memory[i][4] for i in idx])

            h, q_on = self._forward(X, self.W1, self.b1, self.W2, self.b2)
            _, q_t = self._forward(N, self.tW1, self.tb1, self.tW2, self.tb2)

            target = q_on.copy()
            for i in range(self.batch):
                if D[i]:
                    target[i, A[i]] = R[i]
                else:
                    target[i, A[i]] = R[i] + self.gamma * np.max(q_t[i])

            dq = (q_on - target) / self.batch
            dh = dq.dot(self.W2.T) * (1 - h ** 2)

            self.W2 -= self.lr * h.T.dot(dq)
            self.b2 -= self.lr * dq.sum(0)
            self.W1 -= self.lr * X.T.dot(dh)
            self.b1 -= self.lr * dh.sum(0)

            loss = float(np.mean((q_on - target) ** 2))

        self.step_count += 1
        if self.step_count % self.target_update == 0:
            self._sync_target()
        return loss

    def end_episode(self, episode):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.decay)
        return None

    def save(self, path):
        np.save(path, {"W1": self.W1, "b1": self.b1,
                       "W2": self.W2, "b2": self.b2})

    def load(self, path):
        d = np.load(path, allow_pickle=True).item()
        self.W1, self.b1, self.W2, self.b2 = d["W1"], d["b1"], d["W2"], d["b2"]
        self._sync_target()

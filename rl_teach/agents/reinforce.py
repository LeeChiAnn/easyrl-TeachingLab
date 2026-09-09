import numpy as np
from .base import BaseAgent


def _softmax(x):
    e = np.exp(x - np.max(x))
    return e / np.sum(e)


class ReinforceAgent(BaseAgent):
    """策略梯度 REINFORCE：直接学习随机策略（线性+softmax），无价值基线。"""

    def __init__(self, env, gamma=0.99, lr=0.01, hidden=16):
        super().__init__(env)
        self.n_in = env.obs_dim
        self.n_out = env.num_actions
        self.gamma = gamma
        self.lr = lr
        self.W1 = np.random.randn(self.n_in, hidden) * 0.1
        self.b1 = np.zeros(hidden)
        self.W2 = np.random.randn(hidden, self.n_out) * 0.1
        self.b2 = np.zeros(self.n_out)
        self.traj = []

    def _logits(self, x):
        h = np.tanh(x.dot(self.W1) + self.b1)
        return h, h.dot(self.W2) + self.b2

    def act(self, obs, episode):
        x = np.asarray(obs, dtype=float)
        h, logits = self._logits(x)
        probs = _softmax(logits)
        return int(np.random.choice(self.n_out, p=probs))

    def best_action(self, obs):
        x = np.asarray(obs, dtype=float)
        _, logits = self._logits(x)
        return int(np.argmax(logits))

    def learn(self, obs, action, reward, next_obs, next_action, done):
        # 仅记录轨迹，真正的更新在 end_episode 整轮完成
        self.traj.append((np.asarray(obs, float), int(action), float(reward)))
        return 0.0

    def end_episode(self, episode):
        if not self.traj:
            return None
        # 反向计算折扣回报
        returns = []
        G = 0.0
        for _, _, r in reversed(self.traj):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = np.array(returns, dtype=float)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = 0.0
        for (obs, action, _), Gt in zip(self.traj, returns):
            h, logits = self._logits(obs)
            probs = _softmax(logits)
            dlog = probs.copy()
            dlog[action] -= 1.0            # d log pi / d logits
            dh = (dlog.dot(self.W2.T)) * (1 - h ** 2) * Gt
            # 策略上升：W -= -lr * grad
            self.W2 -= -self.lr * np.outer(h, dlog)
            self.b2 -= -self.lr * dlog
            self.W1 -= -self.lr * np.outer(obs, dh)
            self.b1 -= -self.lr * dh
            loss += -np.log(probs[action] + 1e-8) * Gt
        self.traj = []
        return float(loss / len(returns))

    def save(self, path):
        np.save(path, {"W1": self.W1, "b1": self.b1,
                       "W2": self.W2, "b2": self.b2})

    def load(self, path):
        d = np.load(path, allow_pickle=True).item()
        self.W1, self.b1, self.W2, self.b2 = d["W1"], d["b1"], d["W2"], d["b2"]

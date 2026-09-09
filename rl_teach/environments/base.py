"""环境统一封装基类：屏蔽 gym 新旧版本差异，提供离散化、奖励塑形、自绘渲染等接口。"""
import numpy as np
import gym


class EnvWrapper:
    # 子类必须定义
    gym_id = None

    def __init__(self, max_steps=200, num_digitized=6):
        self.max_steps = max_steps
        self.num_digitized = num_digitized
        self.env = gym.make(self.gym_id)
        self.num_actions = self.env.action_space.n
        self.obs_dim = self.env.observation_space.shape[0]
        self.low, self.high = self._bounds()
        self.num_states = num_digitized ** len(self.low)

    # ---------- 子类必须实现 ----------
    def _bounds(self):
        """返回 (low, high) 两个一维数组，用于状态离散化。"""
        raise NotImplementedError

    def compute_reward(self, obs, next_obs, done, step, max_steps):
        """返回用于 Q 更新的塑形奖励（可与 gym 原生奖励不同）。"""
        raise NotImplementedError

    def is_success(self, step, next_obs, done, max_steps):
        """判断本局是否“成功”（用于早停与收敛判定）。"""
        raise NotImplementedError

    def render_frame(self, obs):
        """根据观测值自绘一帧，返回 (H, W, 3) uint8 数组（无显示器可用）。"""
        raise NotImplementedError

    # ---------- 通用实现 ----------
    def _bins(self, clip_min, clip_max, num):
        return np.linspace(clip_min, clip_max, num + 1)[1:-1]

    def digitize_state(self, obs):
        obs = np.asarray(obs, dtype=float)
        digitized = [
            np.digitize(obs[i], self._bins(self.low[i], self.high[i], self.num_digitized))
            for i in range(len(obs))
        ]
        return int(sum(x * (self.num_digitized ** i) for i, x in enumerate(digitized)))

    def reset(self):
        obs = self.env.reset()
        if isinstance(obs, tuple):  # 兼容 gym>=0.26 返回 (obs, info)
            obs = obs[0]
        return np.asarray(obs, dtype=float)

    def step(self, action):
        out = self.env.step(action)
        if len(out) == 4:
            nxt, r, done, info = out
        else:  # gym>=0.26 返回 5 元组
            nxt, r, term, trunc, info = out
            done = term or trunc
        return np.asarray(nxt, dtype=float), r, done, info

    def close(self):
        self.env.close()

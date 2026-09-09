import numpy as np
from .base import EnvWrapper
from rl_teach.viz.gif import render_mountaincar


class MountainCarEnv(EnvWrapper):
    """小车上山：用动力让车冲上右侧山坡。"""

    gym_id = "MountainCar-v0"

    def _bounds(self):
        return (np.array([-1.2, -0.07]),
                np.array([0.6, 0.07]))

    def compute_reward(self, obs, next_obs, done, step, max_steps):
        # 原生奖励每步 -1；到达目标位置 (>=0.5) 给正奖励
        pos = float(np.asarray(next_obs, dtype=float)[0])
        return 1.0 if pos >= 0.5 else -1.0

    def is_success(self, step, next_obs, done, max_steps):
        pos = float(np.asarray(next_obs, dtype=float)[0])
        return bool(pos >= 0.5)

    def render_frame(self, obs):
        return render_mountaincar(obs)

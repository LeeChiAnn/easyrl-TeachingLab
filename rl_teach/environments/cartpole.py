import numpy as np
from .base import EnvWrapper
from rl_teach.viz.gif import render_cartpole


class CartPoleEnv(EnvWrapper):
    """平衡倒立摆：控制小车使杆保持直立。"""

    gym_id = "CartPole-v0"

    def _bounds(self):
        # [小车位置, 小车速度, 杆角度, 杆角速度] 的合理取值范围
        return (np.array([-2.4, -3.0, -0.5, -2.0]),
                np.array([2.4, 3.0, 0.5, 2.0]))

    def compute_reward(self, obs, next_obs, done, step, max_steps):
        if done:
            # 接近满步数才结束 => 视为成功，给正奖励；否则失败给负奖励
            return 1.0 if step >= max_steps - 20 else -1.0
        return 0.0

    def is_success(self, step, next_obs, done, max_steps):
        return bool(done and step >= max_steps - 20)

    def render_frame(self, obs):
        return render_cartpole(obs)

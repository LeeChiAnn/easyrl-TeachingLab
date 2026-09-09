"""智能体统一接口。所有算法实现 act/learn/save/load/best_action。"""
import numpy as np


class BaseAgent:
    def __init__(self, env, **kwargs):
        self.env = env

    def act(self, obs, episode):
        """给定观测与当前轮数，返回一个动作（含探索）。"""
        raise NotImplementedError

    def learn(self, obs, action, reward, next_obs, next_action, done):
        """执行一次更新，返回本次损失（用于绘制 loss 曲线）。"""
        raise NotImplementedError

    def best_action(self, obs):
        """贪心最优动作（用于演示 GIF，不做探索）。"""
        raise NotImplementedError

    def end_episode(self, episode):
        """每轮结束时的回调（如 epsilon 衰减、整轮更新）。返回本轮损失或 None。"""
        return None

    def save(self, path):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError

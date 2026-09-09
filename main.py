#!/usr/bin/env python
# coding: utf-8
"""EasyRL Teaching Lab 入口。

用法示例（在项目根目录执行）：
    python main.py                         # 使用 configs/config.yaml 的默认环境/算法
    python main.py --env mountaincar       # 切换到 MountainCar 环境
    python main.py --algo dqn              # 切换为 DQN 算法
    python main.py --env mountaincar --algo dqn
"""
import argparse
import os
import sys

import yaml

# 保证项目根目录在 sys.path，便于以脚本方式运行
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rl_teach.training.trainer import train


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="EasyRL Teaching Lab")
    parser.add_argument("--env", default=None, help="环境名: cartpole | mountaincar")
    parser.add_argument("--algo", default=None,
                        help="算法: q_learning | sarsa | dqn | reinforce")
    parser.add_argument("--config", default=os.path.join(ROOT, "configs", "config.yaml"),
                        help="配置文件路径")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.env:
        cfg["environment"]["name"] = args.env
    if args.algo:
        cfg["agent"]["type"] = args.algo

    # 输出根目录：相对配置中的 base_dir；非绝对则相对当前工作目录
    base_dir = cfg.get("output", {}).get("base_dir", "output")
    if not os.path.isabs(base_dir):
        base_dir = os.path.join(os.getcwd(), base_dir)

    train(cfg, base_dir)


if __name__ == "__main__":
    main()

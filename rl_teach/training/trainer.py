"""通用训练主循环：加载配置 -> 训练 -> 保存模型 -> 可视化 -> 运维报告。"""
import os
import time
import inspect
import numpy as np

from rl_teach.environments.registry import make_env
from rl_teach.agents import AGENT_REGISTRY
from rl_teach.viz.curves import plot_curves
from rl_teach.viz.gif import make_gif
from rl_teach.ops.monitor import Monitor
from rl_teach.ops.health import check_health
from rl_teach.ops.report import generate_report


def build_agent(name, env, agent_cfg):
    """根据配置构造算法实例，只传入该算法支持的参数。"""
    cls = AGENT_REGISTRY[name]
    sig = inspect.signature(cls.__init__)
    params = {k: v for k, v in agent_cfg.items()
              if k in sig.parameters and k not in ("type", "self", "env")}
    return cls(env, **params)


def _quick_train(env, agent, episodes, max_steps):
    """仅训练少量轮，用于生成“训练很少”的对照 GIF。"""
    for ep in range(episodes):
        obs = env.reset()
        done = False
        step = 0
        for step in range(max_steps):
            action = agent.act(obs, ep)
            next_obs, r, done, _ = env.step(action)
            shaped = env.compute_reward(obs, next_obs, done, step, max_steps)
            next_action = agent.act(next_obs, ep)
            agent.learn(obs, action, shaped, next_obs, next_action, done)
            obs = next_obs
            if done:
                break
        agent.end_episode(ep)


def train(cfg, output_root):
    env_cfg = cfg["environment"]
    agent_cfg = cfg["agent"]
    train_cfg = cfg.get("training", {})
    viz_cfg = cfg.get("visualization", {})
    ops_cfg = cfg.get("ops", {})

    env = make_env(env_cfg["name"],
                   max_steps=env_cfg.get("max_steps", 200),
                   num_digitized=env_cfg.get("num_digitized", 6))
    agent = build_agent(agent_cfg["type"], env, agent_cfg)

    # 输出目录：按 环境/类型 分类
    env_out = os.path.join(output_root, env_cfg["name"])
    models_dir = os.path.join(env_out, "models")
    charts_dir = os.path.join(env_out, "charts")
    gifs_dir = os.path.join(env_out, "gifs")
    ops_dir = os.path.join(env_out, "ops")
    for d in (models_dir, charts_dir, gifs_dir, ops_dir):
        os.makedirs(d, exist_ok=True)

    model_path = os.path.join(models_dir, "{}_{}.npy".format(env_cfg["name"], agent_cfg["type"]))

    if train_cfg.get("load_model", False) and os.path.exists(model_path):
        agent.load(model_path)
        print("已加载模型:", model_path)
    else:
        print("开始训练: env={} algo={}".format(env_cfg["name"], agent_cfg["type"]))

    monitor = Monitor()
    t_start = time.time()
    reward_hist = []
    loss_hist = []
    consecutive = 0
    max_ep = train_cfg.get("episodes", 2000)
    target = train_cfg.get("consecutive_success", 10)
    max_steps = env_cfg.get("max_steps", 200)

    for ep in range(max_ep):
        obs = env.reset()
        ep_reward = 0.0
        step_losses = []
        done = False
        step = 0
        for step in range(max_steps):
            action = agent.act(obs, ep)
            next_obs, env_reward, done, _ = env.step(action)
            shaped = env.compute_reward(obs, next_obs, done, step, max_steps)
            next_action = agent.act(next_obs, ep)   # SARSA 用其计算目标动作；其余算法忽略
            loss = agent.learn(obs, action, shaped, next_obs, next_action, done)
            step_losses.append(loss)
            obs = next_obs
            ep_reward += env_reward
            if done:
                break
        ep_loss = agent.end_episode(ep)
        ep_loss = ep_loss if ep_loss is not None else (float(np.mean(step_losses)) if step_losses else 0.0)

        success = env.is_success(step, next_obs, done, max_steps)
        consecutive = consecutive + 1 if success else 0
        monitor.record(ep, ep_reward, ep_loss, success)
        reward_hist.append(ep_reward)
        loss_hist.append(ep_loss)

        if (ep + 1) % 50 == 0:
            print("Episode {:4d} | reward {:7.1f} | loss {:.4f} | consec {}".format(
                ep + 1, ep_reward, ep_loss, consecutive))

        if consecutive >= target:
            print("已收敛：连续 {} 次成功，于第 {} 轮提前停止。".format(target, ep + 1))
            break

    total_time = time.time() - t_start

    if train_cfg.get("save_model", True):
        agent.save(model_path)
        print("模型已保存:", model_path)

    # 可视化：奖励 / 损失曲线
    reward_png = None
    loss_png = None
    if viz_cfg.get("plot_reward", True):
        reward_png = os.path.join(charts_dir, "reward.png")
    if viz_cfg.get("plot_loss", True):
        loss_png = os.path.join(charts_dir, "loss.png")
    if reward_png or loss_png:
        plot_curves(reward_hist, loss_hist, reward_png, loss_png,
                    agent_cfg["type"], env_cfg["name"], viz_cfg.get("dpi", 120))

    # 可视化：动态 GIF（注意必须在 env.close 之前）
    gif_cfg = viz_cfg.get("gifs", {})
    if gif_cfg.get("trained", True):
        make_gif(env, agent,
                 os.path.join(gifs_dir, "{}_{}_trained.gif".format(env_cfg["name"], agent_cfg["type"])),
                 max_steps, policy="greedy")
    if gif_cfg.get("random", True):
        make_gif(env, agent,
                 os.path.join(gifs_dir, "{}_random.gif".format(env_cfg["name"])),
                 max_steps, policy="random")
    if gif_cfg.get("weak", False):
        weak_ep = gif_cfg.get("weak_episodes", 20)
        weak_env = make_env(env_cfg["name"], max_steps=max_steps,
                            num_digitized=env_cfg.get("num_digitized", 6))
        weak_agent = build_agent(agent_cfg["type"], weak_env, agent_cfg)
        _quick_train(weak_env, weak_agent, weak_ep, max_steps)
        make_gif(weak_env, weak_agent,
                 os.path.join(gifs_dir, "{}_{}_weak.gif".format(env_cfg["name"], agent_cfg["type"])),
                 max_steps, policy="greedy")
        weak_env.close()

    env.close()  # 主环境在所有 GIF 生成后再关闭

    # 运维：监控 + 健康检查 + HTML 报告
    summary = monitor.summary(total_time, target, reward_hist, loss_hist)
    health = check_health(summary, cfg)
    report_path = None
    if ops_cfg.get("enabled", True):
        report_path = os.path.join(ops_dir, "ops_report.html")
        generate_report(report_path, summary, health, reward_png, loss_png, cfg)

    print("\n==== 训练完成 ====")
    print("奖励曲线 :", reward_png)
    print("损失曲线 :", loss_png)
    print("运维报告 :", report_path)
    return {
        "reward_png": reward_png,
        "loss_png": loss_png,
        "report": report_path,
        "summary": summary,
        "health": health,
    }

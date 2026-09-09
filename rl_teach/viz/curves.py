"""训练过程曲线：奖励曲线 & 损失曲线。"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_curves(reward_hist, loss_hist, reward_png, loss_png, algo, env_name, dpi=120):
    if reward_png:
        _save(reward_hist, "Episode Reward", "Episode",
              "Total Reward", reward_png, algo, env_name, dpi, smooth=True)
    if loss_png:
        _save(loss_hist, "Training Loss (TD-error / MSE)", "Episode",
              "Loss", loss_png, algo, env_name, dpi, smooth=False)


def _save(data, title, xlabel, ylabel, path, algo, env_name, dpi, smooth):
    data = np.asarray(data, dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(data, color="#1f77b4", alpha=0.5, label="raw")
    if smooth and len(data) > 10:
        w = max(1, len(data) // 20)
        kernel = np.ones(w) / w
        sm = np.convolve(data, kernel, mode="same")
        ax.plot(sm, color="#d62728", label="smoothed")
        ax.legend()
    ax.set_title("{} | {} | {}".format(title, env_name, algo))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print("曲线已保存:", path)

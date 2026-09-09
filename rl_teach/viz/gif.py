"""动态图与自绘渲染（兼容无显示器环境）。"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def render_cartpole(observation, figsize=(5, 3), dpi=80):
    """根据 [小车位置, 速度, 杆角度, 角速度] 画一帧，返回 (H, W, 3) uint8。"""
    cart_pos, _, pole_angle, _ = observation
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-0.2, 1.8)
    ax.set_aspect("equal")
    ax.axhline(0, color="gray", lw=1)
    cart_w, cart_h, L = 0.4, 0.2, 1.0
    ax.add_patch(Rectangle((cart_pos - cart_w / 2, 0), cart_w, cart_h, color="#1f77b4"))
    x2 = cart_pos + L * np.sin(pole_angle)
    y2 = cart_h + L * np.cos(pole_angle)
    ax.plot([cart_pos, x2], [cart_h, y2], color="#d62728", lw=5)
    ax.set_title("CartPole")
    ax.set_xticks([])
    ax.set_yticks([])
    img = _fig_to_rgb(fig)
    plt.close(fig)
    return img


def render_mountaincar(observation, figsize=(5, 3), dpi=80):
    """根据 [位置, 速度] 画一帧小车上山示意图。"""
    pos, vel = observation
    xs = np.linspace(-1.2, 0.6, 100)
    ys = np.sin(3 * xs) * 0.45 + 0.45
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111)
    ax.plot(xs, ys, color="gray")
    ax.set_xlim(-1.3, 0.7)
    ax.set_ylim(-0.1, 1.1)
    car_y = float(np.interp(pos, xs, ys))
    ax.plot(pos, car_y, "o", color="#1f77b4", ms=12)
    ax.text(0.45, 0.95, "GOAL", color="green")
    ax.set_title("MountainCar  pos={:.2f} vel={:.2f}".format(pos, vel))
    ax.set_xticks([])
    ax.set_yticks([])
    img = _fig_to_rgb(fig)
    plt.close(fig)
    return img


def _fig_to_rgb(fig):
    # 跨 matplotlib 版本的稳健做法：渲染到内存 PNG 再用 PIL 读取，
    # 规避旧版 tostring_rgb / 新版 tobytes_rgb 的 API 差异。
    import io
    from PIL import Image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=fig.dpi)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


def save_gif(frames, path, fps=30):
    """把帧列表保存为 GIF（优先 imageio，缺失则 PIL 兜底）。"""
    if not frames:
        print("警告: 没有帧可保存:", path)
        return
    try:
        import imageio
        imageio.mimsave(path, frames, duration=1000 // fps)
    except ImportError:
        from PIL import Image
        pil = [Image.fromarray(f) for f in frames]
        pil[0].save(path, save_all=True, append_images=pil[1:],
                    duration=1000 // fps, loop=0)
    print("动态图已保存:", path, "({} 帧)".format(len(frames)))


def run_episode_frames(env, agent, policy="greedy", max_steps=200):
    """跑一局并收集自绘帧。policy: greedy(训练后) / random(随机对照)。"""
    obs = env.reset()
    frames = []
    done = False
    step = 0
    for step in range(max_steps):
        frames.append(env.render_frame(obs))
        if policy == "random":
            action = np.random.randint(env.num_actions)
        else:  # greedy
            action = agent.best_action(obs)
        next_obs, _, done, _ = env.step(action)
        obs = next_obs
        if done:
            break
    return frames


def make_gif(env, agent, path, max_steps=200, policy="greedy"):
    frames = run_episode_frames(env, agent, policy, max_steps)
    save_gif(frames, path)
    return path

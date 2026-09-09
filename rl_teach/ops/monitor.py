"""训练过程监控：奖励/损失历史、成功率、耗时、资源占用。"""
import time
import numpy as np


class Monitor:
    def __init__(self):
        self.rewards = []
        self.losses = []
        self.success_flags = []
        self.start_time = time.time()
        self.peak_mem_mb = 0.0

    def record(self, episode, reward, loss, success):
        self.rewards.append(float(reward))
        self.losses.append(float(loss))
        self.success_flags.append(bool(success))
        self._update_resource()

    def _update_resource(self):
        try:
            import psutil
            p = psutil.Process()
            self.peak_mem_mb = max(self.peak_mem_mb, p.memory_info().rss / 1e6)
        except Exception:
            pass

    def summary(self, total_time, target, reward_hist, loss_hist):
        r = np.asarray(reward_hist, dtype=float)
        l = np.asarray(loss_hist, dtype=float)
        n = len(r)
        last_r = r[-100:] if n >= 100 else r
        last_l = l[-100:] if n >= 100 else l
        converged = (n >= target) and all(self.success_flags[-target:])
        return {
            "total_time": total_time,
            "episodes": n,
            "final_avg_reward": float(last_r.mean()) if n else 0.0,
            "best_reward": float(r.max()) if n else 0.0,
            "final_avg_loss": float(last_l.mean()) if n else 0.0,
            "success_rate": float(sum(self.success_flags)) / n if n else 0.0,
            "converged": bool(converged),
            "peak_mem_mb": self.peak_mem_mb,
        }

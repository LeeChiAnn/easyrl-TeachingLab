"""生成自包含的 HTML 运维报告（内嵌曲线图 base64）。"""
import base64
import os
from datetime import datetime


def _img_b64(path):
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        data = f.read()
    ext = os.path.splitext(path)[1].lstrip(".").lower()
    mime = "image/png" if ext == "png" else "image/gif"
    return "data:{};base64,{}".format(mime, base64.b64encode(data).decode())


def generate_report(path, summary, health, reward_png, loss_png, cfg):
    rimg = _img_b64(reward_png)
    limg = _img_b64(loss_png)
    badge_color = {"ok": "#2e7d32", "warning": "#ed6c02", "fail": "#d32f2f"}[health["status"]]
    badge_text = {"ok": "正常", "warning": "告警", "fail": "异常"}[health["status"]]

    rows = "".join(
        "<tr><td>{}</td><td>{}</td></tr>".format(k, v) for k, v in [
            ("环境", cfg["environment"]["name"]),
            ("算法", cfg["agent"]["type"]),
            ("总训练轮数", summary["episodes"]),
            ("总耗时(秒)", round(summary["total_time"], 2)),
            ("最终平均奖励", round(summary["final_avg_reward"], 3)),
            ("最佳奖励", round(summary["best_reward"], 3)),
            ("最终平均损失", round(summary["final_avg_loss"], 6)),
            ("成功率", "{:.2%}".format(summary["success_rate"])),
            ("是否收敛", "是" if summary["converged"] else "否"),
            ("峰值内存(MB)", round(summary["peak_mem_mb"], 1)),
        ]
    )
    msgs = "".join('<div class="msg">{}</div>'.format(m) for m in health["messages"])
    rimg_html = '<img src="{}">'.format(rimg) if rimg else "<p>未生成奖励曲线</p>"
    limg_html = '<img src="{}">'.format(limg) if limg else "<p>未生成损失曲线</p>"

    html = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>RL 训练运维报告</title>
<style>
body{{font-family:-apple-system,Segoe UI,Arial,sans-serif;margin:24px;color:#1f2d3d}}
h1{{font-size:20px}} h2{{font-size:16px;margin-top:24px}}
.badge{{display:inline-block;padding:4px 12px;border-radius:12px;color:#fff;font-weight:600}}
table{{border-collapse:collapse;margin:12px 0}} td{{border:1px solid #ddd;padding:8px 14px}}
.msg{{background:#f5f5f5;padding:8px 12px;border-left:4px solid #ccc;margin:8px 0}}
img{{max-width:100%;margin:12px 0;border:1px solid #eee}}
</style></head>
<body>
<h1>强化学习训练运维报告</h1>
<p>生成时间：{}</p>
<span class="badge" style="background:{}">健康状态：{}</span>
<h2>概览</h2><table>{}</table>
<h2>健康检查</h2>{}
<h2>奖励曲线</h2>{}
<h2>损失曲线</h2>{}
</body></html>""".format(
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        badge_color, badge_text, rows, msgs, rimg_html, limg_html,
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print("运维报告已生成:", path)

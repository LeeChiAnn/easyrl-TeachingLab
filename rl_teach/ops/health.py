"""健康检查：根据监控摘要给出状态与建议。"""


def check_health(summary, cfg):
    messages = []
    status = "ok"

    if summary["converged"]:
        messages.append("模型已连续达成成功阈值，可用于课堂演示。")
    elif summary["best_reward"] > 0 and summary["final_avg_reward"] >= 0.9 * summary["best_reward"]:
        messages.append("奖励接近最佳值，训练较稳定，可尝试增加轮数以完全收敛。")
    else:
        status = "warning"
        messages.append("尚未明显收敛，建议增大训练轮数或调整学习率/折扣因子。")

    if summary["success_rate"] < 0.1 and summary["episodes"] > 50:
        status = "warning"
        messages.append("成功率偏低，可检查奖励函数设计或超参数设置。")

    if status == "ok" and summary["final_avg_loss"] < 1e-6 and summary["episodes"] > 20:
        status = "warning"
        messages.append("损失异常偏低，可能存在更新停滞，建议检查学习率。")

    if not messages:
        messages.append("运行状态正常。")
    return {"status": status, "messages": messages}

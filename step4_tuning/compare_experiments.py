# ============================================================
# 阶段四：调参调优 - 实验对比分析
# ============================================================
# 功能：
#   1. 读取所有实验的 results.csv
#   2. 提取关键指标并对比
#   3. 生成对比表格和结论
#
# 训练在哪里做的？
#   训练是在 Google Colab (T4 GPU) 上完成的
#   本地 CPU 训练太慢（6小时/100轮），GPU 只需 20分钟/50轮
#
# 实验设计：
#   实验1 (Baseline): lr=0.01, SGD, mosaic=开启
#   实验2:            lr=0.001, SGD, mosaic=开启  (学习率缩小10倍)
#   实验3:            lr=0.01, SGD, mosaic=关闭  (关闭马赛克增强)
#
# 使用方法：
#   1. 从 Colab 下载 tuning_results.zip 到 step4_tuning/ 目录
#   2. 解压，确保有 exp1_lr001_sgd、exp2_lr0001_sgd、exp3_no_mosaic 三个文件夹
#   3. 运行：python compare_experiments.py
# ============================================================

import os
import csv

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 实验结果目录（Colab 下载的结果放在这里）
TUNING_DIR = os.path.join(PROJECT_ROOT, "step4_tuning")

# 各实验的配置和路径
experiments = [
    {
        "name": "实验1 (Baseline)",
        "config": "lr=0.01, SGD, mosaic=开",
        "dir": "exp1_lr001_sgd",
        "color": "\033[92m",  # 绿色
    },
    {
        "name": "实验2",
        "config": "lr=0.001, SGD, mosaic=开",
        "dir": "exp2_lr0001_sgd",
        "color": "\033[93m",  # 黄色
    },
    {
        "name": "实验3",
        "config": "lr=0.01, SGD, mosaic=关",
        "dir": "exp3_no_mosaic",
        "color": "\033[94m",  # 蓝色
    },
]

# ANSI 颜色重置
RESET = "\033[0m"


def load_best_metrics(csv_path):
    """
    从 results.csv 中读取最佳指标
    返回一个字典，包含 mAP50、mAP50-95、P、R 等
    """
    if not os.path.exists(csv_path):
        return None

    with open(csv_path, "r") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return None

    # 找 mAP50 最高的那一行
    best_row = max(rows, key=lambda r: float(r["metrics/mAP50(B)"]))

    return {
        "epoch": int(float(best_row["epoch"])),
        "precision": float(best_row["metrics/precision(B)"]),
        "recall": float(best_row["metrics/recall(B)"]),
        "mAP50": float(best_row["metrics/mAP50(B)"]),
        "mAP50_95": float(best_row["metrics/mAP50-95(B)"]),
    }


def main():
    print("=" * 70)
    print("  YOLOv8 调参实验对比分析")
    print("=" * 70)
    print()

    # 读取所有实验的指标
    results = []
    for exp in experiments:
        csv_path = os.path.join(TUNING_DIR, exp["dir"], "results.csv")
        metrics = load_best_metrics(csv_path)
        if metrics:
            results.append({**exp, **metrics})
            print(f"  ✅ {exp['name']} - {exp['config']}")
            print(f"     最佳 mAP50 = {metrics['mAP50']:.4f} (第 {metrics['epoch']} 轮)")
        else:
            print(f"  ❌ {exp['name']} - 找不到 results.csv")
        print()

    if len(results) < 2:
        print("实验数据不足，无法对比")
        return

    # 找最佳实验
    best = max(results, key=lambda r: r["mAP50"])

    # 打印对比表格
    print("=" * 70)
    print(f"  {'实验':<14} {'配置':<28} {'mAP50':>8} {'mAP50-95':>10} {'P':>8} {'R':>8}")
    print("-" * 70)

    for r in results:
        marker = " 🏆" if r["mAP50"] == best["mAP50"] else ""
        print(
            f"  {r['name']:<14} {r['config']:<28} "
            f"{r['mAP50']:>8.4f} {r['mAP50_95']:>10.4f} "
            f"{r['precision']:>8.4f} {r['recall']:>8.4f}{marker}"
        )

    print("=" * 70)
    print()

    # 分析结论
    print("=" * 70)
    print("  分析结论")
    print("=" * 70)
    print()

    # 1. 学习率对比（实验1 vs 实验2）
    exp1 = next((r for r in results if "exp1" in r["dir"]), None)
    exp2 = next((r for r in results if "exp2" in r["dir"]), None)
    exp3 = next((r for r in results if "exp3" in r["dir"]), None)

    if exp1 and exp2:
        diff = exp1["mAP50"] - exp2["mAP50"]
        print("【1. 学习率的影响】")
        print(f"  lr=0.01  vs  lr=0.001")
        print(f"  mAP50 差异: {diff:+.4f} ({diff*100:+.2f}%)")
        if diff > 0:
            print("  结论：默认学习率 0.01 效果更好。")
            print("        小学习率(0.001)在50轮内未充分收敛，精度偏低。")
        else:
            print("  结论：小学习率效果更好，可能默认学习率偏大。")
        print()

    # 2. 马赛克增强对比（实验1 vs 实验3）
    if exp1 and exp3:
        diff = exp1["mAP50"] - exp3["mAP50"]
        print("【2. 数据增强（马赛克）的影响】")
        print(f"  mosaic=开  vs  mosaic=关")
        print(f"  mAP50 差异: {diff:+.4f} ({diff*100:+.2f}%)")
        if abs(diff) < 0.01:
            print("  结论：差异不大（<1%），说明马赛克增强对该数据集增益有限。")
            print("        原因：数据集场景相对单一（人脸口罩），增强带来的多样性提升有限。")
        elif diff > 0:
            print("  结论：开启马赛克增强效果更好，数据增强有正面作用。")
        else:
            print("  结论：关闭马赛克效果反而更好，可能增强引入了过多噪声。")
        print()

    # 3. 最优配置
    print("【3. 最优配置】")
    print(f"  模型: yolov8n")
    print(f"  学习率: {best['config'].split(',')[0].split('=')[1]}")
    print(f"  优化器: SGD")
    print(f"  马赛克增强: {'开启' if 'mosaic=开' in best['config'] else '关闭'}")
    print(f"  最佳 mAP50: {best['mAP50']:.4f}")
    print(f"  最佳 mAP50-95: {best['mAP50_95']:.4f}")
    print()

    print("=" * 70)


if __name__ == "__main__":
    main()

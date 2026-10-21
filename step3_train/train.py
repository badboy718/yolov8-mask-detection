"""
阶段三：模型训练脚本

作用：使用 YOLOv8n 模型在口罩数据集上进行训练。

训练流程：
  1. 加载预训练的 yolov8n.pt 权重（迁移学习，从头训练太慢）
  2. 读取 data.yaml 配置，找到训练集和验证集
  3. 在训练集上学习，在验证集上评估
  4. 训练完成后，保存最优权重到 runs/detect/train/weights/best.pt

关键参数说明：
  - model:   使用 yolov8n（nano，最小模型，参数量约 3M，适合 CPU）
  - data:    数据集配置文件路径
  - epochs:  训练轮数，100 轮（1轮 = 把所有训练图片过一遍）
  - batch:   每次同时处理多少张图，16（CPU 内存不够可以调小到 8 或 4）
  - imgsz:   图片尺寸，640x640（YOLOv8 默认输入尺寸）

运行方式：
    python step3_train/train.py

训练结果保存在：
    runs/detect/train/
    ├── weights/
    │   ├── best.pt        # 验证集上表现最好的权重（用来推理）
    │   └── last.pt       # 最后一轮的权重
    ├── results.png       # 训练过程指标曲线图
    ├── confusion_matrix.png  # 混淆矩阵
    └── val_batch0_pred.jpg   # 验证集预测样例
"""

import os
import sys

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据集配置文件路径
DATA_YAML = os.path.join(PROJECT_ROOT, "data", "dataset", "data.yaml")

# 预训练权重路径（已下载到项目根目录，避免从 GitHub 下载超时）
WEIGHTS = os.path.join(PROJECT_ROOT, "yolov8n.pt")


def main():
    # 检查数据集配置文件是否存在
    if not os.path.exists(DATA_YAML):
        print(f"错误：找不到数据集配置文件 {DATA_YAML}")
        print("请先运行 step2_dataset/prepare_dataset.py 准备数据集")
        sys.exit(1)

    # 导入 YOLO 模型类
    from ultralytics import YOLO

    # --- 加载预训练模型 ---
    # yolov8n.pt 是 YOLOv8 官方在 COCO 数据集上预训练好的 nano 版本
    # 加载预训练权重叫"迁移学习"，比从零开始训练快得多、效果好得多
    # 注意：权重文件已提前下载到项目根目录，避免运行时从 GitHub 下载超时
    if not os.path.exists(WEIGHTS):
        print(f"错误：找不到预训练权重文件 {WEIGHTS}")
        print("请先下载 yolov8n.pt 到项目根目录")
        sys.exit(1)

    print("正在加载 yolov8n 预训练模型...")
    model = YOLO(WEIGHTS)

    # --- 开始训练 ---
    print("开始训练...")
    print(f"  模型: yolov8n (nano)")
    print(f"  数据集: {DATA_YAML}")
    print(f"  训练轮数: 100")
    print(f"  批次大小: 16")
    print(f"  图片尺寸: 640")
    print()

    results = model.train(
        data=DATA_YAML,
        epochs=100,       # 训练 100 轮
        batch=16,          # 每批 16 张图（CPU 内存不够就改成 8 或 4）
        imgsz=640,         # 图片统一缩放到 640x640
        project=os.path.join(PROJECT_ROOT, "runs", "detect"),
        name="train",      # 输出目录名
        # 以下参数使用默认值即可，调参阶段再调整
        # optimizer="auto",  # 自动选择优化器
        # lr0=0.01,          # 初始学习率
        # mosaic=True,       # 马赛克数据增强
    )

    print()
    print("=" * 50)
    print("  训练完成！")
    print("=" * 50)
    print(f"  最优权重: {os.path.join(PROJECT_ROOT, 'runs', 'detect', 'train', 'weights', 'best.pt')}")
    print(f"  训练曲线: {os.path.join(PROJECT_ROOT, 'runs', 'detect', 'train', 'results.png')}")
    print(f"  混淆矩阵: {os.path.join(PROJECT_ROOT, 'runs', 'detect', 'train', 'confusion_matrix.png')}")


if __name__ == "__main__":
    main()

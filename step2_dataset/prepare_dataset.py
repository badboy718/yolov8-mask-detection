"""
阶段二：数据集准备脚本

作用：将原始数据集（archive 目录）按比例划分为 训练集/验证集/测试集，
并生成 YOLOv8 需要的 data.yaml 配置文件。

数据集来源：Kaggle - Face Mask Detection (OpenCV University)
  - 原始数据在 data/archive/ 目录下，图片和标签文件混在一起
  - 每张图片 .jpg/.jpeg 对应一个同名 .txt 标签文件
  - 标签格式：class_id x_center y_center width height（坐标已归一化到 0~1）

划分后的目录结构：
  data/dataset/
  ├── images/
  │   ├── train/   (80% 的图片)
  │   ├── val/     (10% 的图片)
  │   └── test/    (10% 的图片)
  ├── labels/
  │   ├── train/   (对应的标签)
  │   ├── val/
  │   └── test/
  └── data.yaml   (YOLO 配置文件)

运行方式：
    python step2_dataset/prepare_dataset.py
"""

import os
import shutil
import random

# ============================================================
# 配置部分（可以修改的参数）
# ============================================================

# 项目根目录：本文件在 step2_dataset/ 下，所以根目录是上一级
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 原始数据目录（下载解压后的数据）
DATASET_DIR = os.path.join(PROJECT_ROOT, "data", "archive")

# 输出目录（划分后的数据集）
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "dataset")

# 划分比例：训练集 80%，验证集 10%，测试集 10%
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# 支持的图片格式
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

# 随机种子：固定种子保证每次划分结果一样，方便复现实验
random.seed(42)


# ============================================================
# 主逻辑
# ============================================================

def main():
    # --- 第1步：扫描所有图片文件 ---
    image_files = [
        f for f in os.listdir(DATASET_DIR)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
    ]
    print(f"找到 {len(image_files)} 张图片")

    # --- 第2步：检查每张图片是否有对应的标签文件 ---
    pairs = []          # 存放 (图片文件名, 标签文件名) 的配对
    missing_labels = []  # 记录缺少标签的图片

    for img in image_files:
        # 图片名 0.jpg -> 标签名 0.txt
        base = os.path.splitext(img)[0]
        label = base + ".txt"
        label_path = os.path.join(DATASET_DIR, label)

        if os.path.exists(label_path):
            pairs.append((img, label))
        else:
            missing_labels.append(img)

    if missing_labels:
        print(f"警告: {len(missing_labels)} 张图片缺少标签文件，已跳过")
    print(f"有效图片-标签对: {len(pairs)}")

    # --- 第3步：随机打乱并按比例划分 ---
    random.shuffle(pairs)
    total = len(pairs)
    train_count = int(total * TRAIN_RATIO)
    val_count = int(total * VAL_RATIO)

    splits = {
        "train": pairs[:train_count],                              # 前 80%
        "val":   pairs[train_count:train_count + val_count],       # 接下来 10%
        "test":  pairs[train_count + val_count:],                  # 最后 10%
    }

    # --- 第4步：创建目录结构并复制文件 ---
    for split_name, split_pairs in splits.items():
        img_dir = os.path.join(OUTPUT_DIR, "images", split_name)
        lbl_dir = os.path.join(OUTPUT_DIR, "labels", split_name)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        for img_file, lbl_file in split_pairs:
            # 复制图片到 images/train/ 等目录
            shutil.copy(
                os.path.join(DATASET_DIR, img_file),
                os.path.join(img_dir, img_file)
            )
            # 复制标签到 labels/train/ 等目录
            shutil.copy(
                os.path.join(DATASET_DIR, lbl_file),
                os.path.join(lbl_dir, lbl_file)
            )

        print(f"  {split_name}: {len(split_pairs)} 张")

    # --- 第5步：生成 data.yaml 配置文件 ---
    # data.yaml 是 YOLOv8 训练时必须的配置文件，告诉模型：
    #   - 数据集在哪（path）
    #   - 训练/验证/测试图片在哪个子目录（train/val/test）
    #   - 有几个类别（nc）
    #   - 每个类别叫什么名字（names）
    yaml_content = f"""path: {OUTPUT_DIR.replace(os.sep, '/')}
train: images/train
val: images/val
test: images/test

nc: 2
names: ['mask', 'no-mask']
"""
    yaml_path = os.path.join(OUTPUT_DIR, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"\ndata.yaml 已生成: {yaml_path}")
    print("\n数据集准备完成！")


if __name__ == "__main__":
    main()

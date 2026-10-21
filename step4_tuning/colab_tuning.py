# ============================================================
# YOLOv8 调参调优 - Google Colab 笔记本代码
# ============================================================
# 使用方法：
#   1. 打开 https://colab.research.google.com/
#   2. 新建笔记本（File -> New Notebook）
#   3. 把下面的代码按"单元格"分段复制粘贴
#   4. 每个单元格按 Shift+Enter 运行
# ============================================================


# ============================================================
# 单元格 1：安装依赖 + 检查 GPU
# ============================================================
# 这一步装好 ultralytics（YOLOv8 官方库），并确认 Colab 分配了 GPU
# 预期输出：ultralytics 安装成功，GPU 类型为 T4 或更好

!pip install ultralytics -q

import torch
from ultralytics import YOLO

print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU 型号: {torch.cuda.get_device_name(0)}")
else:
    print("警告：没有检测到 GPU！请点击菜单栏 Runtime -> Change runtime type -> T4 GPU")


# ============================================================
# 单元格 2：上传数据集
# ============================================================
# 方法A：直接上传（适合 216MB 的文件，大约 1-2 分钟上传）
# 运行后会弹出文件选择框，选择本地的 dataset.zip

from google.colab import files
uploaded = files.upload()  # 弹出选择框，选 dataset.zip

# 解压数据集
import zipfile
import os

zip_name = list(uploaded.keys())[0]
print(f"正在解压 {zip_name}...")

with zipfile.ZipFile(zip_name, 'r') as zip_ref:
    zip_ref.extractall('/content/dataset')

print("解压完成！")
print("数据集目录结构：")
for item in os.listdir('/content/dataset'):
    print(f"  /content/dataset/{item}")


# ---- 方法B（备选）：从 Google Drive 挂载 ----
# 如果方法A上传太慢，可以先把 dataset.zip 上传到你的 Google Drive
# 然后用下面的代码挂载 Drive：
#
# from google.colab import drive
# drive.mount('/content/drive')
# # 假设你把 zip 放在了 Drive 根目录
# import zipfile
# with zipfile.ZipFile('/content/drive/MyDrive/dataset.zip', 'r') as zip_ref:
#     zip_ref.extractall('/content/dataset')


# ============================================================
# 单元格 3：检查数据集 + 修改 data.yaml 路径
# ============================================================
# Colab 的文件路径和本地不一样，需要修改 data.yaml 里的路径

import yaml

data_yaml_path = '/content/dataset/data.yaml'

# 读取原始配置
with open(data_yaml_path, 'r') as f:
    data_config = yaml.safe_load(f)

print("原始配置：")
print(data_config)

# 修改路径为 Colab 的绝对路径
data_config['path'] = '/content/dataset'
data_config['train'] = 'images/train'
data_config['val'] = 'images/val'
data_config['test'] = 'images/test'

# 写回
with open(data_yaml_path, 'w') as f:
    yaml.dump(data_config, f, default_flow_style=False)

print("\n修改后配置：")
print(data_config)

# 统计数据集
for split in ['train', 'val', 'test']:
    img_dir = f'/content/dataset/images/{split}'
    if os.path.exists(img_dir):
        count = len([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])
        print(f"  {split}: {count} 张图片")


# ============================================================
# 单元格 4：实验1 - 调整学习率（lr0=0.01，默认值）
# ============================================================
# 这是 baseline 实验，和本地训练参数一样，用来对比
# 预计训练时间：T4 GPU 约 5-10 分钟

model = YOLO('yolov8n.pt')  # Colab 能直接从 GitHub 下载权重

results = model.train(
    data='/content/dataset/data.yaml',
    epochs=50,           # 调参阶段 50 轮够了，看趋势
    batch=16,
    imgsz=640,
    lr0=0.01,             # 学习率 = 0.01（YOLOv8 默认值）
    optimizer='SGD',      # 优化器 = SGD（默认）
    mosaic=1.0,           # 马赛克增强开启（默认）
    project='/content/runs/tuning',
    name='exp1_lr001_sgd',
    device=0,             # 使用 GPU
)

print("实验1完成！")
print(f"  mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
print(f"  mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")


# ============================================================
# 单元格 5：实验2 - 调整学习率（lr0=0.001，更小的学习率）
# ============================================================
# 看看更小的学习率是否能带来更好的精度
# 小学习率 = 收敛慢但可能更精细

model = YOLO('yolov8n.pt')

results2 = model.train(
    data='/content/dataset/data.yaml',
    epochs=50,
    batch=16,
    imgsz=640,
    lr0=0.001,            # 学习率 = 0.001（更小）
    optimizer='SGD',
    mosaic=1.0,
    project='/content/runs/tuning',
    name='exp2_lr0001_sgd',
    device=0,
)

print("实验2完成！")
print(f"  mAP50: {results2.results_dict.get('metrics/mAP50(B)', 'N/A')}")
print(f"  mAP50-95: {results2.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")


# ============================================================
# 单元格 6：实验3 - 关闭马赛克增强（mosaic=0）
# ============================================================
# 马赛克增强把 4 张图拼在一起训练，关闭它看效果差异
# 关闭增强可能让模型学得更"保守"，但减少过拟合

model = YOLO('yolov8n.pt')

results3 = model.train(
    data='/content/dataset/data.yaml',
    epochs=50,
    batch=16,
    imgsz=640,
    lr0=0.01,             # 学习率回到默认
    optimizer='SGD',
    mosaic=0.0,           # 关闭马赛克增强！
    project='/content/runs/tuning',
    name='exp3_no_mosaic',
    device=0,
)

print("实验3完成！")
print(f"  mAP50: {results3.results_dict.get('metrics/mAP50(B)', 'N/A')}")
print(f"  mAP50-95: {results3.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")


# ============================================================
# 单元格 7：实验4 - 换优化器 Adam（可选，如果你想多一组实验）
# ============================================================
# Adam 优化器收敛更快，但有时泛化不如 SGD

model = YOLO('yolov8n.pt')

results4 = model.train(
    data='/content/dataset/data.yaml',
    epochs=50,
    batch=16,
    imgsz=640,
    lr0=0.001,            # Adam 通常用更小的学习率
    optimizer='Adam',     # 换成 Adam 优化器
    mosaic=1.0,
    project='/content/runs/tuning',
    name='exp4_adam',
    device=0,
)

print("实验4完成！")
print(f"  mAP50: {results4.results_dict.get('metrics/mAP50(B)', 'N/A')}")
print(f"  mAP50-95: {results4.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")


# ============================================================
# 单元格 8：汇总对比所有实验结果
# ============================================================
# 把 4 组实验的关键指标列出来对比

import csv
import os

experiments = {
    '实验1 (lr=0.01, SGD, mosaic=on)': '/content/runs/tuning/exp1_lr001_sgd/results.csv',
    '实验2 (lr=0.001, SGD, mosaic=on)': '/content/runs/tuning/exp2_lr0001_sgd/results.csv',
    '实验3 (lr=0.01, SGD, mosaic=off)': '/content/runs/tuning/exp3_no_mosaic/results.csv',
    '实验4 (lr=0.001, Adam, mosaic=on)': '/content/runs/tuning/exp4_adam/results.csv',
}

print("=" * 70)
print("  调参实验对比结果")
print("=" * 70)
print(f"{'实验':<35} {'mAP50':>8} {'mAP50-95':>10} {'P':>8} {'R':>8}")
print("-" * 70)

for name, path in experiments.items():
    if not os.path.exists(path):
        print(f"{name:<35} {'未找到':>8}")
        continue
    with open(path, 'r') as f:
        rows = list(csv.DictReader(f))
    best_map50 = max(float(r['metrics/mAP50(B)']) for r in rows)
    best_map95 = max(float(r['metrics/mAP50-95(B)']) for r in rows)
    best_p = max(float(r['metrics/precision(B)']) for r in rows)
    best_r = max(float(r['metrics/recall(B)']) for r in rows)
    print(f"{name:<35} {best_map50:>8.4f} {best_map95:>10.4f} {best_p:>8.4f} {best_r:>8.4f}")

print("=" * 70)


# ============================================================
# 单元格 9：下载训练结果到本地
# ============================================================
# 把所有实验结果打包下载

import shutil

# 打包所有实验结果
shutil.make_archive('/content/tuning_results', 'zip', '/content/runs/tuning')

# 下载
files.download('/content/tuning_results.zip')

# 也下载每个实验的 best.pt
for exp_name in ['exp1_lr001_sgd', 'exp2_lr0001_sgd', 'exp3_no_mosaic', 'exp4_adam']:
    pt_path = f'/content/runs/tuning/{exp_name}/weights/best.pt'
    if os.path.exists(pt_path):
        files.download(pt_path)

print("下载完成！把文件放到项目的 step4_tuning/ 目录下")


# ============================================================
# 注意事项：
# 1. Colab 免费版有时长限制（约12小时），4组实验大概需要 30-60 分钟
# 2. 如果 Colab 断开连接，所有 /content/ 下的文件会丢失
#    建议每组实验跑完后立即下载 best.pt
# 3. 如果 GPU 被断开，重新连接可能需要等待
# 4. 训练过程中不要关闭浏览器标签页
# ============================================================

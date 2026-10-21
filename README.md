# 基于 YOLOv8 的口罩佩戴检测系统

人工智能软件开发课程期末作业

- 汇报人：刘淏荣（202411009111）
- 小组成员：姜瑞洋

## 项目说明

使用 YOLOv8 检测图片中的人脸是否佩戴口罩，类别为 `mask` 和 `no-mask`。

项目按实际流程拆成六个阶段：环境验证、数据准备、模型训练、调参实验、导出部署、网页演示。每个阶段一个脚本，可以单独运行。

数据集来自 Kaggle 的 Face Mask Detection（OpenCV University），共 1356 张图，按 8:1:1 划分为训练集 1084 张、验证集 135 张、测试集 137 张。

## 结果

| 项目 | 结果 |
|---|---|
| 本地训练 | 100 轮，CPU 约 6.14 小时，mAP@50 = 0.8931 |
| 调参实验 | 3 组控制变量，T4 50 轮，最好 mAP@50 = 0.8973 |
| 模型导出 | PyTorch `.pt` → ONNX |
| CPU 推理 | 14.9 FPS，约 67 ms / 张 |
| 网页演示 | Streamlit，单张约 74 ms |

三组调参实验的 mAP@50 差距只有 1.2 个百分点。学习率降到 0.001 会欠拟合，关掉 Mosaic 会掉 0.74 个百分点。结论是继续调参的收益有限，后续更应该补数据。

## 仓库地址

```text
git@github.com:badboy718/yolov8-mask-detection.git
```

## 目录结构

```text
yolo_v8_project/
├── README.md
├── requirements.txt
├── run_demo.bat
├── dataset.zip                  # 本地数据集压缩包，仓库不包含
├── yolov8n.pt                   # 预训练权重，仓库不包含
│
├── step1_env_setup/
│   └── verify_env.py
├── step2_dataset/
│   └── prepare_dataset.py
├── step3_train/
│   └── train.py
├── step4_tuning/
│   ├── colab_tuning.py
│   ├── compare_experiments.py
│   ├── best.pt                  # 调参后的最优权重
│   └── best.onnx                # 导出的 ONNX 模型
├── step5_deploy/
│   ├── deploy_and_evaluate.py
├── step6_web_demo/
│   └── app.py
│
├── data/dataset/                # 划分后的数据集（仓库已包含）
├── runs/detect/train/           # 本地训练结果（仓库已包含）
└── 期末汇报_PPT/
    └── 期末汇报_刘淏荣_姜瑞洋.pptx
```

## 环境

开发环境为 Python 3.10，实测版本见 `requirements.txt`。

```bash
pip install -r requirements.txt
```

如果只想跑网页演示，至少需要：

```bash
pip install ultralytics streamlit pillow
```

## 快速开始

### 1. 准备数据

仓库已经包含划分后的 `data/dataset/`，可以直接训练和运行网页演示。

如果要从原始数据重新划分，从 Kaggle 下载 Face Mask Detection 数据集，把原始图片和标签放到：

```text
data/archive/
```

然后运行：

```bash
python step2_dataset/prepare_dataset.py
```

脚本会重新生成：

```text
data/dataset/
├── images/{train,val,test}/
├── labels/{train,val,test}/
└── data.yaml
```

### 2. 训练

把 `yolov8n.pt` 放到项目根目录，然后运行：

```bash
python step3_train/train.py
```

默认训练 100 轮，CPU 上大约需要 6 小时。训练结果保存在 `runs/detect/train/`。

### 3. 调参实验

`step4_tuning/colab_tuning.py` 是给 Google Colab 用的，里面的数据路径是 `/content/drive/MyDrive/dataset/data.yaml`，在本机运行前需要改成自己的路径。

实验完成后，把结果目录放到 `step4_tuning/` 下，再运行：

```bash
python step4_tuning/compare_experiments.py
```

### 4. 导出与测速

```bash
python step5_deploy/deploy_and_evaluate.py
```

脚本会导出 ONNX，跑 5 张测试图，并测 CPU 推理速度。

### 5. 网页演示

先进入项目根目录（包含 `step1_env_setup` ~ `step6_web_demo` 的 `yolo_v8_project` 文件夹），再执行：

```bash
cd <你的路径>/yolo_v8_project
streamlit run step6_web_demo/app.py
```

如果提示 `File does not exist: step6_web_demo\app.py`，说明终端还在上一级目录，`cd` 到项目根目录即可。Windows 也可以直接双击 `run_demo.bat`。

## 数据和权重说明

仓库包含划分后的 `data/dataset/`、本地训练结果 `runs/detect/train/`，以及 `best.pt` 和 `best.onnx`，老师可以直接查看数据和结果，也可以直接运行网页演示。

原始压缩包 `dataset.zip` 和 `data/archive/` 没有上传，避免同一批图片重复三份。`yolov8n.pt` 会在训练时由 Ultralytics 自动下载，也可以手动放到项目根目录。

## 汇报材料

最终版 PPT 在 `期末汇报_PPT/` 目录。旧版、预览 PDF、素材、发言稿和 Q&A 不上传。

## 已知不足

- 数据场景单一，以正脸近照为主，侧脸、遮挡、暗光样本不足。
- 训练集和测试集的类别分布有差异，聚合 mAP 会掩盖单类别表现。
- CPU 推理还不到 30 FPS，不能直接做实时视频。
- 目前只做静态图片，没有视频流、跟踪和计数去重。
- 没有做 PT 与 ONNX 的严格精度对比，只验证了推理结果正常。

## 后续可以做的事

1. 补侧脸、遮挡、暗光、不同口罩类型的困难样本。
2. 试 YOLOv8s/m，或者做 INT8 量化。
3. 接 ONNX Runtime GPU / TensorRT，做视频流和简单跟踪。
4. 补类别分层评估和跨场景测试。

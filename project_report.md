# 基于 YOLOv8 的口罩佩戴检测系统

> 课程：人工智能软件开发
> 姓名：刘淏荣　　学号：202411009111

---

# 第一部分　汇报发言稿（按 PPT 每页）

> 建议时长 8–10 分钟。

## 第 1 页 · 封面（约 20 秒）

老师好，我是刘淏荣，学号 202411009111。我这次期末作业的题目是**基于 YOLOv8 的口罩佩戴检测系统**，下面我从数据集、训练、调参、部署和演示五个部分来汇报。

## 第 2 页 · 目录（约 10 秒）

整个项目按实际工作流程分成五个部分，前面四部分是技术实现，最后一部分是总结和改进方向。

## 第 3 页 · 项目背景与目标（约 50 秒）

选这个题目是因为口罩检测有比较明确的实际需求。虽然疫情过去了，但在医院、学校、商场这些场所，口罩佩戴的监管基本还是靠人工巡查，效率低、也容易漏。

所以我的目标是：**输入一张图片，模型自动找出人脸位置，并判断这个人有没有戴口罩**。这是一个二分类的目标检测任务，类别就是 mask 和 no-mask 两个。

右边是技术路线：数据集准备 → 本地训练 → 云端调参 → 导出部署 → 网页演示，整个流程是完整走通的。

## 第 4 页 · 数据集准备（约 40 秒）

数据用的是 Kaggle 上一个公开的口罩检测数据集，标注好的，一共 **1356 张**。

我按 8:1:1 划分成训练集 1084 张、验证集 135 张、测试集 137 张。

标注是 YOLO 的标准 TXT 格式，每行一个目标：类别编号加中心点坐标和宽高，都归一化到 0~1。右边两行就是示例，第一列 0 和 1 分别对应戴口罩和没戴口罩。

## 第 5 页 · 技术方案与训练环境（约 50 秒）

这一页说明为什么用了两种环境。

第一次训练在我自己笔记本上，只有 CPU，没有独立显卡，100 轮跑了 **6 个多小时**。好处是验证了不依赖云端、本地也能完整跑通。

但调参要做三组对比实验，本地要十几二十个小时，笔记本长时间满载发热，也不方便带出门。所以改用 Google Colab 的免费 T4 GPU，50 轮只要 **21 分钟**，比本地快大约 **17 倍**。

此外 Colab 有个坑：临时目录断线就会清空，所以数据必须放在 Google Drive 里挂载使用。

## 第 6 页 · 模型训练结果（约 40 秒）

这是本地 100 轮训练的曲线，损失平稳下降，没有明显发散。

最终 **mAP@50 达到 0.893**，mAP@50-95 是 0.595。最佳权重出现在第 73 到 87 轮之间，说明后面几十轮基本已经收敛。

## 第 7 页 · 调参实验对比（约 60 秒）

这是项目的重点。我设计了三组实验，用**控制变量法**，每组只改一个参数，都在 Colab 上训练 50 轮。

第一组是基线，学习率 0.01、SGD、开 mosaic，mAP@50 是 **0.8973**，最好。

第二组把学习率降到 0.001，降到 0.8853。学习率太小，收敛不够，属于欠拟合。

第三组关掉 mosaic 增强，降到 0.8899。Mosaic 只带来 0.74% 的增益，因为数据集场景比较单一，基本都是正脸近照，拼图增强帮助不大。

**这里我要说一句实话**：三组之间差距都很小。这本身就是一个结论——说明默认超参数已经接近这个数据集的上限，真正的瓶颈在数据，不在调参。

## 第 8 页 · 模型部署与推理测试（约 40 秒）

训练出的 .pt 是 PyTorch 专属格式，必须装 PyTorch 才能跑。所以我导出成 **ONNX** 通用格式，文件 6.2 MB，用 onnxruntime 就能加载。

本地 CPU 实测 **14.9 FPS，约 67 毫秒一张**。测试集随机取的 5 张图全部检测正确，左边是结果，框上标了类别和置信度。

## 第 9 页 · 网页演示系统（约 50 秒，含现场演示）

最后用 Streamlit 做了一个网页演示。

可以上传图片或直接选测试集示例图，左侧实时调置信度和 IoU 阈值，右侧看原图与检测结果对比、数量统计和检测明细。

这里做过一个优化：Streamlit 第一次推理要初始化计算图，会卡六秒，看起来像卡死。加了一次预热推理后，从 **6 秒降到约 80 毫秒**。

**（现场演示）** 我现在实际跑一遍——上传这张图，可以看到检测出 8 个戴口罩、1 个没戴。

## 第 10 页 · 总结与展望（约 40 秒）

总结：完整走通了数据集准备、训练、调参、部署到网页演示的全流程，本地训练 mAP@50 达到 0.893，云端调参提升到 0.897，完成 ONNX 导出和 CPU 部署。

不足和改进方向有三个：数据上补充侧脸、遮挡、暗光场景；模型上尝试更大规格；部署上往边缘设备或手机端走一步。

## 第 11 页 · 致谢（约 10 秒）

我的汇报到这里，谢谢老师，请老师批评指正。

---

# 第二部分　各阶段技术详解

> 说明：项目按工作流程拆成六个阶段目录，每个阶段一个独立脚本，可以单独运行。
> 路径均相对项目根目录 `yolo_v8_project/`。

## 阶段一　环境配置

**目录**：`step1_env_setup/verify_env.py`（72 行）

### 做了什么

在写训练代码之前，先确认三个关键库能正常导入：PyTorch、Ultralytics（YOLOv8）、OpenCV。目的是把「环境问题」和「代码问题」分开——如果训练报错，能快速排除是不是库没装好。

### 代码流程

```
检查 torch 是否可导入
  ├─ 能导入 → 打印版本、CUDA 是否可用
  └─ 不能   → 打印安装命令并 sys.exit(1)
检查 ultralytics 是否能导入（含 YOLO 类）
检查 cv2 是否可导入（非必需，只警告）
全部通过 → 打印「环境验证通过」
```

### 核心代码

判断 GPU 是否可用，决定了后续训练用 CPU 还是 CUDA：

```python
def check_torch():
    try:
        import torch
        print(f"[OK] PyTorch 版本: {torch.__version__}")
        print(f"[OK] CUDA 是否可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[OK] GPU 设备: {torch.cuda.get_device_name(0)}")
        else:
            print("[INFO] CUDA 不可用，将使用 CPU 训练（速度较慢但完全可行）")
    except ImportError:
        print("[FAIL] PyTorch 未安装！")
        sys.exit(1)
```

**要点**：本项目实测 `CUDA 是否可用 = False`，所以本地训练走 CPU 分支，这是后面训练耗时长 6 小时的直接原因。

---

## 阶段二　数据集准备

**目录**：`step2_dataset/prepare_dataset.py`（144 行）

### 做了什么

原始数据（`data/archive/`）里图片和标签是混在一起平铺的，YOLOv8 训练要求固定目录结构（`images/train`、`labels/train` 等）。这个脚本把它们按 **8:1:1** 拆开，并生成训练必需的 `data.yaml`。

### 代码流程

```
1. 扫描 archive 目录，按扩展名筛出所有图片（.jpg / .jpeg / .png）
2. 为每张图片找同名 .txt 标签 → 组成 (图片, 标签) 配对
   └─ 缺标签的图片单独记录并跳过
3. 固定随机种子 42，打乱配对列表
4. 按 80% / 10% / 10% 切片成 train / val / test
5. 逐对复制图片和标签到对应目录
6. 写出 data.yaml（路径、子目录、类别数、类别名）
```

### 核心代码

**① 图片与标签配对**——YOLO 靠「同名文件」对应图片和标签，所以必须成对处理：

```python
for img in image_files:
    base = os.path.splitext(img)[0]      # 0.jpg -> 0
    label = base + ".txt"                # 0.jpg -> 0.txt
    label_path = os.path.join(DATASET_DIR, label)

    if os.path.exists(label_path):
        pairs.append((img, label))
    else:
        missing_labels.append(img)
```

**② 用固定种子划分**——保证每次运行划分结果一致，实验可复现：

```python
random.seed(42)
random.shuffle(pairs)

total = len(pairs)
train_count = int(total * TRAIN_RATIO)   # 0.8
val_count = int(total * VAL_RATIO)       # 0.1

splits = {
    "train": pairs[:train_count],
    "val":   pairs[train_count:train_count + val_count],
    "test":  pairs[train_count + val_count:],
}
```

**③ 生成 data.yaml**——这是 YOLOv8 训练的唯一数据入口，`path` 是根目录，`train/val/test` 写相对子目录：

```python
yaml_content = f"""path: {OUTPUT_DIR.replace(os.sep, '/')}
train: images/train
val: images/val
test: images/test

nc: 2
names: ['mask', 'no-mask']
"""
```

**结果**：train 1084 张 / val 135 张 / test 137 张，共 1356 张。

**注意点**：扩展名必须包含 `.jpeg`。之前统计数据集时漏了 `.jpeg`，导致数量统计错误（报成 1023 张），实际是 1084 张。

---

## 阶段三　模型训练

**目录**：`step3_train/train.py`（99 行）

### 做了什么

加载官方在 COCO 数据集上预训练的 `yolov8n.pt`，在口罩数据集上训练 100 轮。这就是**迁移学习**：模型已经学会了「什么是边缘、纹理、物体」这些通用特征，只需要重新学习「戴口罩的人脸长什么样」，比从零训练快很多、效果也好很多。

### 代码流程

```
1. 检查 data.yaml 是否存在（不存在则提示先跑阶段二）
2. 检查本地 yolov8n.pt 是否存在（避免运行时从 GitHub 下载超时）
3. YOLO(WEIGHTS) 加载预训练模型
4. model.train(...) 开始训练
5. 训练完输出 best.pt / results.png / confusion_matrix.png 的路径
```

### 核心代码

```python
model = YOLO(WEIGHTS)      # 加载预训练权重

results = model.train(
    data=DATA_YAML,        # 阶段二生成的配置
    epochs=100,            # 100 轮
    batch=16,              # 每批 16 张（CPU 内存不够可降到 8 或 4）
    imgsz=640,             # 统一缩放到 640×640
    project=os.path.join(PROJECT_ROOT, "runs", "detect"),
    name="train",
)
```

**要点**：
- 加载 `yolov8n.pt` 时，Ultralytics 会自动把检测头从 COCO 的 80 类改成 `data.yaml` 里的 2 类，日志中显示为 `Overriding model.yaml nc=80 with nc=2`
- 训练产物在 `runs/detect/train/`，其中 `best.pt` 是验证集指标最好的那一轮权重，后续部署和网页演示都用它
- 实测：CPU 100 轮约 **6.14 小时**，mAP@50 = **0.8931**，最佳轮次在 73–87 轮之间

---

## 阶段四　调参调优

**目录**：`step4_tuning/colab_tuning.py`（273 行）、`step4_tuning/compare_experiments.py`（179 行）

### 做了什么

在 Colab T4 GPU 上做**控制变量**对比实验，每组只改一个超参数，训练 50 轮，然后横向比较 mAP。

脚本里实际写了 **4 组**配置，但**只执行了前 3 组**（第 4 组 Adam 优化器是预留的，没有跑），所以报告和 PPT 里只出现三组数据。

### 代码流程

```
colab_tuning.py（在 Colab 上运行）
  实验1: lr0=0.01,  SGD, mosaic=1.0  → exp1_lr001_sgd/
  实验2: lr0=0.001, SGD, mosaic=1.0  → exp2_lr0001_sgd/
  实验3: lr0=0.01,  SGD, mosaic=0.0  → exp3_no_mosaic/
  实验4: lr0=0.001, Adam, mosaic=1.0 → exp4_adam/（未执行）

compare_experiments.py（在本地运行）
  1. 遍历三个实验目录，读取各自的 results.csv
  2. 每个 csv 里每轮一行，按 mAP50 列取最大值 → 该实验的最佳成绩
  3. 横向打印对比表，标出最优实验
```

### 核心代码

**① 实验配置**——三组只有一处不同：

```python
# 实验1（基线）
model.train(data=..., epochs=50, batch=16, imgsz=640,
            lr0=0.01, optimizer='SGD', mosaic=1.0,
            project='/content/runs/tuning', name='exp1_lr001_sgd', device=0)

# 实验2：只改学习率
model.train(..., lr0=0.001, optimizer='SGD', mosaic=1.0,
            name='exp2_lr0001_sgd', device=0)

# 实验3：只改 mosaic
model.train(..., lr0=0.01, optimizer='SGD', mosaic=0.0,
            name='exp3_no_mosaic', device=0)
```

**② 从 results.csv 取最佳轮次**——训练日志每轮一行，不能直接取最后一行，要取 mAP50 最高那一行：

```python
def load_best_metrics(csv_path):
    rows = [...]  # 读取 csv 全部行
    best_row = max(rows, key=lambda r: float(r["metrics/mAP50(B)"]))
    return {
        "mAP50": float(best_row["metrics/mAP50(B)"]),
        "mAP50_95": float(best_row["metrics/mAP50-95(B)"]),
        ...
    }
```

### 实验结果

| 实验 | lr0 | 优化器 | Mosaic | mAP@50 | mAP@50-95 |
|------|-----|--------|--------|--------|-----------|
| 实验1（基线） | 0.01 | SGD | 开 | **0.8973** | 0.5937 |
| 实验2 | 0.001 | SGD | 开 | 0.8853 | 0.5778 |
| 实验3 | 0.01 | SGD | 关 | 0.8899 | 0.5938 |

**结论**：
1. 默认学习率 0.01 最好，降到 0.001 会欠拟合
2. 关闭 mosaic 只降 0.74%，说明该数据集场景单一，拼图增强收益有限
3. 三组差距都在 1% 以内，说明瓶颈不在超参数而在数据

**两个实际踩坑**：
- Colab 断线会清空 `/content/`，所以数据集必须放 Google Drive 挂载使用，脚本里的 `data` 路径实际改成了 `/content/drive/MyDrive/dataset/data.yaml`
- 结果文件先 `shutil.copy` 到 Drive 再下载，直接下载 53 MB 很慢

---

## 阶段五　模型部署与评估

**目录**：`step5_deploy/deploy_and_evaluate.py`（217 行）

### 做了什么

把 PyTorch 的 `.pt` 导出成通用格式 `.onnx`，然后做两件事：图片推理测试（看检测对不对）和 CPU 速度测试（看跑得快不快）。

### 代码流程

```
1. 检查 best.pt 是否存在
2. 检查是否装了 onnxruntime（没装就跳过导出，不中断流程）
3. model.export(format='onnx') → 生成 best.onnx
4. 从测试集取 5 张图，逐张推理 → plot() 画框 → 保存到 output/
5. 预热一次 → 连续推理 20 次 → 算平均耗时和 FPS
```

### 核心代码

**① 导出 ONNX**：

```python
model = YOLO(MODEL_PATH)
onnx_path = model.export(format="onnx")     # 同目录生成 best.onnx
```

**② 推理并保存带框的结果图**——`plot()` 返回的是 **BGR** 数组，OpenCV 保存没问题：

```python
results = model(img_path, verbose=False)
result_img = results[0].plot()               # 把检测框画在图上
cv2.imwrite(save_path, result_img)           # cv2 要求 BGR，正好匹配

# 统计各类别数量
for cls_id in results[0].boxes.cls:
    if int(cls_id) == 0:
        mask_count += 1
    else:
        nomask_count += 1
```

**③ 测速前先预热**——第一次推理要初始化计算图，不预热会把平均速度拉低：

```python
model(test_img_path, verbose=False)          # 预热一次，结果丢弃

for i in range(20):
    start = time.time()
    model(test_img_path, verbose=False)
    total_time += time.time() - start

fps = 1 / (total_time / 20)
```

**实测结果**：14.9 FPS（约 67 ms/张），5 张测试图全部检测成功。

---

## 阶段六　网页演示

**目录**：`step6_web_demo/app.py`（294 行）

### 做了什么

用 Streamlit 做一个网页：上传图片 → 模型推理 → 展示原图和检测结果对比、数量统计、推理耗时、检测明细表。

### 代码流程

```
1. 检测启动方式（必须 streamlit run，不能 python 直接跑）
2. 设置页面（标题、宽屏布局）
3. 侧边栏控件：模型格式（.pt / .onnx）、置信度阈值、IoU 阈值
4. 选择图片来源：上传图片 / 测试集示例图
5. load_model() 加载模型（带缓存 + 预热）
6. model(image, conf=..., iou=...) 推理
7. result.plot() 画框 → RGB 转换 → 等比缩放 → 左右两栏显示
8. 4 个指标卡片 + 检测明细表格
```

### 核心代码

**① 模型缓存 + 预热**——Streamlit 每次交互都会从头重跑整个脚本，不缓存的话每调一次滑块都要重新加载模型：

```python
@st.cache_resource
def load_model(model_path: str):
    model = YOLO(model_path)
    # 预热：提前跑一次空图，把第一次推理的初始化开销挪到加载阶段
    model.predict(Image.new("RGB", (640, 640)), verbose=False)
    return model
```

**② 传入 PIL 图片，不要传 numpy 数组**——这是个隐蔽的坑：

```python
# 直接传 PIL 对象
results = model(image, conf=conf_thres, iou=iou_thres, verbose=False)
```

原因：ultralytics 默认把 numpy 数组当作 OpenCV 的 **BGR** 格式处理。如果用 PIL 打开图片再转成 numpy（RGB），颜色通道会错乱，检测结果会变差。直接传 PIL 对象，库内部会按 RGB 正确处理。

**③ 显示前转 RGB**——`plot()` 返回 BGR，Streamlit 按 RGB 渲染，不转换会出现红蓝颠倒：

```python
annotated = result.plot()[..., ::-1]        # BGR → RGB
st.image(fit_width(annotated), caption=f"共检测到 {len(boxes)} 个目标")
```

**④ 等比缩放而不是强行指定宽度**——避免不同版本 Streamlit 的 `width` 参数行为不一致：

```python
def fit_width(img, max_width=560):
    if img.width <= max_width:
        return img
    ratio = max_width / img.width
    return img.resize((max_width, int(img.height * ratio)))
```

### 常见问题

| 现象 | 原因 | 解决 |
|------|------|------|
| `AttributeError: 'NoneType' object has no attribute 'width'` | 用 `python app.py` 启动，进入裸模式，所有组件返回 `None` | 改用 `streamlit run` 启动（脚本已加检测提示） |
| `File does not exist: step6_web_demo\app.py` | 当前目录不在项目根目录 | 先 `cd` 到项目根目录，或双击 `run_demo.bat` |
| 第一次推理卡 6 秒 | 首次推理要初始化计算图和内存池 | 加载模型时做一次预热推理（已实现） |
| 调滑块结果不变 | 自动化工具拖拽没有真正触发受控组件 | 手动点击或键盘方向键调整（实测正常） |

---

# 附　项目文件清单

```
yolo_v8_project/
├── README.md                       项目说明
├── run_demo.bat                    一键启动网页演示
├── generate_pptx.js                PPT 生成脚本
├── 期末汇报_刘淏荣.pptx             汇报 PPT（11 页）
├── yolov8n.pt                      预训练权重
├── dataset.zip                     数据集压缩包
├── step1_env_setup/verify_env.py   环境验证
├── step2_dataset/prepare_dataset.py 数据集划分
├── step3_train/train.py            本地训练
├── step4_tuning/
│   ├── colab_tuning.py             Colab 调参实验
│   ├── compare_experiments.py      实验结果对比
│   ├── exp1_lr001_sgd/             实验1 结果
│   ├── exp2_lr0001_sgd/            实验2 结果
│   ├── exp3_no_mosaic/             实验3 结果
│   ├── best.pt / best.onnx         最优模型
│   └── tuning_results.zip          Colab 结果打包
├── step5_deploy/
│   ├── deploy_and_evaluate.py      导出 + 推理 + 测速
│   └── output/                     推理结果图
├── step6_web_demo/app.py           Streamlit 网页演示
├── data/                           数据集（archive 原始 + dataset 划分后）
└── runs/detect/train/              本地训练输出
```

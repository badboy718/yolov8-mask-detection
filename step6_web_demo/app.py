# ============================================================
# 阶段六：Streamlit 网页演示 —— 图片口罩检测
# ============================================================
# 功能：
#   在网页上上传一张图片，模型自动检测图中的人是否佩戴口罩，
#   并展示检测框、类别统计和检测明细。
#
# 为什么用 Streamlit？
#   它把网页界面简化成 Python 函数调用（st.image / st.slider / st.columns ...），
#   不需要写 HTML / CSS / JavaScript，几十行代码就能做出可交互的演示页面。
#
# 运行方式（在项目根目录执行）：
#   streamlit run step6_web_demo/app.py
#   运行后浏览器会自动打开 http://localhost:8501
#
# 停止运行：在终端按 Ctrl + C
#
# 重要说明：
#   Streamlit 的运行模式是「每次交互都从头重跑一遍整个脚本」。
#   所以模型加载必须用 @st.cache_resource 缓存，
#   否则每上传一张图片都会重新加载一次模型（首次加载需要 20 秒以上）。
# ============================================================

from pathlib import Path
import time

import streamlit as st
from PIL import Image
from ultralytics import YOLO

# ------------------------------------------------------------
# 0. 检查启动方式是否正确
# ------------------------------------------------------------
# Streamlit 应用必须用 `streamlit run xxx.py` 启动，不能直接用 `python xxx.py`。
# 如果直接用 python 运行，会进入「裸模式」：拿不到浏览器交互上下文，
# 所有组件（radio / selectbox / slider）都会返回 None，
# 于是脚本会在后面莫名其妙地报 AttributeError，很难看懂原因。
# 所以这里提前检测，并给出明确的启动提示。
if not st.runtime.exists():
    print("=" * 60)
    print("启动方式不对：Streamlit 应用不能用 python 直接运行")
    print("=" * 60)
    print("请改用下面的命令启动：")
    print()
    print("    streamlit run step6_web_demo/app.py")
    print()
    print("（在 PyCharm 里不要点绿色三角 Run，请在下方 Terminal 里执行上面的命令）")
    print("=" * 60)
    raise SystemExit(1)

# ------------------------------------------------------------
# 1. 路径与常量配置
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent          # 项目根目录
PT_MODEL_PATH = BASE_DIR / "step4_tuning" / "best.pt"      # 训练得到的最优模型
ONNX_MODEL_PATH = BASE_DIR / "step4_tuning" / "best.onnx"  # 导出的 ONNX 模型
TEST_IMG_DIR = BASE_DIR / "data" / "dataset" / "images" / "test"  # 测试集图片

# 类别名 -> 界面上显示的中文名与图标
CLASS_CN = {"mask": "戴口罩", "no-mask": "未戴口罩"}
CLASS_ICON = {"mask": "🟢", "no-mask": "🔴"}

# 模型性能信息（来自阶段三训练 / 阶段四调参的结果），用于侧边栏展示
MODEL_INFO = {
    "模型规格": "yolov8n",
    "mAP50": "0.897",
    "参数量": "3.0 M",
    "文件大小": "6.2 MB",
}

# 页面能显示的最大宽度（像素），过大的图片先缩小，避免撑破页面
MAX_DISPLAY_WIDTH = 560

# ------------------------------------------------------------
# 2. 页面基础设置
# ------------------------------------------------------------
st.set_page_config(page_title="口罩佩戴检测", page_icon="😷", layout="wide")

st.title("😷 基于 YOLOv8 的口罩佩戴检测")
st.caption("人工智能软件开发 · 期末作业 —— 模型部署演示")


# ------------------------------------------------------------
# 3. 通用工具函数
# ------------------------------------------------------------
@st.cache_resource
def load_model(model_path: str):
    """
    加载 YOLO 模型，并立刻做一次预热推理。

    两点说明：
    1. 结果会被缓存：脚本重跑时直接复用，不会重复加载模型。
    2. 为什么要预热：模型第一次推理时需要初始化计算图、内存池和线程池，
       耗时异常高（实测单张图片要 6 秒以上，甚至 20 秒），
       预热之后再计时才是真实速度（约 60~120 毫秒）。
       预热用一张空白图，结果直接丢弃。

    参数：
        model_path: 模型文件路径（.pt 或 .onnx 都可以）
    """
    model = YOLO(model_path)
    model.predict(Image.new("RGB", (640, 640)), verbose=False)
    return model


def class_name_of(names, cls_id: int) -> str:
    """
    把模型输出的类别编号转成类别名。
    ultralytics 里 model.names 通常是 {0: 'mask', 1: 'no-mask'} 这种字典，
    但也可能是列表，这里两种都做兼容。
    """
    if isinstance(names, dict):
        return names.get(cls_id, str(cls_id))
    if isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
        return names[cls_id]
    return str(cls_id)


def fit_width(img: Image.Image, max_width: int = MAX_DISPLAY_WIDTH) -> Image.Image:
    """把过宽的图片等比缩小到指定宽度，避免网页上显示过大。"""
    if img.width <= max_width:
        return img
    ratio = max_width / img.width
    return img.resize((max_width, int(img.height * ratio)))


# ------------------------------------------------------------
# 4. 侧边栏：检测参数设置
# ------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ 检测设置")

    # 4.1 模型格式选择（.pt 是训练得到的原始格式，.onnx 是导出的通用格式）
    model_options = {"PyTorch (.pt)": PT_MODEL_PATH}
    if ONNX_MODEL_PATH.exists():
        model_options["ONNX (.onnx)"] = ONNX_MODEL_PATH

    model_label = st.radio("模型格式", list(model_options.keys()))
    selected_model_path = model_options[model_label]

    st.divider()

    # 4.2 置信度阈值：只保留置信度高于该值的检测框
    conf_thres = st.slider(
        "置信度阈值 (conf)",
        min_value=0.05, max_value=0.95, value=0.25, step=0.05,
        help="调高：检测更严格，误检变少但可能漏检；调低：更敏感，漏检变少但误检增多。",
    )

    # 4.3 IoU 阈值：非极大值抑制时判断两个框是否属于同一目标
    iou_thres = st.slider(
        "IoU 阈值",
        min_value=0.10, max_value=0.90, value=0.45, step=0.05,
        help="控制重叠检测框的合并程度，一般保持默认 0.45 即可。",
    )

    st.divider()
    st.subheader("📊 模型信息")
    for key, value in MODEL_INFO.items():
        st.markdown(f"- **{key}**：{value}")

    st.divider()
    st.caption("数据来源：Kaggle Face Mask Detection 数据集\n\n"
               "训练：Google Colab T4 GPU，推理：本地 CPU")

# ------------------------------------------------------------
# 5. 主区域：图片来源选择
# ------------------------------------------------------------
input_mode = st.radio("图片来源", ["上传图片", "使用示例图片"], horizontal=True)

image = None
image_caption = ""

if input_mode == "上传图片":
    uploaded_file = st.file_uploader("选择一张图片（jpg / jpeg / png）", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # 统一转成 RGB，避免灰度图 / 带透明通道的 PNG 出问题
        image = Image.open(uploaded_file).convert("RGB")
        image_caption = uploaded_file.name

else:
    # 用测试集里的图片做示例（这些图片模型训练时没见过，更能说明泛化能力）
    sample_files = []
    if TEST_IMG_DIR.exists():
        sample_files = sorted(
            p for p in TEST_IMG_DIR.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

    if sample_files:
        picked = st.selectbox(
            "选择测试集图片",
            sample_files,
            format_func=lambda p: p.name,
        )
        image = Image.open(picked).convert("RGB")
        image_caption = picked.name
    else:
        st.warning(f"没有找到示例图片，目录不存在或为空：{TEST_IMG_DIR}")

# 没有图片就直接结束本次渲染，等用户提供图片
if image is None:
    st.info("请上传一张图片，或切换到「使用示例图片」选择一张测试图。")
    st.stop()

# ------------------------------------------------------------
# 6. 执行检测
# ------------------------------------------------------------
# 加载模型（第一次调用会真正加载，之后走缓存）
model = load_model(str(selected_model_path))
names = getattr(model, "names", None) or {0: "mask", 1: "no-mask"}

try:
    # 计时：pref_counter 比 time.time() 精度更高，适合测毫秒级耗时
    start_time = time.perf_counter()
    # 注意：这里直接传 PIL 图片对象，不要先转成 numpy 数组。
    #       ultralytics 默认把 numpy 数组当作 OpenCV 的 BGR 格式处理，
    #       若传入 RGB 数组会导致颜色通道错乱，影响检测结果。
    results = model(image, conf=conf_thres, iou=iou_thres, verbose=False)
    elapsed_ms = (time.perf_counter() - start_time) * 1000
except Exception as error:
    st.error(f"检测失败：{error}")
    st.stop()

result = results[0]
boxes = result.boxes

# ------------------------------------------------------------
# 7. 结果展示——原图 vs 检测结果
# ------------------------------------------------------------
st.divider()

# plot() 返回的是 BGR 顺序的 numpy 数组（给 OpenCV 用的），
# 显示到网页前需要翻转成 RGB，否则口罩颜色会偏色。
annotated = result.plot()[..., ::-1]
annotated_image = Image.fromarray(annotated)

col_left, col_right = st.columns(2)
with col_left:
    st.subheader("原图")
    st.image(fit_width(image), caption=image_caption or "待检测图片")

with col_right:
    st.subheader("检测结果")
    st.image(fit_width(annotated_image), caption=f"共检测到 {len(boxes)} 个目标")

# ------------------------------------------------------------
# 8. 统计信息
# ------------------------------------------------------------
# 统计各类别检测到的数量
class_count = {"mask": 0, "no-mask": 0}
for cls_id in boxes.cls.tolist():
    name = class_name_of(names, int(cls_id))
    class_count[name] = class_count.get(name, 0) + 1

# 每张图的推理耗时换算成 FPS，便于直观判断能否做实时检测
fps = 1000 / elapsed_ms if elapsed_ms > 0 else 0

metric_cols = st.columns(4)
metric_cols[0].metric("🟢 戴口罩", f"{class_count.get('mask', 0)} 个")
metric_cols[1].metric("🔴 未戴口罩", f"{class_count.get('no-mask', 0)} 个")
metric_cols[2].metric("推理耗时", f"{elapsed_ms:.0f} ms")
metric_cols[3].metric("等效帧率", f"{fps:.1f} FPS")

st.caption("说明：推理耗时是稳定状态下的单张图片耗时（模型加载时已完成一次预热推理）；"
           "等效帧率 = 1000 / 耗时，可用来评估能否支撑实时视频检测。")

if len(boxes) == 0:
    st.warning("当前置信度阈值下没有检测到任何目标，可以试着把「置信度阈值」调低一些。")

# ------------------------------------------------------------
# 9. 检测明细表
# ------------------------------------------------------------
st.subheader("检测明细")

if len(boxes) > 0:
    rows = []
    for index, box in enumerate(boxes, start=1):
        cls_id = int(box.cls.item())
        name = class_name_of(names, cls_id)
        x1, y1, x2, y2 = (round(v) for v in box.xyxy[0].tolist())
        rows.append({
            "序号": index,
            "类别": f"{CLASS_ICON.get(name, '')} {CLASS_CN.get(name, name)}",
            "置信度": round(float(box.conf.item()), 3),
            "位置 (x1, y1, x2, y2)": f"({x1}, {y1}, {x2}, {y2})",
            "框大小": f"{x2 - x1} × {y2 - y1}",
        })
    st.dataframe(rows)
else:
    st.caption("暂无检测结果。")

st.divider()
st.caption("提示：调整左侧的置信度阈值，可以直观看到「误检」与「漏检」之间的权衡 —— 这正是调参阶段讨论的核心问题。")

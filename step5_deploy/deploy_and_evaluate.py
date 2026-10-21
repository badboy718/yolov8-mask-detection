# ============================================================
# 阶段五：模型部署与评估
# ============================================================
# 功能：
#   1. 模型导出（ONNX 格式，通用部署格式）
#   2. 图片推理测试
#   3. 速度测试（FPS）
#
# 什么是模型导出？
#   训练得到的 .pt 文件是 PyTorch 专属格式，只能在有 PyTorch 的环境里用
#   导出成 ONNX 格式后，可以在各种框架/设备上运行（TensorRT、OpenCV、手机等）
#   这就是"模型部署"的第一步——把模型变成通用格式
#
# 使用方法：
#   python deploy_and_evaluate.py
# ============================================================

import os
import time
import cv2
from ultralytics import YOLO

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 最优模型路径（从 Colab 下载的 best.pt）
MODEL_PATH = os.path.join(PROJECT_ROOT, "step4_tuning", "best.pt")

# 数据集路径（找几张测试图片用）
DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "dataset")

# 输出目录
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "step5_deploy", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def export_model():
    """
    导出模型为 ONNX 格式
    ONNX 是一种通用的神经网络模型格式，可以被各种推理框架读取
    注意：需要安装 onnxruntime 和 onnx 库，如果没装就跳过
    """
    print("=" * 60)
    print("  1. 模型导出（ONNX 格式）")
    print("=" * 60)

    if not os.path.exists(MODEL_PATH):
        print(f"❌ 找不到模型文件: {MODEL_PATH}")
        print("请先从 Colab 下载 best.pt 到 step4_tuning/ 目录")
        return None

    # 检查是否装了 onnxruntime，没装就跳过导出
    try:
        import onnxruntime
        has_onnx = True
    except ImportError:
        has_onnx = False

    if not has_onnx:
        print("⚠️  未安装 onnxruntime，跳过 ONNX 导出")
        print("   安装命令：pip install onnxruntime onnx -i https://pypi.tuna.tsinghua.edu.cn/simple")
        print("   （等有 WiFi 了再装也行，不影响推理测试）")
        print()
        return None

    model = YOLO(MODEL_PATH)

    print("正在导出为 ONNX 格式...")
    # export 方法会在同目录生成 best.onnx
    onnx_path = model.export(format="onnx")
    print(f"✅ 导出完成: {onnx_path}")

    # 显示文件大小
    pt_size = os.path.getsize(MODEL_PATH) / 1024 / 1024
    onnx_size = os.path.getsize(onnx_path) / 1024 / 1024
    print(f"   PyTorch (.pt):  {pt_size:.2f} MB")
    print(f"   ONNX (.onnx):   {onnx_size:.2f} MB")
    print()

    return onnx_path


def test_image_inference():
    """
    图片推理测试
    从测试集中选几张图片，用模型检测并保存结果
    """
    print("=" * 60)
    print("  2. 图片推理测试")
    print("=" * 60)

    model = YOLO(MODEL_PATH)

    # 从测试集找 5 张图片
    test_img_dir = os.path.join(DATASET_PATH, "images", "test")
    if not os.path.exists(test_img_dir):
        print(f"❌ 找不到测试图片目录: {test_img_dir}")
        return

    test_images = [
        f for f in os.listdir(test_img_dir)
        if f.endswith((".jpg", ".png"))
    ][:5]  # 取前 5 张

    print(f"测试图片数: {len(test_images)}")
    print()

    total_time = 0
    for i, img_name in enumerate(test_images):
        img_path = os.path.join(test_img_dir, img_name)

        # 推理
        start_time = time.time()
        results = model(img_path, verbose=False)
        end_time = time.time()

        infer_time = end_time - start_time
        total_time += infer_time

        # 保存检测结果图
        result_img = results[0].plot()  # 把检测框画在图上
        save_path = os.path.join(OUTPUT_DIR, f"result_{i+1}_{img_name}")
        cv2.imwrite(save_path, result_img)

        # 统计检测到的目标
        boxes = results[0].boxes
        mask_count = 0
        nomask_count = 0
        if boxes is not None:
            for cls_id in boxes.cls:
                if int(cls_id) == 0:
                    mask_count += 1
                else:
                    nomask_count += 1

        print(f"  [{i+1}/{len(test_images)}] {img_name}")
        print(f"      检测到: mask={mask_count}, no-mask={nomask_count}")
        print(f"      耗时: {infer_time*1000:.1f}ms")
        print(f"      保存: {save_path}")
        print()

    avg_time = total_time / len(test_images)
    print(f"平均推理时间: {avg_time*1000:.1f} ms / 张")
    print(f"推理速度: {1/avg_time:.1f} FPS")
    print()


def test_speed():
    """
    速度测试
    连续推理 10 张图，计算平均 FPS
    """
    print("=" * 60)
    print("  3. 速度测试（CPU）")
    print("=" * 60)

    model = YOLO(MODEL_PATH)

    # 预热（第一次推理会慢一些，因为要初始化）
    test_img = os.path.join(DATASET_PATH, "images", "test")
    test_images = [f for f in os.listdir(test_img) if f.endswith((".jpg", ".png"))][:1]
    if test_images:
        model(os.path.join(test_img, test_images[0]), verbose=False)

    # 正式测速：跑 20 次取平均
    num_runs = 20
    total_time = 0

    print(f"连续推理 {num_runs} 次...")
    for i in range(num_runs):
        img_path = os.path.join(test_img, test_images[0])
        start = time.time()
        model(img_path, verbose=False)
        end = time.time()
        total_time += (end - start)

    avg_time = total_time / num_runs
    fps = 1 / avg_time

    print(f"平均推理时间: {avg_time*1000:.1f} ms")
    print(f"推理速度: {fps:.1f} FPS")
    print()
    print("说明：这是 CPU (R7 8845HS) 上的速度")
    print("      如果用 GPU，速度会快 5~10 倍")
    print()


def main():
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║       YOLOv8 阶段五：模型部署与评估                  ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    # 1. 模型导出
    onnx_path = export_model()

    # 2. 图片推理测试
    test_image_inference()

    # 3. 速度测试
    test_speed()

    print("=" * 60)
    print("  部署评估完成！")
    print("=" * 60)
    print(f"  模型文件: {MODEL_PATH}")
    if onnx_path:
        print(f"  ONNX 模型: {onnx_path}")
    print(f"  推理结果: {OUTPUT_DIR}")
    print()
    print("下一步：制作课程作业 PPT")
    print("=" * 60)


if __name__ == "__main__":
    main()

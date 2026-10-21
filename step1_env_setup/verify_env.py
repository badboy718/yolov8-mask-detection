"""
阶段一：环境验证脚本

作用：检查 PyTorch 和 Ultralytics(YOLOv8) 是否正确安装，能否正常导入。
如果输出没有报错，说明环境配置成功。

运行方式：
    python step1_env_setup/verify_env.py
"""

import sys


def check_torch():
    """检查 PyTorch 是否安装，打印版本和 CUDA 支持情况"""
    try:
        import torch
        print(f"[OK] PyTorch 版本: {torch.__version__}")
        print(f"[OK] CUDA 是否可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[OK] CUDA 版本: {torch.version.cuda}")
            print(f"[OK] GPU 设备: {torch.cuda.get_device_name(0)}")
        else:
            print("[INFO] CUDA 不可用，将使用 CPU 训练（速度较慢但完全可行）")
    except ImportError:
        print("[FAIL] PyTorch 未安装！请运行: pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
        sys.exit(1)


def check_ultralytics():
    """检查 Ultralytics(YOLOv8) 是否安装"""
    try:
        import ultralytics
        from ultralytics import YOLO
        print(f"[OK] Ultralytics 版本: {ultralytics.__version__}")
        print(f"[OK] YOLO 模型类导入成功")
    except ImportError:
        print("[FAIL] Ultralytics 未安装！请运行: pip install ultralytics")
        sys.exit(1)


def check_opencv():
    """检查 OpenCV 是否安装（用于图像读取和可视化）"""
    try:
        import cv2
        print(f"[OK] OpenCV 版本: {cv2.__version__}")
    except ImportError:
        print("[WARN] OpenCV 未安装，建议运行: pip install opencv-python")


def main():
    print("=" * 50)
    print("  YOLOv8 环境验证")
    print("=" * 50)
    print()

    check_torch()
    print()

    check_ultralytics()
    print()

    check_opencv()
    print()

    print("=" * 50)
    print("  环境验证通过！可以开始训练了。")
    print("=" * 50)


if __name__ == "__main__":
    main()

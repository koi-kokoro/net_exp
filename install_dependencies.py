"""
一键安装YOLO+PaddleOCR依赖
"""

import subprocess
import sys
import os

def install_requirements():
    """安装所有依赖"""
    print("=" * 70)
    print("安装 YOLO + PaddleOCR 依赖")
    print("=" * 70)
    print()
    
    requirements = [
        ("PaddleOCR", "paddleocr>=2.7.0"),
        ("PaddlePaddle", "paddlepaddle>=2.5.0"),
        ("EasyOCR (备用)", "easyocr>=1.7.0"),
        ("Ultralytics YOLO", "ultralytics>=8.0.0"),
        ("PyTorch", "torch>=1.9.0"),
        ("TorchVision", "torchvision>=0.10.0"),
        ("OpenCV", "opencv-python>=4.5.0"),
        ("Pillow", "Pillow>=9.0.0"),
        ("Flask", "flask>=2.3.0"),
    ]
    
    total = len(requirements)
    failed = []
    
    for i, (name, package) in enumerate(requirements, 1):
        print(f"[{i}/{total}] 安装 {name}... ", end="", flush=True)
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package, "-q"
            ])
            print("✓ 完成")
        except subprocess.CalledProcessError as e:
            print("✗ 失败")
            failed.append((name, package))
    
    print()
    print("=" * 70)
    
    if failed:
        print(f"警告: {len(failed)} 个包安装失败:")
        for name, package in failed:
            print(f"  - {name} ({package})")
        print()
        print("可以尝试手动安装:")
        print("  pip install " + " ".join(p.split(">")[0] for _, p in failed))
        return False
    else:
        print("所有依赖安装成功！")
        print()
        print("下一步:")
        print("  1. 将车牌图片放在 images/uploads 文件夹")
        print("  2. 运行测试: python test_yolo_integration.py")
        print("  3. 启动服务器: python start_server.bat (Windows)")
        return True

if __name__ == "__main__":
    try:
        success = install_requirements()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n安装被用户中止")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n错误: {e}")
        sys.exit(1)

"""
车牌检测模块 — YOLO + OpenCV 混合检测
1. 优先使用 YOLO 检测（需预训练车牌检测模型）
2. 降级使用 OpenCV 颜色+轮廓检测（对蓝牌/绿牌效果好）
3. 检测结果裁剪后传给 OCR，大幅提高识别准确率
"""

import os
import cv2
import numpy as np

# ==================== YOLO 车牌检测 ====================

_yolo_model = None
_yolo_available = False
_yolo_checked = False


def _check_yolo():
    """检查 YOLO 车牌检测模型是否可用"""
    global _yolo_model, _yolo_available, _yolo_checked
    if _yolo_checked:
        return _yolo_available
    _yolo_checked = True

    try:
        from ultralytics import YOLO
        import torch

        # 可能的车牌检测模型路径（按优先级）
        model_paths = [
            os.path.join(os.path.dirname(__file__), "yolo", "YOLOv8", "runs", "detect", "train", "weights", "best.pt"),
            os.path.join(os.path.dirname(__file__), "yolo", "YOLOv8", "runs", "detect", "train", "weights", "last.pt"),
            "yolov8n.pt",  # 通用检测模型（COCO，不专门检测车牌）
        ]

        for mp in model_paths:
            if os.path.exists(mp):
                try:
                    _yolo_model = YOLO(mp)
                    _yolo_available = True
                    print(f"[PLATE-DETECT] ✅ YOLO 模型加载成功: {mp}")
                    return True
                except Exception as e:
                    print(f"[PLATE-DETECT] ⚠️ 模型 {mp} 加载失败: {e}")

        print("[PLATE-DETECT] ⚠️ 未找到车牌检测模型，将使用 OpenCV 颜色检测")
        print("[PLATE-DETECT]    提示: 如需使用 YOLO 检测，请下载车牌检测模型到 yolo/YOLOv8/runs/detect/train/weights/best.pt")
        return False

    except ImportError as e:
        print(f"[PLATE-DETECT] ⚠️ ultralytics 不可用: {e}")
        return False


def detect_plate_yolo(image_path: str, conf_threshold: float = 0.3):
    """
    使用 YOLO 检测车牌区域

    返回: (x1, y1, x2, y2) 或 None
    """
    _check_yolo()
    if not _yolo_available or _yolo_model is None:
        return None

    try:
        results = _yolo_model(image_path, conf=conf_threshold, verbose=False)
        for result in results:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                # 找置信度最高的框
                best_idx = boxes.conf.argmax()
                box = boxes.xyxy[best_idx].cpu().numpy()
                return tuple(int(v) for v in box)  # (x1, y1, x2, y2)
    except Exception as e:
        print(f"[PLATE-DETECT] YOLO 检测出错: {e}")

    return None


# ==================== OpenCV 颜色+轮廓检测 ====================

def detect_plate_opencv(image_path: str):
    """
    使用 OpenCV 颜色检测 + 轮廓查找 定位车牌

    支持：
    - 蓝色车牌（传统燃油车）
    - 绿色车牌（新能源车）

    返回: (x1, y1, x2, y2) 或 None
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    h, w = img.shape[:2]
    boxes = []

    # ---- 蓝色车牌检测 ----
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 蓝色范围 (HSV)
    lower_blue = np.array([100, 60, 60])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # 形态学操作：闭运算填补空隙，开运算去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)

    blue_boxes = _find_plate_contours(blue_mask, img)
    boxes.extend(blue_boxes)

    # ---- 绿色车牌检测（新能源） ----
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([90, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)

    green_boxes = _find_plate_contours(green_mask, img)
    boxes.extend(green_boxes)

    if not boxes:
        return None

    # 选面积最大的候选框
    best = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    return best


def _find_plate_contours(mask, img):
    """从颜色掩码中找到候选车牌矩形"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_img, w_img = img.shape[:2]
    boxes = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        # 过滤太小的区域（至少占图片 0.2%）
        if area < w_img * h_img * 0.002:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect_ratio = bw / max(bh, 1)

        # 中国车牌宽高比约 3.14:1，允许范围 1.8 ~ 5.5
        if 1.8 <= aspect_ratio <= 5.5:
            # 扩展边界 10% 留点余量
            pad_x = int(bw * 0.1)
            pad_y = int(bh * 0.15)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w_img, x + bw + pad_x)
            y2 = min(h_img, y + bh + pad_y)
            boxes.append((x1, y1, x2, y2))

    return boxes


# ==================== 主检测函数 ====================

def detect_plate(image_path: str) -> dict:
    """
    检测车牌位置，返回裁剪后的车牌图片路径

    检测策略（按优先级）：
    1. YOLO（如果模型可用）
    2. OpenCV 颜色检测（蓝牌/绿牌）

    返回:
        {
            "success": True/False,
            "plate_image_path": "裁剪后的车牌图片路径" 或 None,
            "bbox": (x1,y1,x2,y2) 或 None,
            "method": "yolo" / "opencv" / "none",
            "msg": "..."
        }
    """
    if not os.path.exists(image_path):
        return {"success": False, "plate_image_path": None, "bbox": None,
                "method": "none", "msg": "图片不存在"}

    bbox = None
    method = "none"

    # 1. 尝试 YOLO
    bbox = detect_plate_yolo(image_path)
    if bbox is not None:
        method = "yolo"

    # 2. 降级到 OpenCV 颜色检测
    if bbox is None:
        bbox = detect_plate_opencv(image_path)
        if bbox is not None:
            method = "opencv"

    if bbox is None:
        return {"success": False, "plate_image_path": None, "bbox": None,
                "method": "none",
                "msg": "未检测到车牌区域，请确保图片中有清晰的中国车牌（蓝牌或绿牌）"}

    # 裁剪车牌区域
    img = cv2.imread(image_path)
    if img is None:
        return {"success": False, "plate_image_path": None, "bbox": None,
                "method": "none",
                "msg": "读取原始图片失败"}
    
    x1, y1, x2, y2 = bbox
    
    # 验证 bbox 坐标的有效性
    img_h, img_w = img.shape[:2]
    if x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h or x1 >= x2 or y1 >= y2:
        print(f"[PLATE-DETECT] ⚠️ bbox 坐标超出范围: ({x1},{y1},{x2},{y2}) vs 图片({img_w}x{img_h})")
        return {"success": False, "plate_image_path": None, "bbox": None,
                "method": "none",
                "msg": "检测到的车牌区域坐标无效"}
    
    plate_img = img[y1:y2, x1:x2]
    
    # 检查裁剪结果是否为空
    if plate_img is None or plate_img.size == 0:
        print(f"[PLATE-DETECT] ⚠️ 裁剪后图像为空")
        return {"success": False, "plate_image_path": None, "bbox": None,
                "method": "none",
                "msg": "裁剪图像为空"}
    
    crop_h, crop_w = plate_img.shape[:2]
    
    # 图像太小的话，放大处理
    min_width, min_height = 50, 20
    if crop_w < min_width or crop_h < min_height:
        print(f"[PLATE-DETECT] ⚠️ 裁剪图像过小 ({crop_w}x{crop_h})，放大处理")
        scale = max(min_width / crop_w, min_height / crop_h)
        new_w = int(crop_w * scale)
        new_h = int(crop_h * scale)
        plate_img = cv2.resize(plate_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        print(f"[PLATE-DETECT] 放大后: {new_w}x{new_h}")

    # 保存裁剪结果
    import tempfile
    fd, crop_path = tempfile.mkstemp(suffix='_plate.jpg', prefix='plate_crop_')
    os.close(fd)
    success = cv2.imwrite(crop_path, plate_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    if not success:
        print(f"[PLATE-DETECT] ⚠️ 保存裁剪图像失败")
        return {"success": False, "plate_image_path": None, "bbox": None,
                "method": "none",
                "msg": "保存裁剪图像失败"}

    print(f"[PLATE-DETECT] ✅ 检测到车牌 (方法={method}) bbox={bbox} 裁剪大小={crop_w}x{crop_h}")

    return {
        "success": True,
        "plate_image_path": crop_path,
        "bbox": bbox,
        "method": method,
        "msg": f"检测到车牌区域 ({method})"
    }


def cleanup_plate_image(plate_image_path: str):
    """清理临时车牌裁剪图片"""
    if plate_image_path and os.path.exists(plate_image_path):
        try:
            os.remove(plate_image_path)
        except:
            pass

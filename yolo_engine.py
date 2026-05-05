"""
YOLO Engine Module - License Plate Detection using YOLOv8
该模块使用YOLO进行车牌位置检测和ROI提取
"""

import cv2
import numpy as np
import os
import torch
from typing import List, Tuple, Optional

# 尝试导入YOLO
try:
    from yolo.YOLOv8.ultralytics import YOLO
    YOLO_AVAILABLE = True
except:
    YOLO_AVAILABLE = False

_yolo_model = None
_model_lock = None
import threading
_model_lock = threading.Lock()


def init_yolo_model(model_path: str = None) -> bool:
    """
    初始化YOLO模型
    Args:
        model_path: 模型路径，如为None则使用默认的yolov8n.pt
    Returns:
        是否初始化成功
    """
    global _yolo_model, YOLO_AVAILABLE
    
    if not YOLO_AVAILABLE:
        print("[YOLO] YOLO not available, skipping initialization")
        return False
    
    try:
        with _model_lock:
            if _yolo_model is not None:
                return True
            
            # 使用默认模型或指定模型
            if model_path is None:
                # 尝试从本地YOLO目录加载，否则使用默认yolov8n.pt
                default_paths = [
                    "./yolo/YOLOv8/yolov8s.pt",
                    "./yolov8n.pt",
                    "yolov8n.pt"
                ]
                model_path = None
                for path in default_paths:
                    if os.path.exists(path):
                        model_path = path
                        break
                
                if model_path is None:
                    print("[YOLO] Warning: No local model found, will download yolov8n.pt from internet")
                    model_path = "yolov8n.pt"
            
            print(f"[YOLO] Loading model from {model_path}...")
            _yolo_model = YOLO(model_path)
            print("[YOLO] Model loaded successfully")
            return True
    except Exception as e:
        print(f"[YOLO] Failed to load model: {e}")
        YOLO_AVAILABLE = False
        return False


def detect_license_plates(image_path: str, conf_threshold: float = 0.3) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """
    使用YOLO检测图像中的车牌位置
    Args:
        image_path: 图像路径
        conf_threshold: 置信度阈值
    Returns:
        [(裁剪的车牌图像, (x1, y1, x2, y2)), ...] 列表
    """
    global _yolo_model
    
    if _yolo_model is None:
        return []
    
    try:
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"[YOLO] Failed to read image: {image_path}")
            return []
        
        # YOLO推理
        results = _yolo_model(image, conf=conf_threshold, verbose=False)
        
        detections = []
        for result in results:
            for box in result.boxes:
                # 获取边界框坐标
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                # 确保坐标在图像范围内
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(image.shape[1], x2)
                y2 = min(image.shape[0], y2)
                
                # 裁剪ROI区域
                roi = image[y1:y2, x1:x2]
                if roi.size > 0:  # 确保ROI有效
                    detections.append((roi, (x1, y1, x2, y2)))
        
        return detections
    except Exception as e:
        print(f"[YOLO] Detection failed: {e}")
        return []


def detect_and_crop_plates(image_path: str, expand_ratio: float = 0.1) -> List[Tuple[np.ndarray, dict]]:
    """
    检测车牌并进行智能裁剪，支持扩展边界和角度校正
    Args:
        image_path: 图像路径
        expand_ratio: 边界扩展比例（例如0.1表示扩展10%）
    Returns:
        [(裁剪的车牌图像, {"box": (x1,y1,x2,y2), "confidence": conf}), ...]
    """
    detections = detect_license_plates(image_path)
    
    if not detections:
        return []
    
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    
    results = []
    for roi, (x1, y1, x2, y2) in detections:
        # 计算扩展
        bbox_w = x2 - x1
        bbox_h = y2 - y1
        expand_w = int(bbox_w * expand_ratio)
        expand_h = int(bbox_h * expand_ratio)
        
        # 应用扩展
        x1_expanded = max(0, x1 - expand_w)
        y1_expanded = max(0, y1 - expand_h)
        x2_expanded = min(w, x2 + expand_w)
        y2_expanded = min(h, y2 + expand_h)
        
        # 提取扩展后的ROI
        roi_expanded = image[y1_expanded:y2_expanded, x1_expanded:x2_expanded]
        
        results.append((roi_expanded, {
            "box": (x1, y1, x2, y2),
            "box_expanded": (x1_expanded, y1_expanded, x2_expanded, y2_expanded),
            "confidence": 0.0  # YOLO检测框本身的置信度（如需要可额外保存）
        }))
    
    return results


def extract_plate_regions_fallback(image_path: str) -> List[Tuple[np.ndarray, dict]]:
    """
    当YOLO不可用时的备用方案：使用形态学和边缘检测寻找矩形区域
    这是一个简单的启发式方法，用于在没有预训练模型时工作
    Args:
        image_path: 图像路径
    Returns:
        [(裁剪的车牌图像, {"box": (x1,y1,x2,y2)}), ...]
    """
    image = cv2.imread(image_path)
    if image is None:
        return []
    
    # 转灰度
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 直方图均衡化
    gray = cv2.equalizeHist(gray)
    
    # 边缘检测
    edges = cv2.Canny(gray, 30, 100)
    
    # 膨胀和腐蚀
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.erode(edges, kernel, iterations=1)
    
    # 找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    results = []
    h, w = image.shape[:2]
    
    # 筛选可能是车牌的矩形轮廓
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        
        # 车牌宽高比通常在2.5-4之间
        aspect_ratio = cw / ch if ch > 0 else 0
        area = cw * ch
        
        # 过滤条件
        if 2.0 < aspect_ratio < 5.0 and area > 1000 and area < (h * w * 0.3):
            roi = image[y:y+ch, x:x+cw]
            results.append((roi, {
                "box": (x, y, x+cw, y+ch),
                "source": "fallback_morphology"
            }))
    
    return results

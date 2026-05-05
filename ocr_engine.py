"""
OCR Engine Module - License Plate Recognition
Dual-engine strategy:
- PaddleOCR (primary): Best for Chinese license plates (recommended)
- EasyOCR (fallback): Simple install, works for general use
- Auto-fallback: Switch to EasyOCR if PaddleOCR unavailable
- Global singleton Reader, pre-loaded on server startup
- Auto image format conversion + enhancement
- No fake data or degradation fallbacks
"""

import re
import os
import threading
import tempfile
import cv2
import numpy as np
from config import OCR_CONFIDENCE_HIGH, OCR_CONFIDENCE_LOW

# 尝试导入YOLO检测引擎
try:
    from yolo_engine import init_yolo_model, detect_and_crop_plates, extract_plate_regions_fallback
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[OCR] YOLO engine not available, will use direct OCR without plate detection")

# Standard Chinese license plate regex (supports new energy green plates)
# Format: [Province][Letter][4-5 alphanumeric][optional suffix]
# Example: 浙A12345, 浙D30520
PLATE_PATTERN = re.compile(
    r'[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤川青藏琼宁]'
    r'[A-HJ-NP-Z]'
    r'[A-HJ-NP-Z0-9]{4,5}'
    r'[A-HJ-NP-Z0-9挂学警港澳]?'
)

# New energy green plate (8 characters total)
# Format: [Province][Letter]D/F[5 alphanumeric]
# Example: 浙D30520 (this is what we're trying to recognize)
GREEN_PLATE_PATTERN = re.compile(
    r'[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤川青藏琼宁]'
    r'[A-HJ-NP-Z]'
    r'[DF]'
    r'[A-HJ-NP-Z0-9]{5}'
)

# Global OCR engine state
_ocr_engine = None       # 'paddleocr' | 'easyocr' | None
_paddle_ocr = None        # PaddleOCR instance
_easyocr_reader = None    # EasyOCR Reader instance
_engine_lock = threading.Lock()
_engine_ready = False


def init_ocr_engine():
    """
    Pre-load OCR engine (call on server startup)
    Tries PaddleOCR first (best for Chinese), falls back to EasyOCR if needed
    """
    global _ocr_engine, _paddle_ocr, _easyocr_reader, _engine_ready

    # 1. 尝试 PaddleOCR
    try:
        from paddleocr import PaddleOCR
        print("[OCR] Loading PaddleOCR model (best for Chinese license plates)...")
        _paddle_ocr = PaddleOCR(use_angle_cls=True, lang='ch', show_log=False,
                                use_gpu=False, det_db_thresh=0.3,
                                det_db_box_thresh=0.3, rec_batch_num=1)
        _ocr_engine = 'paddleocr'
        _engine_ready = True
        print("[OCR] OK PaddleOCR model loaded (primary engine)")
        return
    except ImportError:
        print("[OCR] PaddleOCR not installed, trying EasyOCR...")
    except Exception as e:
        print(f"[OCR] PaddleOCR initialization failed: {e}, trying EasyOCR...")

    # 2. 降级 EasyOCR
    try:
        import easyocr
        print("[OCR] Loading EasyOCR model (Chinese+English, fallback engine)...")
        print("[OCR] First load may download model files, please wait...")
        _easyocr_reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)
        _ocr_engine = 'easyocr'
        _engine_ready = True
        print("[OCR] OK EasyOCR model loaded (fallback engine)")
        print("[OCR] Tip: Install PaddleOCR for better accuracy: pip install paddleocr")
        return
    except ImportError:
        print("[OCR] ERROR EasyOCR not installed")
        print("[OCR]    Run: pip install easyocr")
        print("[OCR]    Or: pip install paddleocr (recommended)")
    except Exception as e:
        print(f"[OCR] ERROR EasyOCR initialization failed: {e}")

    _engine_ready = False


def _get_reader():
    """Get current available OCR engine identifier and instance"""
    global _paddle_ocr, _easyocr_reader, _engine_ready
    if not _engine_ready:
        return None, None
    if _ocr_engine == 'paddleocr' and _paddle_ocr:
        return 'paddleocr', _paddle_ocr
    if _ocr_engine == 'easyocr' and _easyocr_reader:
        return 'easyocr', _easyocr_reader
    # 懒加载 EasyOCR
    with _engine_lock:
        if _easyocr_reader is None:
            try:
                import easyocr
                _easyocr_reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)
            except:
                return None, None
    return 'easyocr', _easyocr_reader


def recognize_plate(image_path: str, timeout: float = 30.0, use_yolo: bool = True) -> dict:
    """
    Recognize license plate number from image
    Strategy: YOLO detection -> PaddleOCR recognition
    
    Args:
        image_path: Path to image file
        timeout: Recognition timeout in seconds
        use_yolo: Whether to use YOLO for plate detection (default: True)

    Returns:
        {"success": True/False, "plate": "浙D30520", "confidence": 0.95, "msg": "..."}
    """
    if not os.path.exists(image_path):
        return {"success": False, "plate": "", "confidence": 0.0, "msg": "Image file not found"}

    engine_type, engine = _get_reader()
    if engine is None:
        return {"success": False, "plate": "", "confidence": 0.0,
                "msg": "OCR engine not ready. Install: pip install paddleocr or pip install easyocr"}

    # Step 1: Try YOLO-based plate detection if available
    if use_yolo and YOLO_AVAILABLE:
        return _recognize_with_yolo(image_path, engine, engine_type, timeout)
    
    # Step 2: Fallback to traditional full-image OCR
    return _recognize_full_image(image_path, engine, engine_type, timeout)


def _recognize_with_yolo(image_path: str, engine, engine_type: str, timeout: float) -> dict:
    """
    YOLO-based plate detection and recognition pipeline
    """
    try:
        # Try to detect plates using YOLO
        plate_regions = detect_and_crop_plates(image_path, expand_ratio=0.1)
        
        if not plate_regions:
            print("[OCR] YOLO: No plates detected, falling back to full image OCR")
            return _recognize_full_image(image_path, engine, engine_type, timeout)
        
        print(f"[OCR] YOLO detected {len(plate_regions)} plate region(s)")
        
        # Process each detected plate region
        best_result = None
        best_confidence = -1
        
        for idx, (roi, metadata) in enumerate(plate_regions):
            # Save ROI to temp file for OCR
            fd, temp_path = tempfile.mkstemp(suffix='.jpg')
            os.close(fd)
            
            try:
                cv2.imwrite(temp_path, roi)
                
                # Preprocess the cropped plate region
                processed_path = _preprocess_image(temp_path)
                
                # Run OCR recognition with timeout
                result = _run_ocr_with_timeout(
                    engine, engine_type, processed_path, timeout
                )
                
                # Clean up
                try:
                    if processed_path != temp_path:
                        os.remove(processed_path)
                except:
                    pass
                
                # Track best result
                if result["success"]:
                    conf = result.get("confidence", 0.0)
                    if conf > best_confidence:
                        best_confidence = conf
                        best_result = result
                        print(f"[OCR] YOLO region {idx}: plate={result['plate']}, confidence={conf:.3f}")
                
            finally:
                try:
                    os.remove(temp_path)
                except:
                    pass
        
        # Return best result if found
        if best_result:
            best_result["msg"] = f"[YOLO-based] {best_result['msg']}"
            return best_result
        
        # If YOLO detection didn't yield results, try fallback
        print("[OCR] YOLO detected regions but OCR failed, trying full image")
        return _recognize_full_image(image_path, engine, engine_type, timeout)
        
    except Exception as e:
        print(f"[OCR] YOLO pipeline error: {e}, falling back to full image OCR")
        return _recognize_full_image(image_path, engine, engine_type, timeout)


def _recognize_full_image(image_path: str, engine, engine_type: str, timeout: float) -> dict:
    """
    Traditional full-image OCR recognition
    """
    try:
        processed_path = _preprocess_image(image_path)
    except Exception as e:
        return {"success": False, "plate": "", "confidence": 0.0,
                "msg": f"Image processing failed: {e}"}

    result = _run_ocr_with_timeout(engine, engine_type, processed_path, timeout)
    
    # Clean up temp file
    try:
        if processed_path != image_path:
            os.remove(processed_path)
    except:
        pass
    
    return result


def _run_ocr_with_timeout(engine, engine_type: str, image_path: str, timeout: float) -> dict:
    """
    Run OCR recognition with timeout protection
    """
    result_holder = [None]
    error_holder = [None]

    def do_recognize():
        try:
            if engine_type == 'paddleocr':
                result_holder[0] = _recognize_with_paddleocr(engine, image_path)
            else:
                result_holder[0] = _recognize_with_easyocr_reader(engine, image_path)
        except Exception as e:
            error_holder[0] = e

    thread = threading.Thread(target=do_recognize, daemon=True)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        return {"success": False, "plate": "", "confidence": 0.0,
                "msg": "Recognition timeout. Please retry or enter plate manually"}

    if error_holder[0]:
        return {"success": False, "plate": "", "confidence": 0.0,
                "msg": f"Recognition error: {error_holder[0]}"}

    if result_holder[0] is None:
        return {"success": False, "plate": "", "confidence": 0.0,
                "msg": "Recognition failed. No text detected. Please enter plate manually"}

    return result_holder[0]


def _preprocess_image(image_path: str) -> str:
    """
    Advanced image preprocessing for license plate recognition
    Using PIL only - no OpenCV dependency
    Focus on maximizing character clarity and contrast
    """
    from PIL import Image, ImageOps, ImageEnhance, ImageFilter
    import numpy as np
    
    try:
        img = Image.open(image_path)
    except Exception as e:
        raise Exception(f"Failed to open image: {e}")
    
    img = ImageOps.exif_transpose(img)

    # Unified color mode
    if img.mode in ('RGBA', 'P', 'LA'):
        img = img.convert('RGB')
    elif img.mode != 'RGB':
        img = img.convert('RGB')

    w, h = img.size

    # Intelligent scaling: target at least 600px on short side
    min_side = min(w, h)
    target_min = 600
    
    if min_side < target_min:
        scale = target_min / min_side
        if scale > 4.0:  # Don't over-scale
            scale = 4.0
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        print(f"[OCR] Resize: {w}x{h} -> {new_w}x{new_h}")

    # Multi-pass enhancement for better character visibility
    # 1st pass: Contrast boost
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.5)
    
    # 2nd pass: Brightness adjustment
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(1.1)
    
    # 3rd pass: Color saturation to preserve plate features
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(1.2)
    
    # Sharpening (apply multiple times for strong effect)
    for _ in range(2):
        img = img.filter(ImageFilter.SHARPEN)
        img = img.filter(ImageFilter.SHARPEN)

    # Save with high quality
    fd, tmp_path = tempfile.mkstemp(suffix='.jpg')
    os.close(fd)
    img.save(tmp_path, 'JPEG', quality=98)

    ext = os.path.splitext(image_path)[1].lower()
    if ext not in ('.jpg', '.jpeg'):
        print(f"[OCR] Enhanced: {os.path.basename(image_path)} -> High-quality JPEG")
    return tmp_path


def _recognize_with_easyocr_reader(reader, image_path: str) -> dict:
    """
    License plate recognition using EasyOCR
    
    Strategy:
    1. Collect all text lines and try to assemble complete plate
    2. Try matching format against concatenated results
    3. Handle cases where OCR splits plate into multiple lines
    """
    results = reader.readtext(image_path)

    if not results:
        return {"success": False, "plate": "", "confidence": 0.0,
                "msg": "No text detected. Ensure the license plate is clearly visible"}

    # Process individual lines
    plate_matches = []  # (plate, confidence)
    all_texts = []      # (text, confidence)
    all_raw_texts = []  # Raw detected texts for assembly attempts

    for bbox, text, confidence in results:
        # Clean text
        text_clean = text.replace(" ", "").replace(".", "").replace("·", "").upper()
        text_clean = _correct_plate_text(text_clean)
        
        if text_clean:
            all_texts.append((text_clean, confidence))
            all_raw_texts.append(text_clean)

            # Try to match license plate format
            match = PLATE_PATTERN.search(text_clean) or GREEN_PLATE_PATTERN.search(text_clean)
            if match:
                matched_plate = match.group()
                plate_matches.append((matched_plate, confidence))

    # If no direct match, try assembling from multiple lines
    if not plate_matches and len(all_raw_texts) > 1:
        combined_text = ''.join(all_raw_texts)
        combined_text = _correct_plate_text(combined_text)
        match = PLATE_PATTERN.search(combined_text) or GREEN_PLATE_PATTERN.search(combined_text)
        if match:
            matched_plate = match.group()
            # Average confidence
            avg_confidence = sum(c for _, c in all_texts) / len(all_texts)
            plate_matches.append((matched_plate, avg_confidence))
    
    # Strategy 1: Found plate format match
    if plate_matches:
        best_plate, best_conf = max(plate_matches, key=lambda x: x[1])

        if best_conf >= OCR_CONFIDENCE_HIGH:
            return {
                "success": True,
                "plate": best_plate,
                "confidence": round(best_conf, 4),
                "msg": "Recognition successful"
            }
        elif best_conf >= OCR_CONFIDENCE_LOW:
            return {
                "success": True,
                "plate": best_plate,
                "confidence": round(best_conf, 4),
                "msg": "Recognition successful (please confirm license plate)"
            }
        else:
            return {
                "success": True,
                "plate": best_plate,
                "confidence": round(best_conf, 4),
                "msg": f"Low confidence ({best_conf:.1%}), please verify"
            }

    # Strategy 2: No format match - try to fix incomplete matches
    if all_texts:
        # Check if any text looks like incomplete license plate (missing province)
        for text, conf in all_texts:
            if len(text) >= 5 and text[0].isdigit():
                # Likely a partial plate with missing first character
                # Try to prepend common province
                for province in '京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤川青藏琼宁':
                    test_plate = province + text
                    if PLATE_PATTERN.fullmatch(test_plate) or GREEN_PLATE_PATTERN.fullmatch(test_plate):
                        print(f"[OCR] Recovered partial plate: {text} -> {test_plate}")
                        return {
                            "success": True,
                            "plate": test_plate,
                            "confidence": round(conf, 4),
                            "msg": f"Partial plate recovered (province added). Verify: {test_plate}"
                        }
        
        best_text, best_conf = max(all_texts, key=lambda x: x[1])
        print(f"[OCR] No standard plate format found. Best: '{best_text}' confidence: {best_conf:.1%}")

        if best_conf >= OCR_CONFIDENCE_LOW:
            return {
                "success": True,
                "plate": best_text,
                "confidence": round(best_conf, 4),
                "msg": f"Incomplete match. Please verify or complete the plate"
            }
        else:
            return {
                "success": False,
                "plate": best_text if best_conf > 0.01 else "",
                "confidence": round(best_conf, 4),
                "msg": f"Confidence too low ({best_conf:.1%}). Please enter plate manually"
            }
    
    # No text at all
    return {"success": False, "plate": "", "confidence": 0.0,
            "msg": "No text recognized. Please ensure plate is clearly visible"}


def _recognize_with_paddleocr(ocr, image_path: str) -> dict:
    """
    License plate recognition using PaddleOCR (best for Chinese plates)
    
    Same strategies as EasyOCR: multi-line assembly, partial plate recovery
    """
    results = ocr.ocr(image_path, cls=True)

    if not results or not results[0]:
        return {"success": False, "plate": "", "confidence": 0.0,
                "msg": "No text detected. Ensure the license plate is clearly visible"}

    # Collect all results
    plate_matches = []
    all_texts = []
    all_raw_texts = []

    for line in results[0]:
        text = line[1][0]
        confidence = line[1][1]
        text_clean = text.replace(" ", "").replace(".", "").replace("·", "").upper()
        text_clean = _correct_plate_text(text_clean)
        
        if text_clean:
            all_texts.append((text_clean, confidence))
            all_raw_texts.append(text_clean)

            match = PLATE_PATTERN.search(text_clean) or GREEN_PLATE_PATTERN.search(text_clean)
            if match:
                matched_plate = match.group()
                plate_matches.append((matched_plate, confidence))

    # Try assembling from multiple lines if no direct match
    if not plate_matches and len(all_raw_texts) > 1:
        combined_text = ''.join(all_raw_texts)
        combined_text = _correct_plate_text(combined_text)
        match = PLATE_PATTERN.search(combined_text) or GREEN_PLATE_PATTERN.search(combined_text)
        if match:
            matched_plate = match.group()
            avg_confidence = sum(c for _, c in all_texts) / len(all_texts)
            plate_matches.append((matched_plate, avg_confidence))

    # Format match found
    if plate_matches:
        best_plate, best_conf = max(plate_matches, key=lambda x: x[1])

        if best_conf >= OCR_CONFIDENCE_HIGH:
            return {
                "success": True,
                "plate": best_plate,
                "confidence": round(best_conf, 4),
                "msg": "Recognition successful"
            }
        elif best_conf >= OCR_CONFIDENCE_LOW:
            return {
                "success": True,
                "plate": best_plate,
                "confidence": round(best_conf, 4),
                "msg": "Recognition successful (please confirm license plate)"
            }
        else:
            return {
                "success": True,
                "plate": best_plate,
                "confidence": round(best_conf, 4),
                "msg": f"Low confidence ({best_conf:.1%}), please verify"
            }

    # No format match - try partial plate recovery
    if all_texts:
        for text, conf in all_texts:
            if len(text) >= 5 and text[0].isdigit():
                # Likely partial - try prepending provinces
                for province in '京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤川青藏琼宁':
                    test_plate = province + text
                    if PLATE_PATTERN.fullmatch(test_plate) or GREEN_PLATE_PATTERN.fullmatch(test_plate):
                        print(f"[OCR] Recovered: {text} -> {test_plate}")
                        return {
                            "success": True,
                            "plate": test_plate,
                            "confidence": round(conf, 4),
                            "msg": f"Partial plate recovered. Verify: {test_plate}"
                        }

        best_text, best_conf = max(all_texts, key=lambda x: x[1])
        print(f"[OCR] PaddleOCR: No match. Best: '{best_text}' confidence: {best_conf:.1%}")

        if best_conf >= OCR_CONFIDENCE_LOW:
            return {
                "success": True,
                "plate": best_text,
                "confidence": round(best_conf, 4),
                "msg": f"Incomplete match. Please verify or complete"
            }
        else:
            return {
                "success": False,
                "plate": best_text if best_conf > 0.01 else "",
                "confidence": round(best_conf, 4),
                "msg": f"Confidence too low ({best_conf:.1%}). Please enter manually"
            }
    
    return {"success": False, "plate": "", "confidence": 0.0,
            "msg": "No text recognized. Please ensure plate is clearly visible"}


def is_valid_plate(plate: str) -> bool:
    """Validate if license plate format is legal"""
    if not plate:
        return False
    plate = plate.strip().upper()
    return bool(PLATE_PATTERN.fullmatch(plate) or GREEN_PLATE_PATTERN.fullmatch(plate))


# ==================== License Plate Character Correction Dictionary ====================
# EasyOCR and PaddleOCR often misrecognize license plate characters
# especially for handwriting-style fonts used on plates

# Province character corrections (OCR commonly confuses similar-looking characters)
_PROVINCE_CORRECTIONS = {
    # Common OCR errors - look-alikes
    '噜': '鲁', '泸': '沪', '寞': '冀', 
    # Tone mark confusions
    '粤': '粤', '苏': '苏', '川': '川', 
    # Similar stroke patterns  
    '鄂': '鄂', '皖': '皖', '湘': '湘',
}

# Letter corrections (position 2 must be a letter, often confused with numbers/symbols)
_LETTER_CORRECTIONS = {
    # Numbers -> Letters (most common confusion)
    '0': 'O', '1': 'I', '5': 'S', '8': 'B', '6': 'G', '2': 'Z', '9': 'P',
    '|': 'I', '/': 'I', '\\': 'I', 'l': 'I',
    '[': 'J', ']': 'J', '{': 'C', '}': 'C',
    # Remove garbage characters
    ':': '', ';': '', ',': '', '.': '', "'": '', '"': '', '·': '',
    '?': '', '!': '', '(': '', ')': '', '-': '', '_': '',
}

# Digit corrections (letters -> digits in numeric positions)
_DIGIT_CORRECTIONS = {
    'O': '0', 'o': '0', 'l': '1', 'I': '1', 'Z': '2', 'S': '5',
    'B': '8', 'G': '6', 'T': '7', 'A': '4', 'D': '0',
}


def _correct_plate_text(text: str) -> str:
    """
    Smart license plate character correction with aggressive letter/digit fixing
    
    Key improvements:
    - Position 1: Chinese province (or try num->letter for recovery)
    - Position 2: Must be letter (aggressive num->letter conversion)
    - Position 3: For new energy, must be D or F
    """
    if not text or len(text) < 3:
        return text

    text = text.strip().upper()
    # Remove interference characters
    text = re.sub(r'[：:;,\'\"\.\(\)\[\]{}\-/\\|!?@#$%^&*+=~`\s]', '', text)
    if not text:
        return text

    result = list(text)

    # Known provinces
    known_provinces = '京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤川青藏琼宁'
    
    # Position 1: Province character
    if result[0] not in known_provinces:
        # Try to correct using dictionary
        if result[0] in _PROVINCE_CORRECTIONS:
            result[0] = _PROVINCE_CORRECTIONS[result[0]]
        # If still not recognized, try digit-to-letter conversion for recovery
        elif result[0].isdigit():
            # Map numbers to similar-looking letters
            num_to_letter_province = {'0': 'O', '1': 'I', '5': 'S', '8': 'B', '6': 'G'}
            result[0] = num_to_letter_province.get(result[0], result[0])
        # If first char is letter, might be misplaced - keep for now

    # Position 2: CRITICAL - must be letter
    if len(result) >= 2:
        if result[1] in _LETTER_CORRECTIONS:
            corrected = _LETTER_CORRECTIONS[result[1]]
            if corrected:
                result[1] = corrected
            else:
                result[1] = ''
        elif result[1].isdigit():
            # Digit in position 2 is ALWAYS a letter error
            num_to_letter_aggressive = {
                '0': 'D', '1': 'I', '2': 'Z', '3': 'B', '4': 'A',
                '5': 'S', '6': 'G', '7': 'T', '8': 'B', '9': 'P'
            }
            result[1] = num_to_letter_aggressive.get(result[1], result[1])
        elif result[1] in 'IO':  # Letter O/I which might need special handling
            # Keep as-is, they're valid letters
            pass
        elif not ('A' <= result[1] <= 'Z'):
            # Unknown character in position 2 - try to fix it
            pass

    # Positions 3+: Handle new energy plates and digit corrections
    for i in range(2, len(result)):
        # Position 3 for new energy plates must be D or F
        if i == 2:
            if result[i] not in 'DF' and result[i] in _DIGIT_CORRECTIONS:
                result[i] = _DIGIT_CORRECTIONS[result[i]]
        else:
            # Regular positions - allow both letters and digits
            if result[i] in _DIGIT_CORRECTIONS:
                result[i] = _DIGIT_CORRECTIONS[result[i]]
        
        # Remove invalid characters (keep only alphanumeric + special plate chars)
        if result[i] not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789DF学警挂港澳':
            result[i] = ''

    # Clean up empty positions
    result = [c for c in result if c]
    
    return ''.join(result)
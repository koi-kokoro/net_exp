"""
Web服务器 — 停车场出入场系统手机端
使用 Flask 提供 Web 服务，手机浏览器打开即可使用
功能：
- 手机拍照/选图上传
- OCR车牌识别
- 入场/出场操作
- 车位状态查询
- 停车记录查询与统计
"""

import os
import sys
import time
import json
from datetime import datetime

# 确保能导入同目录模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import (
    Flask, render_template, request, jsonify, send_from_directory, session
)
from werkzeug.utils import secure_filename

from config import (
    SERVER_HOST, MAX_IMAGE_SIZE, IMAGE_SAVE_DIR, TOTAL_PARKING_SPACES,
)
from database import (
    init_db, register_entry, register_exit, get_occupied_spaces,
    query_records, get_statistics,
)
from ocr_engine_new import recognize_plate, is_valid_plate, init_ocr_engine
#from ocr_engine_lite import recognize_plate_lite, is_valid_plate_lite, init_ocr_engine_lite

app = Flask(__name__)
app.secret_key = os.urandom(24).hex()
app.config['MAX_CONTENT_LENGTH'] = MAX_IMAGE_SIZE

# 上传临时目录
UPLOAD_FOLDER = os.path.join(IMAGE_SAVE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 日志
def log(msg: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")


# ==================== 页面路由 ====================

@app.route('/')
def index():
    """首页 — 移动端操作界面"""
    return render_template('index.html')


# ==================== API 路由 ====================

@app.route('/api/status')
def api_status():
    """获取停车场状态"""
    occupied = get_occupied_spaces()
    return jsonify({
        'code': 200,
        'occupied': occupied,
        'total': TOTAL_PARKING_SPACES,
        'available': TOTAL_PARKING_SPACES - occupied,
    })


@app.route('/api/upload', methods=['POST'])
def api_upload():
    """上传图片并OCR识别"""
    if 'image' not in request.files:
        return jsonify({'code': 400, 'msg': '未收到图片'})

    file = request.files['image']
    if file.filename == '':
        return jsonify({'code': 400, 'msg': '未选择文件'})

    # 保存图片
    ext = os.path.splitext(file.filename)[1] or '.jpg'
    filename = f"upload_{int(time.time() * 1000)}{ext}"
    filepath = os.path.join(UPLOAD_FOLDER, secure_filename(filename))
    file.save(filepath)

    log(f"图片已上传: {filepath}")

    # OCR识别
    ocr_result = recognize_plate(filepath)
    log(f"OCR识别结果: plate={ocr_result.get('plate', '?')} success={ocr_result['success']} msg={ocr_result['msg']}")

    if ocr_result['success']:
        return jsonify({
            'code': 200,
            'msg': ocr_result['msg'],
            'plate': ocr_result['plate'],
            'confidence': ocr_result['confidence'],
            'image_path': filename,
        })
    else:
        return jsonify({
            'code': 422,
            'msg': ocr_result['msg'],
            'confidence': ocr_result.get('confidence', 0),
            'image_path': filename,
        })


@app.route('/api/entry', methods=['POST'])
def api_entry():
    """车辆入场"""
    data = request.get_json()
    if not data:
        return jsonify({'code': 400, 'msg': '请求数据为空'})

    plate = data.get('plate', '').strip().upper()
    if not is_valid_plate(plate):
        return jsonify({'code': 400, 'msg': f'车牌号格式不正确: {plate}'})

    # 检查车位
    occupied = get_occupied_spaces()
    if occupied >= TOTAL_PARKING_SPACES:
        return jsonify({'code': 503, 'msg': '车位已满，禁止入场'})

    result = register_entry(plate)
    if result['success']:
        log(f"入场登记: {plate}  时间: {result['entry_time']}  占用: {occupied + 1}/{TOTAL_PARKING_SPACES}")
        return jsonify({'code': 200, **result})
    else:
        return jsonify({'code': 409, **result})


@app.route('/api/exit', methods=['POST'])
def api_exit():
    """车辆出场"""
    data = request.get_json()
    if not data:
        return jsonify({'code': 400, 'msg': '请求数据为空'})

    plate = data.get('plate', '').strip().upper()
    result = register_exit(plate)
    if result['success']:
        log(f"出场计费: {plate}  费用: ¥{result['fee']}  时长: {result['duration_minutes']}分钟")
        return jsonify({'code': 200, **result})
    else:
        return jsonify({'code': 404, **result})


@app.route('/api/records')
def api_records():
    """查询停车记录"""
    plate = request.args.get('plate', '')
    records = query_records(plate_number=plate if plate else None)
    return jsonify({
        'code': 200,
        'records': records,
        'count': len(records),
    })


@app.route('/api/stats')
def api_stats():
    """获取统计数据"""
    date_str = request.args.get('date', '')
    stats = get_statistics(date_str if date_str else None)
    return jsonify({
        'code': 200,
        'statistics': stats,
    })


# ==================== 启动入口 ====================

def main():
    init_db()
    print("=" * 50)
    print("  停车场出入场系统 — Web服务器")
    print("=" * 50)
    print()

    # 预加载 OCR 模型（首次会下载，约1-2分钟）
    print("  正在初始化 OCR 引擎...")
    init_ocr_engine()
    print()

    print("  📱 手机端访问方式：")
    print("     1. 确保手机和电脑在同一局域网")
    print("     2. 查看下方「本机IP地址」")
    print("     3. 手机浏览器打开: http://<本机IP>:5000")
    print()
    
    # 显示本机IP
    import socket as sock
    hostname = sock.gethostname()
    try:
        local_ip = sock.gethostbyname(hostname)
        print(f"  本机IP: {local_ip}")
        print(f"  访问地址: http://{local_ip}:5000")
    except:
        print("  无法获取本机IP，请手动查看网络设置")
    
    print()
    print("  提示: 按 Ctrl+C 停止服务器")
    print("=" * 50)
    print()
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)


if __name__ == '__main__':
    main()

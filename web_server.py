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
    Flask, render_template, request, jsonify, send_from_directory, session, Response
)
from werkzeug.utils import secure_filename

from config import (
    SERVER_HOST, MAX_IMAGE_SIZE, IMAGE_SAVE_DIR, TOTAL_PARKING_SPACES,
)
from database import (
    init_db, register_entry, register_exit, get_occupied_spaces,
    query_records, get_statistics, get_connection,
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


# ==================== 管理员后台路由 ====================

@app.route('/admin')
def admin_panel():
    """管理员后台页面"""
    return render_template('admin.html')


@app.route('/api/admin/config', methods=['GET', 'POST'])
def admin_config():
    """管理员：查看/修改计费配置"""
    if request.method == 'GET':
        # 从 config 模块和数据库读取当前配置
        from config import (
            DEFAULT_PRICE_PER_HOUR, DAILY_CAP, FREE_MINUTES,
            NIGHT_DISCOUNT_START, NIGHT_DISCOUNT_END, NIGHT_DISCOUNT_RATE
        )
        return jsonify({
            'code': 200,
            'price_per_hour': DEFAULT_PRICE_PER_HOUR,
            'daily_cap': DAILY_CAP,
            'free_minutes': FREE_MINUTES,
            'night_discount_start': NIGHT_DISCOUNT_START,
            'night_discount_end': NIGHT_DISCOUNT_END,
            'night_discount_rate': NIGHT_DISCOUNT_RATE,
        })
    else:
        # 保存配置到数据库 fee_rules 表
        data = request.get_json()
        if not data:
            return jsonify({'code': 400, 'msg': '数据为空'})
        try:
            conn = get_connection()
            conn.execute("""
                UPDATE fee_rules SET 
                    price_per_hour=?, daily_cap=?, free_minutes=?
                WHERE id=1
            """, (
                data.get('price_per_hour', 5.0),
                data.get('daily_cap', 30.0),
                data.get('free_minutes', 15),
            ))
            conn.commit()
            conn.close()
            log(f"计费配置已更新: {data}")
            # 同时更新 config 模块变量（运行时生效）
            import config
            config.DEFAULT_PRICE_PER_HOUR = data.get('price_per_hour', 5.0)
            config.DAILY_CAP = data.get('daily_cap', 30.0)
            config.FREE_MINUTES = data.get('free_minutes', 15)
            config.NIGHT_DISCOUNT_START = data.get('night_discount_start', 20)
            config.NIGHT_DISCOUNT_END = data.get('night_discount_end', 8)
            config.NIGHT_DISCOUNT_RATE = data.get('night_discount_rate', 0.8)
            return jsonify({'code': 200, 'msg': '配置保存成功'})
        except Exception as e:
            return jsonify({'code': 500, 'msg': f'保存失败: {e}'})


@app.route('/api/admin/export')
def admin_export():
    """管理员：导出停车记录为CSV"""
    import csv
    import io
    
    records = query_records()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['ID', '车牌号', '入场时间', '出场时间', '停车时长(分钟)', '费用(元)', '状态'])
    
    for r in records:
        status_text = '在场' if r.get('status') == 'in' else '已离场'
        writer.writerow([
            r.get('id', ''),
            r.get('plate_number', ''),
            r.get('entry_time', ''),
            r.get('exit_time', '') or '',
            r.get('duration_minutes', '') or '',
            r.get('fee', '') or '',
            status_text,
        ])
    
    output.seek(0)
    return Response(
        output.getvalue().encode('utf-8-sig'),
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=parking_records.csv'}
    )


@app.route('/api/admin/image-stats')
def admin_image_stats():
    """管理员：获取图片统计信息"""
    import time as time_module
    
    images_dir = os.path.join(IMAGE_SAVE_DIR)
    uploads_dir = os.path.join(IMAGE_SAVE_DIR, 'uploads')
    
    total_count = 0
    total_size = 0
    old_count = 0
    old_size = 0
    cutoff = time_module.time() - 7 * 24 * 3600  # 7天前
    
    for dir_path in [images_dir, uploads_dir]:
        if not os.path.exists(dir_path):
            continue
        for fname in os.listdir(dir_path):
            fpath = os.path.join(dir_path, fname)
            if os.path.isfile(fpath) and fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                size = os.path.getsize(fpath)
                total_count += 1
                total_size += size
                if os.path.getmtime(fpath) < cutoff:
                    old_count += 1
                    old_size += size
    
    return jsonify({
        'code': 200,
        'total_count': total_count,
        'total_size': total_size,
        'old_count': old_count,
        'old_size': old_size,
    })


@app.route('/api/admin/cleanup', methods=['POST'])
def admin_cleanup():
    """管理员：清理7天前的旧图片"""
    import time as time_module
    
    images_dir = os.path.join(IMAGE_SAVE_DIR)
    uploads_dir = os.path.join(IMAGE_SAVE_DIR, 'uploads')
    cutoff = time_module.time() - 7 * 24 * 3600
    deleted = 0
    freed_bytes = 0
    
    for dir_path in [images_dir, uploads_dir]:
        if not os.path.exists(dir_path):
            continue
        for fname in os.listdir(dir_path):
            fpath = os.path.join(dir_path, fname)
            if os.path.isfile(fpath) and fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                if os.path.getmtime(fpath) < cutoff:
                    try:
                        size = os.path.getsize(fpath)
                        os.remove(fpath)
                        deleted += 1
                        freed_bytes += size
                    except Exception as e:
                        log(f"清理图片失败: {fpath} - {e}")
    
    log(f"图片清理完成: 删除 {deleted} 张, 释放 {freed_bytes} 字节")
    return jsonify({
        'code': 200,
        'msg': f'清理完成，删除 {deleted} 张图片',
        'deleted': deleted,
        'freed_bytes': freed_bytes,
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

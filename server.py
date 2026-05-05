"""
TCP服务器 — 停车场出入场系统
架构：主线程监听 + 每客户端一个处理线程
特性：
- 自定义协议粘包处理
- OCR车牌识别
- 多客户端并发支持
- 完整的入场/出场/计费逻辑
- 车位管理
- 日志系统
"""

import socket
import threading
import json
import os
import time
from datetime import datetime

from config import (
    SERVER_HOST, SERVER_PORT, BUFFER_SIZE, MAX_IMAGE_SIZE,
    IMAGE_SAVE_DIR, TOTAL_PARKING_SPACES,
    MSG_TYPE_JSON_META, MSG_TYPE_IMAGE_DATA, MSG_TYPE_JSON_RESP, MSG_TYPE_JSON_BILL,
)
from protocol import ProtocolHandler, pack_json
from database import init_db, register_entry, register_exit, get_occupied_spaces, query_records, get_statistics
from ocr_engine import recognize_plate, is_valid_plate, init_ocr_engine

# 尝试导入YOLO引擎
try:
    from yolo_engine import init_yolo_model
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# 日志锁
_log_lock = threading.Lock()


def log(msg: str):
    """线程安全的日志输出"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with _log_lock:
        print(f"[{timestamp}] {msg}")


class ParkingServer:
    """停车场服务器"""

    def __init__(self, host: str = SERVER_HOST, port: int = SERVER_PORT):
        self.host = host
        self.port = port
        self.server_socket = None
        self.running = False
        self._clients = []  # 保存客户端线程引用

    def start(self):
        """启动服务器"""
        init_db()
        log("数据库初始化完成")

        # 预加载 OCR 模型
        log("正在初始化 OCR 引擎...")
        init_ocr_engine()

        # 预加载 YOLO 模型
        if YOLO_AVAILABLE:
            log("正在初始化 YOLO 引擎...")
            if init_yolo_model():
                log("YOLO 模型加载成功，将使用 YOLO 进行车牌检测")
            else:
                log("YOLO 初始化失败，将使用传统 OCR 方法")
        else:
            log("YOLO 未安装，使用传统 OCR 方法识别车牌")

        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(50)
        # 设置超时，使 accept() 不永久阻塞，Ctrl+C 能够被及时响应
        self.server_socket.settimeout(1.0)
        self.running = True

        log(f"服务器启动: {self.host}:{self.port}")
        log(f"总车位数: {TOTAL_PARKING_SPACES}")
        log("提示: 按 Ctrl+C 可安全退出服务器")

        try:
            while self.running:
                try:
                    client_sock, client_addr = self.server_socket.accept()
                    log(f"新客户端连接: {client_addr[0]}:{client_addr[1]}")

                    client_thread = threading.Thread(
                        target=self._handle_client,
                        args=(client_sock, client_addr),
                        daemon=True
                    )
                    client_thread.start()
                    self._clients.append(client_thread)
                except socket.timeout:
                    # 超时后回到循环顶部检查 self.running，同时允许 KeyboardInterrupt 被捕获
                    continue
                except OSError:
                    break
        except KeyboardInterrupt:
            print()  # 换行
            log("收到中断信号(Ctrl+C)，正在关闭服务器...")
        finally:
            self.stop()

    def stop(self):
        """关闭服务器"""
        self.running = False
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        log("服务器已关闭")

    def _handle_client(self, client_sock: socket.socket, client_addr: tuple):
        """处理单个客户端连接"""
        client_ip = client_addr[0]
        handler = ProtocolHandler()

        try:
            while self.running:
                data = client_sock.recv(BUFFER_SIZE)
                if not data:
                    log(f"客户端断开: {client_ip}")
                    break

                # 喂入协议处理器，解析出完整消息
                messages = handler.feed(data)

                for msg_type, body, checksum_ok in messages:
                    if not checksum_ok:
                        log(f"[{client_ip}] 校验和不匹配，丢弃消息")
                        resp = pack_json(MSG_TYPE_JSON_RESP, {"code": 400, "msg": "数据校验失败，请重传"})
                        self._safe_send(client_sock, resp)
                        continue

                    # 根据消息类型分发处理
                    response = self._dispatch(client_ip, msg_type, body)
                    if response:
                        self._safe_send(client_sock, response)

        except ConnectionResetError:
            log(f"客户端异常断开: {client_ip}")
        except Exception as e:
            log(f"处理客户端 {client_ip} 时出错: {e}")
        finally:
            try:
                client_sock.close()
            except:
                pass

    def _dispatch(self, client_ip: str, msg_type: int, body: bytes) -> bytes:
        """消息分发器"""
        if msg_type == MSG_TYPE_JSON_META:
            return self._handle_json_meta(client_ip, body)
        elif msg_type == MSG_TYPE_IMAGE_DATA:
            return self._handle_image(client_ip, body)
        else:
            log(f"[{client_ip}] 未知消息类型: {msg_type:#04x}")
            return pack_json(MSG_TYPE_JSON_RESP, {"code": 400, "msg": f"未知消息类型: {msg_type:#04x}"})

    def _handle_json_meta(self, client_ip: str, body: bytes) -> bytes:
        """处理JSON元数据消息"""
        try:
            meta = json.loads(body.decode("utf-8"))
            cmd = meta.get("cmd", "")

            if cmd == "entry_query":
                # 入场前查询：告知客户端当前状态
                occupied = get_occupied_spaces()
                available = TOTAL_PARKING_SPACES - occupied
                return pack_json(MSG_TYPE_JSON_RESP, {
                    "code": 200,
                    "msg": f"当前车位: {available}/{TOTAL_PARKING_SPACES}",
                    "available": available,
                    "total": TOTAL_PARKING_SPACES,
                    "ready_for_image": True  # 服务器准备接收图片
                })

            elif cmd == "exit_query":
                return pack_json(MSG_TYPE_JSON_RESP, {
                    "code": 200,
                    "msg": "准备接收出场图片",
                    "ready_for_image": True
                })

            elif cmd == "manual_entry":
                # 手动输入车牌入场
                plate = meta.get("plate", "").strip().upper()
                if not is_valid_plate(plate):
                    return pack_json(MSG_TYPE_JSON_RESP, {"code": 400, "msg": f"车牌号格式不正确: {plate}"})

                occupied = get_occupied_spaces()
                if occupied >= TOTAL_PARKING_SPACES:
                    return pack_json(MSG_TYPE_JSON_RESP, {"code": 503, "msg": "车位已满，禁止入场"})

                result = register_entry(plate)
                if result["success"]:
                    log(f"[{client_ip}] 入场登记(手动): {plate}  时间: {result['entry_time']}  占用: {occupied + 1}/{TOTAL_PARKING_SPACES}")
                    return pack_json(MSG_TYPE_JSON_RESP, {"code": 200, **result})
                else:
                    return pack_json(MSG_TYPE_JSON_RESP, {"code": 409, **result})

            elif cmd == "manual_exit":
                plate = meta.get("plate", "").strip().upper()
                result = register_exit(plate)
                if result["success"]:
                    log(f"[{client_ip}] 出场计费(手动): {plate}  费用: ¥{result['fee']}  时长: {result['duration_minutes']}分钟")
                    return pack_json(MSG_TYPE_JSON_BILL, {"code": 200, **result})
                else:
                    return pack_json(MSG_TYPE_JSON_RESP, {"code": 404, **result})

            elif cmd == "query":
                plate = meta.get("plate", "")
                records = query_records(plate_number=plate)
                return pack_json(MSG_TYPE_JSON_RESP, {
                    "code": 200,
                    "msg": f"查询到 {len(records)} 条记录",
                    "records": records
                })

            elif cmd == "stats":
                date_str = meta.get("date", "")
                stats = get_statistics(date_str if date_str else None)
                return pack_json(MSG_TYPE_JSON_RESP, {
                    "code": 200,
                    "msg": "统计数据",
                    "statistics": stats
                })

            else:
                return pack_json(MSG_TYPE_JSON_RESP, {"code": 400, "msg": f"未知命令: {cmd}"})

        except json.JSONDecodeError:
            return pack_json(MSG_TYPE_JSON_RESP, {"code": 400, "msg": "JSON解析失败"})

    def _handle_image(self, client_ip: str, body: bytes) -> bytes:
        """处理图片上传"""
        if len(body) > MAX_IMAGE_SIZE:
            return pack_json(MSG_TYPE_JSON_RESP, {"code": 413, "msg": f"图片过大，最大 {MAX_IMAGE_SIZE // 1024 // 1024}MB"})

        # 保存图片
        timestamp = int(time.time() * 1000)
        filename = f"{client_ip.replace('.', '_')}_{timestamp}.jpg"
        image_path = os.path.join(IMAGE_SAVE_DIR, filename)

        try:
            with open(image_path, "wb") as f:
                f.write(body)
            log(f"[{client_ip}] 图片已保存: {image_path} ({len(body)} bytes)")
        except Exception as e:
            return pack_json(MSG_TYPE_JSON_RESP, {"code": 500, "msg": f"图片保存失败: {e}"})

        # OCR识别
        ocr_result = recognize_plate(image_path)
        log(f"[{client_ip}] OCR识别: {ocr_result}")

        # 识别成功后返回结果，让客户端确认入场还是出场
        if ocr_result["success"]:
            return pack_json(MSG_TYPE_JSON_RESP, {
                "code": 200,
                "msg": ocr_result["msg"],
                "plate": ocr_result["plate"],
                "confidence": ocr_result["confidence"],
                "image_path": filename,
                "need_confirm": True  # 需要客户端确认入场/出场操作
            })
        else:
            return pack_json(MSG_TYPE_JSON_RESP, {
                "code": 422,
                "msg": ocr_result["msg"] + "，请手动输入车牌号",
                "confidence": ocr_result["confidence"],
                "image_path": filename,
                "need_manual_input": True
            })

    def _safe_send(self, sock: socket.socket, data: bytes):
        """安全发送数据"""
        try:
            sock.sendall(data)
        except Exception as e:
            log(f"发送数据失败: {e}")


def main():
    server = ParkingServer()
    try:
        server.start()
    except KeyboardInterrupt:
        print("\n服务器已停止")


if __name__ == "__main__":
    main()

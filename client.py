"""
客户端 — 停车场出入场系统
支持两种模式：
  - GUI模式：Tkinter图形界面（需要 tkinter 模块）
  - CLI模式：命令行交互界面（无需额外模块，自动降级）
功能：
- 连接服务器
- 选择/拍摄车牌图片（CLI模式下输入图片路径或使用示例图片）
- 上传并识别
- 入场/出场操作
- 车位状态查询
- 停车记录查询
"""

import socket
import threading
import json
import os
import sys
import time

from config import (
    SERVER_HOST, SERVER_PORT, BUFFER_SIZE,
    MSG_TYPE_JSON_META, MSG_TYPE_IMAGE_DATA, MSG_TYPE_JSON_RESP, MSG_TYPE_JSON_BILL,
)
from protocol import ProtocolHandler, pack_message, pack_json
from ocr_engine import is_valid_plate

# 尝试导入 GUI 依赖
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox, scrolledtext
    from PIL import Image, ImageTk
    HAS_GUI = True
except (ImportError, ModuleNotFoundError):
    HAS_GUI = False


class ParkingClient:
    """停车场客户端 GUI"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("停车场出入场系统 - 客户端")
        self.root.geometry("700x750")
        self.root.resizable(True, True)

        # 网络状态
        self.sock: socket.socket = None
        self.handler = ProtocolHandler()
        self.connected = False
        self.reconnect_enabled = tk.BooleanVar(value=True)

        # 当前操作
        self.current_image_path: str = None
        self.current_plate: str = None
        self.pending_action: str = None  # "entry" or "exit"

        # 构建界面
        self._build_ui()

        # 启动接收线程
        self._recv_thread = None
        self._running = False

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        """构建图形界面"""
        # 顶部标题
        title_frame = tk.Frame(self.root, bg="#2c3e50", height=50)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)
        tk.Label(
            title_frame, text="🅿️ 停车场出入场管理系统",
            font=("Microsoft YaHei", 16, "bold"),
            fg="white", bg="#2c3e50"
        ).pack(expand=True)

        # 连接区域
        conn_frame = tk.LabelFrame(self.root, text="服务器连接", font=("Microsoft YaHei", 10))
        conn_frame.pack(fill=tk.X, padx=10, pady=5)

        row1 = tk.Frame(conn_frame)
        row1.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(row1, text="主机:", font=("Microsoft YaHei", 9)).pack(side=tk.LEFT)
        self.host_entry = tk.Entry(row1, width=15, font=("Consolas", 10))
        self.host_entry.insert(0, SERVER_HOST)
        self.host_entry.pack(side=tk.LEFT, padx=5)
        tk.Label(row1, text="端口:", font=("Microsoft YaHei", 9)).pack(side=tk.LEFT)
        self.port_entry = tk.Entry(row1, width=8, font=("Consolas", 10))
        self.port_entry.insert(0, str(SERVER_PORT))
        self.port_entry.pack(side=tk.LEFT, padx=5)

        self.connect_btn = tk.Button(row1, text="🔗 连接", command=self._toggle_connect,
                                     bg="#3498db", fg="white", font=("Microsoft YaHei", 9),
                                     width=8)
        self.connect_btn.pack(side=tk.LEFT, padx=5)

        self.status_label = tk.Label(row1, text="⚫ 未连接", fg="red", font=("Microsoft YaHei", 9))
        self.status_label.pack(side=tk.LEFT, padx=10)

        tk.Checkbutton(row1, text="断线重连", variable=self.reconnect_enabled,
                       font=("Microsoft YaHei", 8)).pack(side=tk.RIGHT)

        # 车位信息
        self.parking_info_label = tk.Label(conn_frame, text="车位: --/--", fg="#7f8c8d",
                                           font=("Microsoft YaHei", 9))
        self.parking_info_label.pack(anchor=tk.W, padx=10, pady=(0, 5))

        # 操作区域 notebook
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Tab 1: 出入场操作
        op_frame = tk.Frame(notebook)
        notebook.add(op_frame, text="🚗 出入场操作")
        self._build_operation_tab(op_frame)

        # Tab 2: 记录查询
        query_frame = tk.Frame(notebook)
        notebook.add(query_frame, text="📋 记录查询")
        self._build_query_tab(query_frame)

        # Tab 3: 日志
        log_frame = tk.Frame(notebook)
        notebook.add(log_frame, text="📜 日志")
        self._build_log_tab(log_frame)

    def _build_operation_tab(self, parent):
        """构建出入场操作标签页"""
        # 选择图片区域
        img_frame = tk.LabelFrame(parent, text="选择车牌图片", font=("Microsoft YaHei", 10))
        img_frame.pack(fill=tk.X, padx=10, pady=5)

        btn_row = tk.Frame(img_frame)
        btn_row.pack(pady=10)
        tk.Button(btn_row, text="📷 选择图片", command=self._select_image,
                  bg="#27ae60", fg="white", font=("Microsoft YaHei", 10),
                  width=15, height=2).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_row, text="🖼️ 使用示例图片", command=self._use_sample_image,
                  bg="#8e44ad", fg="white", font=("Microsoft YaHei", 10),
                  width=15, height=2).pack(side=tk.LEFT, padx=5)

        # 图片预览
        self.image_preview_label = tk.Label(img_frame, text="尚未选择图片", bg="#ecf0f1",
                                            font=("Microsoft YaHei", 9), width=50, height=10)
        self.image_preview_label.pack(pady=5)

        self.image_path_label = tk.Label(img_frame, text="", fg="#7f8c8d", font=("Consolas", 8))
        self.image_path_label.pack(pady=(0, 5))

        # 识别结果
        result_frame = tk.LabelFrame(parent, text="识别与操作", font=("Microsoft YaHei", 10))
        result_frame.pack(fill=tk.X, padx=10, pady=5)

        r1 = tk.Frame(result_frame)
        r1.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(r1, text="车牌号:", font=("Microsoft YaHei", 10)).pack(side=tk.LEFT)
        self.plate_var = tk.StringVar()
        self.plate_entry = tk.Entry(r1, textvariable=self.plate_var, font=("Consolas", 14, "bold"),
                                    width=12, justify="center", state="readonly")
        self.plate_entry.pack(side=tk.LEFT, padx=5)
        tk.Label(r1, text="置信度:", font=("Microsoft YaHei", 9)).pack(side=tk.LEFT, padx=(10, 0))
        self.confidence_label = tk.Label(r1, text="--", font=("Consolas", 10), fg="#2980b9")
        self.confidence_label.pack(side=tk.LEFT, padx=5)

        # 手动输入（OCR失败时）
        self.manual_frame = tk.Frame(result_frame)
        tk.Label(self.manual_frame, text="手动输入:", font=("Microsoft YaHei", 9)).pack(side=tk.LEFT)
        self.manual_plate_entry = tk.Entry(self.manual_frame, font=("Consolas", 12), width=12)
        self.manual_plate_entry.pack(side=tk.LEFT, padx=5)
        tk.Button(self.manual_frame, text="确认", command=self._manual_confirm,
                  bg="#e67e22", fg="white", font=("Microsoft YaHei", 9)).pack(side=tk.LEFT, padx=5)

        # 操作按钮
        btn_row2 = tk.Frame(result_frame)
        btn_row2.pack(pady=10)
        self.entry_btn = tk.Button(btn_row2, text="🅿️ 入场", command=lambda: self._do_action("entry"),
                                   bg="#2980b9", fg="white", font=("Microsoft YaHei", 11),
                                   width=12, height=2, state="disabled")
        self.entry_btn.pack(side=tk.LEFT, padx=10)

        self.exit_btn = tk.Button(btn_row2, text="🚙 出场", command=lambda: self._do_action("exit"),
                                  bg="#c0392b", fg="white", font=("Microsoft YaHei", 11),
                                  width=12, height=2, state="disabled")
        self.exit_btn.pack(side=tk.LEFT, padx=10)

        # 结果展示
        result_text_frame = tk.LabelFrame(parent, text="操作结果", font=("Microsoft YaHei", 10))
        result_text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.result_text = scrolledtext.ScrolledText(result_text_frame, height=8,
                                                     font=("Consolas", 9), state="disabled")
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def _build_query_tab(self, parent):
        """构建记录查询标签页"""
        # 查询条件
        qf = tk.Frame(parent)
        qf.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(qf, text="车牌号:", font=("Microsoft YaHei", 9)).pack(side=tk.LEFT)
        self.query_plate = tk.Entry(qf, width=12, font=("Consolas", 10))
        self.query_plate.pack(side=tk.LEFT, padx=5)

        tk.Button(qf, text="🔍 查询", command=self._do_query,
                  bg="#3498db", fg="white", font=("Microsoft YaHei", 9)).pack(side=tk.LEFT, padx=5)
        tk.Button(qf, text="📊 统计", command=self._do_stats,
                  bg="#2ecc71", fg="white", font=("Microsoft YaHei", 9)).pack(side=tk.LEFT, padx=5)
        tk.Button(qf, text="📤 导出CSV", command=self._export_csv,
                  bg="#f39c12", fg="white", font=("Microsoft YaHei", 9)).pack(side=tk.LEFT, padx=5)

        # 结果表格
        columns = ("id", "车牌号", "入场时间", "出场时间", "费用", "状态")
        self.query_tree = ttk.Treeview(parent, columns=columns, show="headings", height=12)
        for col in columns:
            self.query_tree.heading(col, text=col)
            self.query_tree.column(col, width=100, anchor="center")
        self.query_tree.column("入场时间", width=140)
        self.query_tree.column("出场时间", width=140)
        self.query_tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=self.query_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
        self.query_tree.configure(yscrollcommand=scrollbar.set)

    def _build_log_tab(self, parent):
        """构建日志标签页"""
        toolbar = tk.Frame(parent)
        toolbar.pack(fill=tk.X, padx=10, pady=5)
        tk.Button(toolbar, text="🗑️ 清空", command=self._clear_log,
                  bg="#95a5a6", fg="white", font=("Microsoft YaHei", 9)).pack(side=tk.RIGHT)

        self.log_text = scrolledtext.ScrolledText(parent, font=("Consolas", 9),
                                                  state="disabled", bg="#2c3e50", fg="#ecf0f1")
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    # ==================== 网络操作 ====================

    def _toggle_connect(self):
        """切换连接状态"""
        if self.connected:
            self._disconnect()
        else:
            self._connect()

    def _connect(self):
        """连接服务器"""
        host = self.host_entry.get().strip()
        try:
            port = int(self.port_entry.get().strip())
        except ValueError:
            messagebox.showerror("错误", "端口号必须为整数")
            return

        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(5)
            self.sock.connect((host, port))
            self.sock.settimeout(None)  # 取消超时

            self.connected = True
            self._running = True
            self._update_connection_status(True)

            # 启动接收线程
            self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
            self._recv_thread.start()

            self._log(f"✅ 已连接到 {host}:{port}")
            self._query_parking_info()

        except Exception as e:
            messagebox.showerror("连接失败", str(e))
            self._log(f"❌ 连接失败: {e}")

    def _disconnect(self):
        """断开连接"""
        self._running = False
        self.connected = False
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
            self.sock = None
        self._update_connection_status(False)
        self._log("🔌 已断开连接")

    def _update_connection_status(self, connected: bool):
        """更新连接状态 UI"""
        if connected:
            self.status_label.config(text="🟢 已连接", fg="green")
            self.connect_btn.config(text="🔌 断开", bg="#e74c3c")
            self.entry_btn.config(state="normal")
            self.exit_btn.config(state="normal")
        else:
            self.status_label.config(text="⚫ 未连接", fg="red")
            self.connect_btn.config(text="🔗 连接", bg="#3498db")
            self.entry_btn.config(state="disabled")
            self.exit_btn.config(state="disabled")
            self.parking_info_label.config(text="车位: --/--")

    def _recv_loop(self):
        """接收数据循环"""
        while self._running:
            try:
                if not self.sock:
                    time.sleep(0.5)
                    continue

                data = self.sock.recv(BUFFER_SIZE)
                if not data:
                    self._log("⚠️ 服务器断开连接")
                    self.root.after(0, lambda: self._update_connection_status(False))
                    self._try_reconnect()
                    break

                messages = self.handler.feed(data)
                for msg_type, body, checksum_ok in messages:
                    self.root.after(0, lambda mt=msg_type, b=body, c=checksum_ok: self._process_message(mt, b, c))

            except (ConnectionResetError, ConnectionAbortedError, OSError):
                if self._running:
                    self._log("⚠️ 连接丢失")
                    self.root.after(0, lambda: self._update_connection_status(False))
                    self._try_reconnect()
                break
            except Exception as e:
                if self._running:
                    self._log(f"⚠️ 接收异常: {e}")
                time.sleep(0.5)

    def _try_reconnect(self):
        """尝试自动重连"""
        if not self.reconnect_enabled.get():
            return

        self._log("🔄 尝试自动重连...")
        for attempt in range(3):
            time.sleep(2)
            try:
                host = self.host_entry.get().strip()
                port = int(self.port_entry.get().strip())
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.settimeout(5)
                self.sock.connect((host, port))
                self.sock.settimeout(None)
                self.connected = True
                self._running = True
                self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
                self._recv_thread.start()
                self.root.after(0, lambda: self._update_connection_status(True))
                self._log(f"✅ 重连成功 (第{attempt + 1}次尝试)")
                self._query_parking_info()
                return
            except:
                self._log(f"⏳ 重连失败 ({attempt + 1}/3)")

        self._log("❌ 重连失败，请手动连接")

    def _send_message(self, data: bytes):
        """发送消息"""
        if not self.sock or not self.connected:
            self._log("❌ 未连接服务器")
            return False
        try:
            self.sock.sendall(data)
            return True
        except Exception as e:
            self._log(f"❌ 发送失败: {e}")
            return False

    def _process_message(self, msg_type: int, body: bytes, checksum_ok: bool):
        """处理服务器响应"""
        if not checksum_ok:
            self._log("⚠️ 收到校验和错误的数据包")
            return

        try:
            if msg_type == MSG_TYPE_JSON_RESP:
                resp = json.loads(body.decode("utf-8"))
                self._handle_response(resp)
            elif msg_type == MSG_TYPE_JSON_BILL:
                bill = json.loads(body.decode("utf-8"))
                self._handle_bill(bill)
        except json.JSONDecodeError:
            self._log("⚠️ JSON解析失败")

    def _handle_response(self, resp: dict):
        """处理JSON响应"""
        code = resp.get("code", 0)
        msg = resp.get("msg", "")

        if "ready_for_image" in resp and resp["ready_for_image"]:
            # 服务器准备好接收图片
            self._send_image()
            return

        if "plate" in resp:
            # OCR识别结果
            plate = resp["plate"]
            confidence = resp.get("confidence", 0)
            self.current_plate = plate
            self.plate_var.set(plate)
            self.confidence_label.config(text=f"{confidence:.1%}")

            if resp.get("need_confirm", False):
                self._show_result(f"✅ 识别成功: {plate}\n置信度: {confidence:.1%}\n请点击【入场】或【出场】按钮")
                self._log(f"OCR识别: {plate} (置信度: {confidence:.1%})")
            elif resp.get("need_manual_input", False):
                self._show_result(f"❌ {msg}\n请手动输入车牌号")
                self.manual_frame.pack(fill=tk.X, padx=10, pady=5, before=self.entry_btn.master)
                self._log(f"OCR识别失败: {msg}")
            return

        if "available" in resp:
            # 车位信息
            self.parking_info_label.config(
                text=f"车位: {resp['available']}/{resp['total']}"
            )

        if "records" in resp:
            self._update_query_table(resp["records"])

        if "statistics" in resp:
            self._show_stats(resp["statistics"])

        if code == 200 and "entry_time" in resp:
            self._show_result(f"✅ 入场成功!\n车牌: {resp['plate_number']}\n入场时间: {resp['entry_time']}")
            self._log(f"入场: {resp['plate_number']} @ {resp['entry_time']}")
            self._query_parking_info()

        self._log(f"[响应] {msg}")

    def _handle_bill(self, bill: dict):
        """处理计费响应"""
        code = bill.get("code", 0)
        if code == 200:
            msg = (
                f"💰 出场计费\n"
                f"{'='*30}\n"
                f"车牌号: {bill['plate_number']}\n"
                f"入场时间: {bill['entry_time']}\n"
                f"出场时间: {bill['exit_time']}\n"
                f"停车时长: {bill['duration_minutes']} 分钟\n"
                f"费用: ¥{bill['fee']}\n"
                f"{'='*30}"
            )
            self._show_result(msg)
            self._log(f"出场计费: {bill['plate_number']} ¥{bill['fee']} ({bill['duration_minutes']}分钟)")
            self._query_parking_info()
        else:
            self._show_result(f"❌ {bill.get('msg', '出场失败')}")

    def _send_image(self):
        """发送当前选择的图片"""
        if not self.current_image_path:
            self._log("❌ 请先选择图片")
            return

        try:
            with open(self.current_image_path, "rb") as f:
                image_data = f.read()

            # 先发送图片数据
            packet = pack_message(MSG_TYPE_IMAGE_DATA, image_data)
            self._send_message(packet)
            self._log(f"📤 已上传图片: {os.path.basename(self.current_image_path)} ({len(image_data)} bytes)")
        except Exception as e:
            self._log(f"❌ 发送图片失败: {e}")

    # ==================== 业务操作 ====================

    def _select_image(self):
        """选择图片文件"""
        file_path = filedialog.askopenfilename(
            title="选择车牌图片",
            filetypes=[
                ("图片文件", "*.jpg *.jpeg *.png *.bmp"),
                ("所有文件", "*.*")
            ]
        )
        if file_path:
            self.current_image_path = file_path
            self._preview_image(file_path)
            self.image_path_label.config(text=file_path)
            self._log(f"📷 已选择: {os.path.basename(file_path)}")

    def _use_sample_image(self):
        """使用示例图片（生成一张模拟车牌图片）"""
        try:
            from PIL import Image, ImageDraw, ImageFont

            # 创建模拟车牌图片
            img = Image.new('RGB', (440, 140), color='#003399')
            draw = ImageDraw.Draw(img)

            # 添加边框
            draw.rectangle([5, 5, 435, 135], outline='white', width=3)

            # 添加车牌文字
            try:
                font = ImageFont.truetype("simhei.ttf", 60)
            except:
                font = ImageFont.load_default()

            import random
            provinces = ["京", "沪", "粤", "苏", "浙", "鲁", "豫", "川", "鄂", "闽"]
            letters = [chr(ord('A') + i) for i in range(22)]
            plate_text = f"{random.choice(provinces)}{random.choice(letters)}{random.randint(10000, 99999)}"
            draw.text((30, 35), plate_text, fill='white', font=font)

            # 保存
            sample_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")
            os.makedirs(sample_dir, exist_ok=True)
            file_path = os.path.join(sample_dir, f"sample_plate_{int(time.time())}.png")
            img.save(file_path)

            self.current_image_path = file_path
            self._preview_image(file_path)
            self.image_path_label.config(text=file_path)
            self._log(f"🖼️ 已生成示例车牌图片: {plate_text}")
        except Exception as e:
            self._log(f"❌ 生成示例图片失败: {e}")
            messagebox.showerror("错误", f"生成示例图片失败: {e}")

    def _preview_image(self, file_path: str):
        """预览图片"""
        try:
            img = Image.open(file_path)
            img.thumbnail((300, 180), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.image_preview_label.config(image=photo, text="")
            self.image_preview_label.image = photo  # 保持引用
        except Exception as e:
            self.image_preview_label.config(text=f"无法预览: {e}")

    def _do_action(self, action: str):
        """执行入场或出场操作"""
        if not self.connected:
            messagebox.showwarning("提示", "请先连接服务器")
            return

        self.pending_action = action

        # 如果已有识别结果（车牌号），直接发送确认操作
        if self.current_plate:
            self._confirm_action(action, self.current_plate)
        elif self.current_image_path:
            # 先上传图片识别
            if action == "entry":
                self._send_message(pack_json(MSG_TYPE_JSON_META, {"cmd": "entry_query"}))
            else:
                self._send_message(pack_json(MSG_TYPE_JSON_META, {"cmd": "exit_query"}))
        else:
            messagebox.showwarning("提示", "请先选择车牌图片")

    def _confirm_action(self, action: str, plate: str):
        """确认入场/出场操作"""
        cmd = "manual_entry" if action == "entry" else "manual_exit"
        self._send_message(pack_json(MSG_TYPE_JSON_META, {"cmd": cmd, "plate": plate}))
        self._log(f"{'入场' if action == 'entry' else '出场'}确认: {plate}")

    def _manual_confirm(self):
        """手动输入车牌号确认"""
        plate = self.manual_plate_entry.get().strip().upper()
        if not plate:
            messagebox.showwarning("提示", "请输入车牌号")
            return

        if not is_valid_plate(plate):
            messagebox.showwarning("提示", "车牌号格式不正确")
            return

        self.current_plate = plate
        self.plate_var.set(plate)
        self.confidence_label.config(text="手动")
        self.manual_frame.pack_forget()
        self._log(f"手动输入车牌: {plate}")

    def _query_parking_info(self):
        """查询车位信息"""
        self._send_message(pack_json(MSG_TYPE_JSON_META, {"cmd": "entry_query"}))

    def _do_query(self):
        """查询停车记录"""
        plate = self.query_plate.get().strip()
        self._send_message(pack_json(MSG_TYPE_JSON_META, {"cmd": "query", "plate": plate}))
        self._log(f"查询记录: {plate if plate else '全部'}")

    def _do_stats(self):
        """获取统计数据"""
        self._send_message(pack_json(MSG_TYPE_JSON_META, {"cmd": "stats"}))
        self._log("获取统计数据")

    def _update_query_table(self, records: list):
        """更新查询结果表格"""
        for item in self.query_tree.get_children():
            self.query_tree.delete(item)

        for rec in records:
            status = "在场" if rec["status"] == "in" else "已出场"
            fee_str = f"¥{rec['fee']}" if rec["fee"] else "--"
            exit_str = rec["exit_time"] if rec["exit_time"] else "--"
            self.query_tree.insert("", "end", values=(
                rec["id"], rec["plate_number"], rec["entry_time"],
                exit_str, fee_str, status
            ))

        self._log(f"查询结果: {len(records)} 条记录")

    def _show_stats(self, stats: dict):
        """显示统计信息"""
        msg = (
            f"📊 统计数据\n"
            f"{'='*30}\n"
            f"总入场: {stats['total_entries']} 辆\n"
            f"总出场: {stats['total_exits']} 辆\n"
            f"总营收: ¥{stats['total_revenue']}\n"
            f"当前在场: {stats['current_in']} 辆\n"
            f"{'='*30}"
        )
        self._show_result(msg)
        self._log(f"统计: 入场{stats['total_entries']} 出场{stats['total_exits']} 营收¥{stats['total_revenue']}")

    def _export_csv(self):
        """导出CSV"""
        import csv
        file_path = filedialog.asksaveasfilename(
            title="导出CSV",
            defaultextension=".csv",
            filetypes=[("CSV文件", "*.csv")]
        )
        if not file_path:
            return

        # 获取表格数据
        rows = []
        for item in self.query_tree.get_children():
            rows.append(self.query_tree.item(item)["values"])

        if not rows:
            messagebox.showwarning("提示", "没有数据可导出，请先查询")
            return

        with open(file_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "车牌号", "入场时间", "出场时间", "费用", "状态"])
            writer.writerows(rows)

        self._log(f"📤 已导出CSV: {file_path}")
        messagebox.showinfo("成功", f"已导出到 {file_path}")

    def _show_result(self, text: str):
        """显示操作结果"""
        self.result_text.config(state="normal")
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(1.0, text)
        self.result_text.config(state="disabled")

    def _clear_log(self):
        """清空日志"""
        self.log_text.config(state="normal")
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state="disabled")

    def _log(self, msg: str):
        """添加日志"""
        timestamp = time.strftime("%H:%M:%S")
        line = f"[{timestamp}] {msg}\n"

        def _append():
            self.log_text.config(state="normal")
            self.log_text.insert(tk.END, line)
            self.log_text.see(tk.END)
            self.log_text.config(state="disabled")

        self.root.after(0, _append)

    def _on_close(self):
        """关闭窗口"""
        self._running = False
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
        self.root.destroy()

    def run(self):
        """启动客户端"""
        self.root.mainloop()


# ==================== CLI 命令行客户端（tkinter 不可用时的降级方案） ====================

class ParkingClientCLI:
    """停车场客户端 — 命令行交互模式"""

    def __init__(self):
        self.sock: socket.socket = None
        self.handler = ProtocolHandler()
        self.connected = False
        self.running = False
        self._recv_thread = None
        self._recv_lock = threading.Lock()
        self._pending_responses = []  # 收到的响应队列

        self.current_image_path: str = None
        self.current_plate: str = None

    # ========== 网络操作 ==========

    def _connect(self, host: str = SERVER_HOST, port: int = SERVER_PORT):
        """连接服务器"""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(5)
            self.sock.connect((host, port))
            self.sock.settimeout(None)
            self.connected = True
            self.running = True

            self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
            self._recv_thread.start()

            print(f"✅ 已连接到 {host}:{port}")
            return True
        except Exception as e:
            print(f"❌ 连接失败: {e}")
            return False

    def _disconnect(self):
        """断开连接"""
        self.running = False
        self.connected = False
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
            self.sock = None
        print("🔌 已断开连接")

    def _recv_loop(self):
        """接收数据循环"""
        while self.running:
            try:
                if not self.sock:
                    time.sleep(0.5)
                    continue
                data = self.sock.recv(BUFFER_SIZE)
                if not data:
                    print("\n⚠️ 服务器断开连接")
                    self.connected = False
                    self.running = False
                    break

                messages = self.handler.feed(data)
                for msg_type, body, checksum_ok in messages:
                    if checksum_ok and msg_type in (MSG_TYPE_JSON_RESP, MSG_TYPE_JSON_BILL):
                        try:
                            resp = json.loads(body.decode("utf-8"))
                            with self._recv_lock:
                                self._pending_responses.append(resp)
                        except json.JSONDecodeError:
                            pass
            except (ConnectionResetError, ConnectionAbortedError, OSError):
                if self.running:
                    print("\n⚠️ 连接丢失")
                    self.connected = False
                    self.running = False
                break
            except Exception as e:
                if self.running:
                    pass  # 静默处理
                time.sleep(0.5)

    def _send_message(self, data: bytes) -> bool:
        """发送消息"""
        if not self.sock or not self.connected:
            print("❌ 未连接服务器")
            return False
        try:
            self.sock.sendall(data)
            return True
        except Exception as e:
            print(f"❌ 发送失败: {e}")
            return False

    def _wait_response(self, timeout: float = 10.0) -> dict:
        """等待服务器响应"""
        start = time.time()
        while time.time() - start < timeout:
            with self._recv_lock:
                if self._pending_responses:
                    return self._pending_responses.pop(0)
            time.sleep(0.1)
        return {"code": 408, "msg": "等待响应超时"}

    def _send_and_wait(self, data: bytes, timeout: float = 10.0) -> dict:
        """发送并等待响应"""
        if not self._send_message(data):
            return {"code": 500, "msg": "发送失败"}
        return self._wait_response(timeout)

    # ========== 业务操作 ==========

    def _upload_image_and_recognize(self) -> bool:
        """上传图片并等待识别结果"""
        if not self.current_image_path:
            print("❌ 请先选择图片")
            return False

        try:
            with open(self.current_image_path, "rb") as f:
                image_data = f.read()

            print(f"📤 正在上传图片 ({len(image_data)} bytes)...")
            resp = self._send_and_wait(pack_message(MSG_TYPE_IMAGE_DATA, image_data))

            if resp.get("code") == 200 and resp.get("plate"):
                plate = resp["plate"]
                confidence = resp.get("confidence", 0)
                self.current_plate = plate
                print(f"✅ 识别成功: {plate} (置信度: {confidence:.1%})")
                return True
            elif resp.get("need_manual_input"):
                print(f"⚠️ 自动识别失败: {resp.get('msg')}")
                plate = input("请手动输入车牌号: ").strip().upper()
                if is_valid_plate(plate):
                    self.current_plate = plate
                    return True
                else:
                    print("❌ 车牌号格式不正确")
                    return False
            else:
                print(f"❌ 识别失败: {resp.get('msg', '未知错误')}")
                return False
        except Exception as e:
            print(f"❌ 上传失败: {e}")
            return False

    def do_entry(self):
        """执行入场操作"""
        if not self.current_plate:
            print("❌ 请先上传图片识别车牌")
            return

        resp = self._send_and_wait(
            pack_json(MSG_TYPE_JSON_META, {"cmd": "manual_entry", "plate": self.current_plate})
        )
        if resp.get("code") == 200:
            print(f"✅ 入场成功! 车牌: {resp['plate_number']}  时间: {resp['entry_time']}")
        else:
            print(f"❌ 入场失败: {resp.get('msg', '未知错误')}")

    def do_exit(self):
        """执行出场操作"""
        if not self.current_plate:
            print("❌ 请先上传图片识别车牌")
            return

        resp = self._send_and_wait(
            pack_json(MSG_TYPE_JSON_META, {"cmd": "manual_exit", "plate": self.current_plate})
        )
        if resp.get("code") == 200:
            print()
            print("=" * 40)
            print(f"  💰 出场计费")
            print(f"  车牌号: {resp['plate_number']}")
            print(f"  入场时间: {resp['entry_time']}")
            print(f"  出场时间: {resp['exit_time']}")
            print(f"  停车时长: {resp['duration_minutes']} 分钟")
            print(f"  费用: ¥{resp['fee']}")
            print("=" * 40)
        else:
            print(f"❌ 出场失败: {resp.get('msg', '未知错误')}")

    def do_query(self):
        """查询记录"""
        plate = input("车牌号（回车查全部）: ").strip()
        resp = self._send_and_wait(
            pack_json(MSG_TYPE_JSON_META, {"cmd": "query", "plate": plate})
        )
        records = resp.get("records", [])
        if records:
            print(f"\n{'ID':<6} {'车牌号':<12} {'入场时间':<20} {'出场时间':<20} {'费用':<10} {'状态'}")
            print("-" * 80)
            for r in records:
                fee_str = f"¥{r['fee']}" if r['fee'] else "--"
                exit_str = r['exit_time'] if r['exit_time'] else "--"
                status = "在场" if r['status'] == 'in' else "已出场"
                print(f"{r['id']:<6} {r['plate_number']:<12} {r['entry_time']:<20} {exit_str:<20} {fee_str:<10} {status}")
            print(f"\n共 {len(records)} 条记录")
        else:
            print("无记录")

    def do_stats(self):
        """查看统计"""
        resp = self._send_and_wait(
            pack_json(MSG_TYPE_JSON_META, {"cmd": "stats"})
        )
        stats = resp.get("statistics", {})
        print()
        print("=" * 30)
        print(f"  📊 停车场统计")
        print(f"  总入场: {stats.get('total_entries', 0)} 辆")
        print(f"  总出场: {stats.get('total_exits', 0)} 辆")
        print(f"  总营收: ¥{stats.get('total_revenue', 0)}")
        print(f"  当前在场: {stats.get('current_in', 0)} 辆")
        print("=" * 30)

    def select_image(self):
        """选择图片文件"""
        path = input("请输入车牌图片路径（直接回车生成示例图片）: ").strip().strip('"')
        if not path:
            self._generate_sample_image()
            return

        if not os.path.exists(path):
            print(f"❌ 文件不存在: {path}")
            return

        self.current_image_path = path
        print(f"📷 已选择: {os.path.basename(path)}")

    def _generate_sample_image(self):
        """生成示例车牌图片"""
        try:
            from PIL import Image as PILImage, ImageDraw, ImageFont
            img = PILImage.new('RGB', (440, 140), color='#003399')
            draw = ImageDraw.Draw(img)
            draw.rectangle([5, 5, 435, 135], outline='white', width=3)

            try:
                font = ImageFont.truetype("simhei.ttf", 60)
            except:
                font = ImageFont.load_default()

            import random
            provinces = ["京", "沪", "粤", "苏", "浙", "鲁", "豫", "川", "鄂", "闽"]
            letters = [chr(ord('A') + i) for i in range(22)]
            plate_text = f"{random.choice(provinces)}{random.choice(letters)}{random.randint(10000, 99999)}"
            draw.text((30, 35), plate_text, fill='white', font=font)

            sample_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")
            os.makedirs(sample_dir, exist_ok=True)
            file_path = os.path.join(sample_dir, f"sample_plate_{int(time.time())}.png")
            img.save(file_path)

            self.current_image_path = file_path
            print(f"🖼️ 已生成示例车牌图片: {plate_text}")
        except ImportError:
            print("❌ 需要 Pillow 库来生成示例图片，请运行: pip install Pillow")
            print("   或者手动输入图片路径。")
        except Exception as e:
            print(f"❌ 生成示例图片失败: {e}")

    def run(self):
        """命令行主循环"""
        print()
        print("=" * 50)
        print("  🅿️  停车场出入场系统 — 命令行客户端")
        print("=" * 50)

        host = input(f"服务器地址 [{SERVER_HOST}]: ").strip() or SERVER_HOST
        port_str = input(f"端口 [{SERVER_PORT}]: ").strip() or str(SERVER_PORT)
        try:
            port = int(port_str)
        except ValueError:
            print("端口号格式错误")
            return

        if not self._connect(host, port):
            return

        print()
        print("提示: 输入 help 查看命令列表，输入 quit 退出")
        print()

        try:
            while self.connected and self.running:
                cmd = input("> ").strip().lower()

                if cmd in ("quit", "exit", "q"):
                    break
                elif cmd == "help":
                    self._print_help()
                elif cmd in ("select", "s", "1"):
                    self.select_image()
                elif cmd in ("upload", "u", "2"):
                    if not self.current_image_path:
                        print("请先选择图片 (select)")
                    else:
                        self._upload_image_and_recognize()
                elif cmd in ("entry", "e", "3"):
                    self.do_entry()
                elif cmd in ("exit_park", "x", "4"):
                    self.do_exit()
                elif cmd in ("query", "qy", "5"):
                    self.do_query()
                elif cmd in ("stats", "st", "6"):
                    self.do_stats()
                elif cmd == "":
                    continue
                else:
                    print(f"未知命令: {cmd}，输入 help 查看帮助")
        except KeyboardInterrupt:
            print()
        finally:
            self._disconnect()

        print("👋 再见!")

    def _print_help(self):
        print("""
┌────────────────────────────────────────────┐
│  可用命令:                                   │
│  select / s  / 1  — 选择车牌图片             │
│  upload / u  / 2  — 上传图片并识别车牌        │
│  entry  / e  / 3  — 车辆入场登记             │
│  exit   / x  / 4  — 车辆出场计费             │
│  query  / qy / 5  — 查询停车记录             │
│  stats  / st / 6  — 查看统计数据             │
│  help            — 显示此帮助                │
│  quit  / q       — 退出程序                  │
│                                              │
│  典型操作流程:                                │
│    select → upload → entry  (入场)           │
│    select → upload → exit   (出场)           │
└────────────────────────────────────────────┘
""")


# ==================== 启动入口 ====================

def main():
    if HAS_GUI:
        print("🖥️ 启动 GUI 模式...")
        client = ParkingClient()
        client.run()
    else:
        print("⚠️ tkinter 模块不可用，使用命令行模式")
        print("   如需 GUI，请安装带 tkinter 的 Python 版本")
        print("   或运行: pip install pillow (用于生成示例图片)")
        client = ParkingClientCLI()
        client.run()


if __name__ == "__main__":
    main()

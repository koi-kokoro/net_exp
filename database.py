"""
数据库模块 — SQLite持久化停车记录
表结构：
  - parking_records: 停车记录表
  - fee_rules: 计费规则表
"""

import sqlite3
import threading
from datetime import datetime
from config import DB_PATH

# 数据库连接锁（多线程安全）
_db_lock = threading.Lock()


def get_connection() -> sqlite3.Connection:
    """获取数据库连接（每次新建，线程安全）"""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")  # 提高并发写入性能
    return conn


def init_db():
    """初始化数据库，创建表结构"""
    conn = get_connection()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS parking_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_number VARCHAR(20) NOT NULL,
                entry_time DATETIME NOT NULL,
                exit_time DATETIME,
                entry_image_path VARCHAR(255),
                exit_image_path VARCHAR(255),
                fee REAL,
                status VARCHAR(10) DEFAULT 'in'
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS fee_rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rule_name VARCHAR(50),
                price_per_hour REAL,
                daily_cap REAL,
                free_minutes INTEGER DEFAULT 15
            )
        """)
        # 插入默认计费规则（如果不存在）
        conn.execute("""
            INSERT OR IGNORE INTO fee_rules (id, rule_name, price_per_hour, daily_cap, free_minutes)
            VALUES (1, '标准计费', 5.0, 30.0, 15)
        """)
        conn.commit()
    finally:
        conn.close()


def register_entry(plate_number: str, image_path: str = "") -> dict:
    """
    入场登记
    
    返回:
        {"success": True/False, "msg": "...", "record_id": int, "entry_time": str}
    """
    with _db_lock:
        conn = get_connection()
        try:
            # 检查是否已入场且未出场
            cur = conn.execute(
                "SELECT id FROM parking_records WHERE plate_number=? AND status='in'",
                (plate_number,)
            )
            if cur.fetchone():
                return {"success": False, "msg": f"车牌 {plate_number} 已在停车场内，禁止重复入场"}

            entry_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cur = conn.execute(
                "INSERT INTO parking_records (plate_number, entry_time, entry_image_path, status) VALUES (?, ?, ?, 'in')",
                (plate_number, entry_time, image_path)
            )
            conn.commit()
            record_id = cur.lastrowid
            return {
                "success": True,
                "msg": f"入场成功",
                "record_id": record_id,
                "plate_number": plate_number,
                "entry_time": entry_time
            }
        finally:
            conn.close()


def register_exit(plate_number: str, image_path: str = "") -> dict:
    """
    出场登记 + 计费
    
    返回:
        {"success": True/False, "msg": "...", "fee": float, "duration_minutes": int, ...}
    """
    with _db_lock:
        conn = get_connection()
        try:
            # 查找在场记录
            cur = conn.execute(
                "SELECT id, entry_time FROM parking_records WHERE plate_number=? AND status='in' ORDER BY entry_time DESC LIMIT 1",
                (plate_number,)
            )
            row = cur.fetchone()
            if not row:
                return {"success": False, "msg": f"车牌 {plate_number} 不在停车场内或已出场"}

            record_id = row["id"]
            entry_time_str = row["entry_time"]
            entry_time = datetime.strptime(entry_time_str, "%Y-%m-%d %H:%M:%S")
            exit_time = datetime.now()
            exit_time_str = exit_time.strftime("%Y-%m-%d %H:%M:%S")

            # 计算费用
            fee, duration_minutes = calculate_fee(entry_time, exit_time)

            conn.execute(
                "UPDATE parking_records SET exit_time=?, exit_image_path=?, fee=?, status='out' WHERE id=?",
                (exit_time_str, image_path, fee, record_id)
            )
            conn.commit()

            return {
                "success": True,
                "msg": "出场成功",
                "record_id": record_id,
                "plate_number": plate_number,
                "entry_time": entry_time_str,
                "exit_time": exit_time_str,
                "duration_minutes": duration_minutes,
                "fee": fee
            }
        finally:
            conn.close()


def calculate_fee(entry_time: datetime, exit_time: datetime) -> tuple:
    """
    计算停车费用
    
    计费规则：
    - 免费时长内免费
    - 白天 (8:00-20:00): 标准费率
    - 夜间 (20:00-8:00): 8折优惠
    - 每日封顶
    
    返回: (费用, 停车分钟数)
    """
    conn = get_connection()
    try:
        cur = conn.execute("SELECT * FROM fee_rules WHERE id=1")
        rule = cur.fetchone()
        if rule:
            price_per_hour = rule["price_per_hour"]
            daily_cap = rule["daily_cap"]
            free_minutes = rule["free_minutes"]
        else:
            price_per_hour = 5.0
            daily_cap = 30.0
            free_minutes = 15
    finally:
        conn.close()

    duration_seconds = (exit_time - entry_time).total_seconds()
    duration_minutes = max(0, int(duration_seconds / 60))

    # 免费时长
    if duration_minutes <= free_minutes:
        return (0.0, duration_minutes)

    # 简化计费：按小时计费，不足1小时按1小时算
    billable_minutes = duration_minutes - free_minutes
    hours = (billable_minutes + 59) // 60  # 向上取整

    # 判断白天/夜间（简化：取入场时间判断）
    hour = entry_time.hour
    is_night = hour >= 20 or hour < 8
    rate = price_per_hour * (0.8 if is_night else 1.0)

    fee = hours * rate

    # 每日封顶
    days = max(1, (duration_minutes + 1439) // 1440)
    fee = min(fee, daily_cap * days)

    return (round(fee, 2), duration_minutes)


def get_occupied_spaces() -> int:
    """获取当前已占用车位数"""
    conn = get_connection()
    try:
        cur = conn.execute("SELECT COUNT(*) as cnt FROM parking_records WHERE status='in'")
        return cur.fetchone()["cnt"]
    finally:
        conn.close()


def query_records(plate_number: str = None, start_date: str = None, end_date: str = None) -> list:
    """查询停车记录"""
    conn = get_connection()
    try:
        sql = "SELECT * FROM parking_records WHERE 1=1"
        params = []
        if plate_number:
            sql += " AND plate_number LIKE ?"
            params.append(f"%{plate_number}%")
        if start_date:
            sql += " AND entry_time >= ?"
            params.append(start_date)
        if end_date:
            sql += " AND entry_time <= ?"
            params.append(end_date + " 23:59:59")
        sql += " ORDER BY entry_time DESC LIMIT 100"
        cur = conn.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def get_statistics(date_str: str = None) -> dict:
    """
    获取统计数据
    date_str: 日期字符串 "2026-05-05"，为空则统计全部
    """
    conn = get_connection()
    try:
        if date_str:
            date_filter = "WHERE date(entry_time) = ?"
            params = (date_str,)
        else:
            date_filter = ""
            params = ()

        cur = conn.execute(f"SELECT COUNT(*) as cnt FROM parking_records {date_filter}", params)
        total_entries = cur.fetchone()["cnt"]

        cur = conn.execute(
            f"SELECT COUNT(*) as cnt FROM parking_records WHERE status='out' {'AND date(entry_time) = ?' if date_str else ''}",
            params
        )
        total_exits = cur.fetchone()["cnt"]

        cur = conn.execute(
            f"SELECT COALESCE(SUM(fee), 0) as total FROM parking_records WHERE status='out' {'AND date(entry_time) = ?' if date_str else ''}",
            params
        )
        total_revenue = cur.fetchone()["total"]

        cur = conn.execute("SELECT COUNT(*) as cnt FROM parking_records WHERE status='in'")
        current_in = cur.fetchone()["cnt"]

        return {
            "total_entries": total_entries,
            "total_exits": total_exits,
            "total_revenue": round(total_revenue, 2),
            "current_in": current_in
        }
    finally:
        conn.close()

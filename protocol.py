"""
自定义应用层协议 — 混合法（固定头部 + 长度前缀 + 校验和 + 魔数）
解决TCP粘包/拆包问题

协议格式（共13字节头部 + N字节数据体）：
┌──────────────────────────────────────────────────────┐
│  魔数 (4B)  │  类型 (1B)  │  数据长度 (4B)  │  校验和 (4B)  │
│  0xCAFE0001 │  0x01~0xFF  │  N bytes       │  MD5前4字节   │
├──────────────────────────────────────────────────────┤
│              消息体（变长，N bytes）                    │
└──────────────────────────────────────────────────────┘

粘包处理策略：
1. 混合法（仿HTTP）：固定头部（JSON元数据）+ 长度前缀（二进制数据）
2. 三重校验：魔数 + 长度范围 + MD5校验和
3. 接收缓冲区 + 循环解析，处理多条消息粘连
"""

import struct
import hashlib
import json
from config import PROTO_MAGIC, PROTO_HEADER_SIZE, MSG_TYPE_JSON_META, MSG_TYPE_IMAGE_DATA, MSG_TYPE_JSON_RESP, MSG_TYPE_JSON_BILL


def calc_checksum(data: bytes) -> int:
    """计算MD5校验和的前4字节（作为32位整数）"""
    md5 = hashlib.md5(data).digest()
    return struct.unpack(">I", md5[:4])[0]


def pack_message(msg_type: int, data: bytes) -> bytes:
    """
    打包消息：头部 + 数据体
    
    参数:
        msg_type: 消息类型 (MSG_TYPE_JSON_META / MSG_TYPE_IMAGE_DATA / MSG_TYPE_JSON_RESP / MSG_TYPE_JSON_BILL)
        data: 消息体（JSON字符串需先encode为bytes，图片为原始bytes）
    
    返回:
        完整的二进制消息包
    """
    data_len = len(data)
    checksum = calc_checksum(data)
    header = struct.pack(">IBiI", PROTO_MAGIC, msg_type, data_len, checksum)
    return header + data


def pack_json(msg_type: int, obj: dict) -> bytes:
    """便捷方法：将字典打包为JSON消息"""
    json_str = json.dumps(obj, ensure_ascii=False)
    return pack_message(msg_type, json_str.encode("utf-8"))


def unpack_header(header: bytes) -> tuple:
    """
    解析消息头部
    
    返回:
        (magic, msg_type, data_len, checksum) 或 解析失败返回 (0, 0, 0, 0)
    """
    if len(header) < PROTO_HEADER_SIZE:
        return (0, 0, 0, 0)
    try:
        magic, msg_type, data_len, checksum = struct.unpack(">IBiI", header)
        return (magic, msg_type, data_len, checksum)
    except struct.error:
        return (0, 0, 0, 0)


def verify_message(data: bytes, expected_checksum: int) -> bool:
    """校验消息体完整性"""
    return calc_checksum(data) == expected_checksum


class ProtocolHandler:
    """
    协议处理器 — 负责从TCP字节流中正确拆分出完整消息
    核心功能：
    1. 接收缓冲区管理
    2. 粘包/拆包处理（循环解析，返回多条完整消息）
    3. 魔数校验 + 长度校验 + MD5校验和校验
    """

    def __init__(self):
        self._buffer = b""  # 接收缓冲区

    def feed(self, data: bytes) -> list:
        """
        喂入新收到的字节数据，返回解析出的完整消息列表
        
        每条完整消息格式: (msg_type, raw_data_bytes, checksum_ok)
        
        粘包处理逻辑：
        1. 将新数据追加到缓冲区
        2. 循环尝试从缓冲区读取头部（13字节）
        3. 校验魔数 → 读取数据长度 → 读取完整数据体 → 校验校验和
        4. 如果数据不足（拆包），等待下次 feed
        5. 如果解析完一条，继续尝试解析下一条（粘包）
        """
        self._buffer += data
        messages = []

        while len(self._buffer) >= PROTO_HEADER_SIZE:
            # 读取并校验头部
            magic, msg_type, data_len, checksum = unpack_header(self._buffer[:PROTO_HEADER_SIZE])

            # 魔数校验
            if magic != PROTO_MAGIC:
                # 魔数不匹配，丢弃1字节后重新尝试同步
                self._buffer = self._buffer[1:]
                continue

            # 消息类型合法性校验
            if msg_type not in (MSG_TYPE_JSON_META, MSG_TYPE_IMAGE_DATA, MSG_TYPE_JSON_RESP, MSG_TYPE_JSON_BILL):
                self._buffer = self._buffer[PROTO_HEADER_SIZE:]  # 跳过非法头部
                continue

            # 长度合法性校验
            if data_len < 0 or data_len > 100 * 1024 * 1024:  # 最大100MB
                self._buffer = self._buffer[PROTO_HEADER_SIZE:]
                continue

            total_size = PROTO_HEADER_SIZE + data_len
            if len(self._buffer) < total_size:
                # 数据体不完整（拆包），等待更多数据
                break

            # 提取数据体
            body = self._buffer[PROTO_HEADER_SIZE:total_size]
            self._buffer = self._buffer[total_size:]

            # 校验和校验
            checksum_ok = verify_message(body, checksum)

            messages.append((msg_type, body, checksum_ok))

        return messages

    def reset(self):
        """重置缓冲区"""
        self._buffer = b""

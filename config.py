"""
配置文件 — 停车场出入场系统
"""
import os

# ==================== 服务器配置 ====================
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8888
MAX_CLIENTS = 50
BUFFER_SIZE = 4096
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB

# ==================== 协议配置 ====================
PROTO_MAGIC = 0xCAFE0001
PROTO_HEADER_SIZE = 13  # 魔数4 + 类型1 + 数据长度4 + 校验和4

# 消息类型
MSG_TYPE_JSON_META = 0x01    # JSON元数据
MSG_TYPE_IMAGE_DATA = 0x02   # 二进制图片数据
MSG_TYPE_JSON_RESP = 0x03    # JSON响应
MSG_TYPE_JSON_BILL = 0x04    # JSON计费

# ==================== 停车场配置 ====================
TOTAL_PARKING_SPACES = 50
DEFAULT_PRICE_PER_HOUR = 5.0   # 每小时5元
DAILY_CAP = 30.0               # 每日封顶30元
FREE_MINUTES = 15              # 免费15分钟
NIGHT_DISCOUNT_START = 20      # 夜间优惠开始 20:00
NIGHT_DISCOUNT_END = 8         # 夜间优惠结束 08:00
NIGHT_DISCOUNT_RATE = 0.8      # 夜间8折

# ==================== 数据库配置 ====================
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parking.db")

# ==================== 图片保存路径 ====================
IMAGE_SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)

# ==================== OCR配置 ====================
# 置信度阈值（EasyOCR 优化配置）
OCR_CONFIDENCE_HIGH = 0.55   # 高置信度：自动接受（55%）
OCR_CONFIDENCE_LOW = 0.3     # 低置信度：匹配车牌格式则接受(需确认)，否则拒绝（30%）

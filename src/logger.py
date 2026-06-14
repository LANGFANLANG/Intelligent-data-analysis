"""
结构化日志模块
──────────────
统一的日志配置，支持:
  - 文件输出: JSON 格式，适合机器解析和日志平台接入
  - 控制台输出: 带时间戳的易读格式
  - 自动轮转: 单文件 10MB，保留最近 5 个
  - 级别控制: 环境变量 LOG_LEVEL，默认 INFO

用法:
    from src.logger import get_logger
    logger = get_logger(__name__)
    logger.info("数据库初始化完成")
    logger.error("保存链路失败", exc_info=True)
"""
import os
import logging
import json
from logging.handlers import RotatingFileHandler
from pathlib import Path

# ── 日志级别 ──
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# ── 日志文件路径 ──
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "app.log"

# ── 日志格式器 ──

class _JsonFormatter(logging.Formatter):
    """JSON 格式，每行一条结构化日志"""

    def format(self, record: logging.LogRecord) -> str:
        return json.dumps({
            "ts": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            **({"exc": self.formatException(record.exc_info)}
               if record.exc_info else {}),
        }, ensure_ascii=False)


_CONSOLE_FMT = logging.Formatter(
    "%(asctime)s [%(levelname)-5s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)

# ── 根日志器配置（模块级，import 时自动执行）──
_root_logger = logging.getLogger()
_root_logger.setLevel(LOG_LEVEL)

# 文件处理器（JSON + 轮转）
_file_handler = RotatingFileHandler(
    LOG_FILE, encoding="utf-8", maxBytes=10 * 1024 * 1024, backupCount=5,
)
_file_handler.setLevel(LOG_LEVEL)
_file_handler.setFormatter(_JsonFormatter())
_root_logger.addHandler(_file_handler)

# 控制台处理器
_console_handler = logging.StreamHandler()
_console_handler.setLevel(LOG_LEVEL)
_console_handler.setFormatter(_CONSOLE_FMT)
_root_logger.addHandler(_console_handler)


def get_logger(name: str) -> logging.Logger:
    """获取指定名称的日志器"""
    return logging.getLogger(name)

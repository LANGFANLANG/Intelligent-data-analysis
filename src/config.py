"""
全局配置模块
─────────────
负责读取 .env 文件和系统环境变量，统一管理所有配置项。
启动时会校验必要的配置是否存在。
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 从项目根目录加载 .env 文件（显式指定路径，避免中文路径兼容问题）
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# ── DeepSeek 模型配置 ──
# 优先读取 .env 中的 DEEPSEEK_API_KEY，fallback 到系统环境变量 DEEPSEEK
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY") or os.getenv("DEEPSEEK", "")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

# ── PostgreSQL 数据库配置 ──
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://data_agent:data_agent_pass@localhost:5432/data_agent")

# ── 启动校验 ──
if not DEEPSEEK_API_KEY:
    raise ValueError(
        "未找到 API Key，请通过以下任一方式设置：\n"
        "  1. 在 .env 文件中设置 DEEPSEEK_API_KEY=sk-xxx\n"
        "  2. 设置系统环境变量 DEEPSEEK 或 DEEPSEEK_API_KEY\n"
        "可参考 .env.example 文件格式"
    )

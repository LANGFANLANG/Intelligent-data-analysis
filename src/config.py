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
load_dotenv(dotenv_path=env_path, override=True)

# ── DeepSeek 模型配置 ──
# 优先读取 .env 中的 DEEPSEEK_API_KEY，fallback 到系统环境变量 DEEPSEEK
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY") or os.getenv("DEEPSEEK", "")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

# ── PostgreSQL 数据库配置 ──
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://data_agent:data_agent_pass@localhost:5432/data_agent")
DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "5"))
DB_POOL_RECYCLE = int(os.getenv("DB_POOL_RECYCLE", "3600"))

# ── Agent 运行参数 ──
AGENT_MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "40"))
AGENT_TIMEOUT_SECONDS = int(os.getenv("AGENT_TIMEOUT_SECONDS", "120"))

# ── 可视化参数 ──
VIZ_MAX_PIE_CATEGORIES = int(os.getenv("VIZ_MAX_PIE_CATEGORIES", "8"))
VIZ_MAX_BAR_CATEGORIES = int(os.getenv("VIZ_MAX_BAR_CATEGORIES", "30"))
VIZ_DPI = int(os.getenv("VIZ_DPI", "200"))

# ── 数据加载参数 ──
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "100"))
SAMPLE_MAX_ROWS = int(os.getenv("SAMPLE_MAX_ROWS", "50000"))

# ── Streamlit 端口 ──
STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))

# ── MySQL 数据库直连配置 ──
DB_TYPE = os.getenv("DB_TYPE", "")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_NAME = os.getenv("DB_NAME", "")
DB_USER = os.getenv("DB_USER", "")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_CHARSET = os.getenv("DB_CHARSET", "utf8mb4")

# ── SQL 执行限制 ──
SQL_MAX_ROWS = int(os.getenv("SQL_MAX_ROWS", "1000"))
SQL_TIMEOUT_SEC = int(os.getenv("SQL_TIMEOUT_SEC", "30"))

# ── Schema 缓存 ──
SCHEMA_CACHE_TTL_MINUTES = int(os.getenv("SCHEMA_CACHE_TTL_MINUTES", "30"))

# ── Langfuse 观测平台配置 ──
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "http://localhost:3000")
LANGFUSE_ENABLED = bool(LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY and "xxx" not in LANGFUSE_SECRET_KEY)

# ── 启动校验 ──
if not DEEPSEEK_API_KEY:
    raise ValueError(
        "未找到 API Key，请通过以下任一方式设置：\n"
        "  1. 在 .env 文件中设置 DEEPSEEK_API_KEY=sk-xxx\n"
        "  2. 设置系统环境变量 DEEPSEEK 或 DEEPSEEK_API_KEY\n"
        "可参考 .env.example 文件格式"
    )

# ── API Key 安全提醒 ──
if env_path.exists() and os.getenv("DEEPSEEK_API_KEY"):
    from src.logger import get_logger
    _log = get_logger("config")
    import stat as _stat
    try:
        mode = env_path.stat().st_mode
        if mode & _stat.S_IROTH or mode & _stat.S_IWOTH:
            _log.warning(".env 文件权限过宽，建议执行 chmod 600 .env")
    except Exception:
        pass
    _log.warning("API Key 存储在 .env 文件中，确保不要提交到 Git 仓库")
    _log.warning("建议: 定期到 DeepSeek 后台轮换 API Key (https://platform.deepseek.com/api_keys)")

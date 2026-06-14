"""
API 限流模块
───────────
基于令牌桶的轻量级限流器，防止并发请求击穿 DeepSeek API 配额。

用法:
    from src.llm.rate_limiter import rate_limiter

    if not rate_limiter.acquire():
        raise Exception("请求过于频繁，请稍后重试")

配置（环境变量）:
    LLM_RATE_LIMIT: 每分钟最大请求数，默认 30
    LLM_BURST_SIZE: 突发容忍数，默认 5
"""
import os
import time
import threading

# ── 配置 ──
_RATE_LIMIT = int(os.getenv("LLM_RATE_LIMIT", "30"))      # 请求/分钟
_BURST_SIZE = int(os.getenv("LLM_BURST_SIZE", "5"))        # 突发容量


class TokenBucket:
    """线程安全的令牌桶"""

    def __init__(self, rate: int, burst: int):
        self._rate = rate
        self._burst = burst
        self._tokens = float(burst)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> bool:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(self._burst, self._tokens + elapsed * self._rate / 60)
            self._last_refill = now

            if self._tokens >= 1:
                self._tokens -= 1
                return True
            return False


rate_limiter = TokenBucket(_RATE_LIMIT, _BURST_SIZE)

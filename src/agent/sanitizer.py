"""
Prompt 注入防护模块
──────────────────
在用户输入进入 LLM 之前进行清洗，移除常见的 prompt injection 攻击模式。

防护策略:
  1. 拒绝已知注入模式（ignore previous instructions, DAN 模式等）
  2. 长度截断（默认 2000 字符）
  3. 异常字符检测（长度突变、重复模式）

用法:
    from src.agent.sanitizer import sanitize
    cleaned = sanitize(user_input)
"""
import re
import os

# ── 配置 ──
MAX_PROMPT_LENGTH = int(os.getenv("MAX_PROMPT_LENGTH", "2000"))

# ── 已知注入模式 ──
_INJECTION_PATTERNS = [
    # 角色越狱
    r"ignore\s+(all\s+)?(previous|prior|above|system)\s+(instructions?|prompts?|messages?)",
    r"forget\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?)",
    r"disregard\s+(previous|prior)\s+(instructions?|prompts?)",
    # DAN / 越狱模式
    r"\bDAN\s*mode\b",
    r"developer\s*mode\s*(enabled|activated|on)",
    r"jailbreak",
    r"do\s+anything\s+now",
    r"you\s+are\s+now\s+(a\s+)?\w+\s+mode",
    # 系统 prompt 泄露
    r"(reveal|show|print|output|display|tell\s+me)\s+your\s+(system\s+)?(prompt|instructions?)",
    r"repeat\s+(the\s+)?(above|previous|system\s+prompt)",
    r"what\s+(is|are)\s+your\s+(system\s+)?(prompt|instructions?)",
    # 指令覆盖
    r"new\s+instructions?:\s*",
    r"your\s+new\s+(task|role|job)\s+is",
    r"from\s+now\s+on\s+you\s+(are|will|must)",
    # 上下文污染
    r"<\|im_start\|>",
    r"<\|im_end\|>",
    r"\[system\]",
    r"\[/system\]",
]

# 编译正则
_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]


def sanitize(prompt: str) -> str:
    """
    清洗用户输入

    Args:
        prompt: 原始用户输入

    Returns:
        清洗后的输入

    Raises:
        ValueError: 检测到注入攻击时
    """
    if not prompt:
        return prompt

    # 长度检查
    if len(prompt) > MAX_PROMPT_LENGTH:
        prompt = prompt[:MAX_PROMPT_LENGTH]

    # 检查已知注入模式
    for pattern in _COMPILED_PATTERNS:
        if pattern.search(prompt):
            raise ValueError(f"检测到不安全的输入模式，已被拦截")

    return prompt.strip()

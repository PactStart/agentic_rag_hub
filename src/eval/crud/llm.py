"""DeepSeek chat 封装（CRUD 评测专用，不用企业拒答 SYSTEM）。"""

from __future__ import annotations

import os
import time
from typing import Any

from loguru import logger

from src.llm.generate import llm_client


def extract_response_tag(text: str) -> str:
    """官方解析：取最后一个 ``<response>...</response>``；无标签则全文。"""
    if not text:
        return ""
    if "<response>" in text:
        part = text.split("<response>")[-1]
        return part.split("</response>")[0].strip()
    return text.strip()


def deepseek_chat(
    user_prompt: str,
    *,
    temperature: float = 0.1,
    max_tokens: int | None = None,
    retries: int = 3,
    sleep_s: float = 1.0,
) -> str:
    cli = llm_client()
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
    kwargs: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "messages": [{"role": "user", "content": user_prompt}],
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            resp = cli.chat.completions.create(**kwargs)
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            wait = sleep_s * (attempt + 1)
            logger.warning("DeepSeek 调用失败 ({}/{}): {}；{}s 后重试", attempt + 1, retries, exc, wait)
            time.sleep(wait)
    logger.error("DeepSeek 放弃: {}", last_err)
    return ""

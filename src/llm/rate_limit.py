"""通用客户端限流：主动控 RPM/TPM + 被动消化 HTTP 429。

设计动机
--------
云厂商对「每个模型 / 每个接口」的配额通常不同，因此本模块：

- **不读环境变量**，不绑定硅基等具体厂商；
- 由调用方构造 ``RateBudget(key, rpm=..., tpm=...)`` 传入配额；
- ``key`` 建议写成 ``厂商:能力:模型``，不同 key 互不影响。

两种配额的处理方式为什么不一样
------------------------------
- **RPM**（Requests Per Minute）：调用前就能确定「这次算 1 次请求」，
  用 pyrate-limiter 在调用前 ``try_acquire(weight=1)`` 占坑即可。
- **TPM**（Tokens Per Minute）：调用前不知道真实消耗（各家返回字段也不同），
  因此采用「调用前只判断是否已触顶 → 发起请求 → 调用后按响应记账」。
  代价是：单次超大请求仍可能短暂超过阈值（软限制），靠 429 重试兜底。

典型用法（伪代码）::

    budget = RateBudget("siliconflow:rerank:bge", rpm=2000, tpm=500_000)
    budget.before_call()          # 可能阻塞
    resp = http_call(...)
    budget.record(actual_tokens)  # 用响应里的真实 token
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, TypeVar

import httpx
from pyrate_limiter import Duration, Limiter, Rate
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

T = TypeVar("T")


@lru_cache(maxsize=128)
def _rpm_limiter(bucket: str, capacity: int) -> Limiter:
    """按桶返回**进程内单例** RPM 限流器。

    为何需要 ``bucket`` 参数（即使构造 ``Limiter`` 时没用到它）
    ----------------------------------------------------------
    ``@lru_cache`` 用全部入参当缓存键。``bucket`` 不同 → 缓存未命中 →
    得到**另一个** ``Limiter`` 实例。这样 embed / rerank 等接口各算各的 RPM。

    不能删掉 ``bucket`` 只按 ``capacity`` 缓存：pyrate-limiter 在**同一个**
    ``Limiter`` 上，``try_acquire(name=...)`` 的 ``name`` **并不隔离配额**
    （实测 name=a 把额度用尽后，name=b 也会失败）。若只按 capacity 缓存，
    所有 ``rpm=2000`` 的预算会抢同一个桶。

    为何必须 ``lru_cache``
    ---------------------
    限流靠「跨多次调用累计已用量」。若每次 ``before_call`` 都 ``Limiter()``，
    计数永远从 0 开始，等于没限流。缓存保证同一 ``(bucket, capacity)``
    始终拿到同一对象。``maxsize=128`` 足够覆盖常见模型/接口组合。
    """
    # Rate(n, MINUTE)：滑动/漏桶语义由库实现；每分钟最多放行 n 次 weight 之和
    return Limiter(Rate(max(1, capacity), Duration.MINUTE))


class _TpmWindow:
    """进程内、按分钟滑动的 TPM 计数器（自研，不用 pyrate-limiter）。

    为何 TPM 不用 pyrate-limiter
    ---------------------------
    pyrate 适合「调用前已知 weight」。我们的 TPM 是调用后才知道实际 token，
    需要「先检查当前累计是否触顶，再事后 ``add``」，用简单的时间戳队列更直观。

    数据结构
    --------
    ``_events`` 为 ``deque[(monotonic_ts, tokens)]``：每次成功调用追加一条。
    ``_purge`` 丢掉超出 ``window_sec``（默认 60s）的旧事件，实现滑动窗口。

    并发
    ----
    ``threading.Lock`` 保护读改写；``wait_until_under_limit`` 在 lock 外
    ``sleep``，避免持锁睡眠堵死其它线程。
    """

    def __init__(self, limit: int, window_sec: float = 60.0) -> None:
        self.limit = max(1, limit)
        self.window_sec = window_sec
        self._events: deque[tuple[float, int]] = deque()
        self._lock = threading.Lock()

    def _purge(self, now: float) -> None:
        """剔除窗口外的事件；调用方须已持有 ``_lock``。"""
        while self._events and now - self._events[0][0] >= self.window_sec:
            self._events.popleft()

    def _used(self, now: float) -> int:
        """当前窗口内已记账的 token 总和；调用方须已持有 ``_lock``。"""
        self._purge(now)
        return sum(n for _, n in self._events)

    def wait_until_under_limit(self) -> None:
        """若已达/超过 TPM 阈值则阻塞，直到最老一笔用量滑出窗口。

        注意：这里只保证 ``used < limit`` 才放行，**不预留**本次即将消耗的
        token（未知）。因此并发下多请求同时通过检查后，事后累计仍可能短暂
        超过 ``limit``——这是刻意的软限制，服务端 429 + ``with_retry`` 兜底。
        """
        while True:
            with self._lock:
                now = time.monotonic()
                used = self._used(now)
                if used < self.limit:
                    return
                # 窗口非空时，睡到「最老事件刚好过期」再重试，避免忙等
                oldest_ts = self._events[0][0]
                sleep_for = self.window_sec - (now - oldest_ts) + 0.01
            time.sleep(max(0.01, sleep_for))

    def add(self, tokens: int) -> None:
        """把一次成功调用的实际消耗写入窗口（应在收到响应之后调用）。"""
        n = max(0, int(tokens))
        if n == 0:
            return
        with self._lock:
            now = time.monotonic()
            self._purge(now)
            self._events.append((now, n))


@lru_cache(maxsize=128)
def _tpm_window(bucket: str, capacity: int) -> _TpmWindow:
    """与 ``_rpm_limiter`` 同理：``bucket`` 分桶 + ``lru_cache`` 保单例。"""
    return _TpmWindow(capacity)


@dataclass(frozen=True)
class RateBudget:
    """某一模型或接口的速率预算（不可变配置对象）。

    实例本身只保存配额数字；真正的计数器在模块级 cache 里，按
    ``f\"{key}:rpm\"`` / ``f\"{key}:tpm\"`` 查找。因此：

    - 多次 ``RateBudget(同一 key, 同一 rpm)`` 会共享同一套计数；
    - ``frozen`` 只是防止改字段，不阻止共享底层 limiter。

    Attributes:
        key: 隔离名。不同 key → 不同桶。同厂商不同模型请用不同 key。
        rpm: 每分钟请求上限；``None`` 表示不限制请求次数。
        tpm: 每分钟 token 上限；``None`` 表示不限制 token（也不记账）。
        retry_max: 交给 ``with_retry`` 的最大尝试次数（含首次），默认 3。
    """

    key: str
    rpm: int | None = None
    tpm: int | None = None
    retry_max: int = 3

    def before_call(self) -> None:
        """发起上游 API **之前**调用。

        1. RPM：阻塞直到本分钟还能再发 1 次（``weight=1``）。
        2. TPM：若窗口内已记账量已触顶则阻塞等待滑窗；否则立即返回。

        与重试的关系：每次尝试（含 429 后的重试）都应再调一次本方法，
        因为每一次真实 HTTP 都算一次请求。失败（如 429）不应 ``record``。
        """
        if self.rpm is not None:
            # try_acquire 的 name 仍传入 key，便于库内日志/调试；
            # 真正的隔离靠上面 cache 出的不同 Limiter 实例。
            _rpm_limiter(f"{self.key}:rpm", self.rpm).try_acquire(
                f"{self.key}:rpm", weight=1, blocking=True
            )
        if self.tpm is not None:
            _tpm_window(f"{self.key}:tpm", self.tpm).wait_until_under_limit()

    def record(self, tokens: int) -> None:
        """上游调用**成功返回后**，把实际消耗的 token 记入 TPM 窗口。

        ``tokens`` 应由调用方从响应解析（例如 rerank 的
        ``meta.tokens.input_tokens + output_tokens``，或 embeddings 的
        ``usage.total_tokens``）。不要在此处再做字数估算。
        """
        if self.tpm is None:
            return
        _tpm_window(f"{self.key}:tpm", self.tpm).add(tokens)


def is_rate_limit_error(exc: BaseException) -> bool:
    """判断异常是否为「可重试的限流错误」。

    覆盖：
    - httpx：``raise_for_status()`` 抛出的 ``HTTPStatusError`` 且 status=429；
    - OpenAI SDK 等：带 ``status_code == 429`` 的异常；
    - 类名为 ``RateLimitError`` 的异常（避免强依赖 openai 包类型）。
    """
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 429
    if getattr(exc, "status_code", None) == 429:
        return True
    return exc.__class__.__name__ == "RateLimitError"


def with_retry(fn: Callable[[], T], *, max_attempts: int = 3) -> T:
    """对无参可调用对象做 429 专用重试。

    - 仅当 ``is_rate_limit_error`` 为真时重试，其它异常直接抛出；
    - ``wait_exponential_jitter``：指数退避并加抖动，降低惊群；
    - ``reraise=True``：用尽次数后抛出最后一次异常。

    ``fn`` 内部应自行调用 ``budget.before_call()``；成功路径再 ``record``。
    不要把 ``before_call`` 放到本函数外面只调一次，否则重试不会重新占 RPM。
    """

    @retry(
        retry=retry_if_exception(is_rate_limit_error),
        wait=wait_exponential_jitter(initial=1, max=60),
        stop=stop_after_attempt(max(1, max_attempts)),
        reraise=True,
    )
    def _wrapped() -> T:
        return fn()

    return _wrapped()

"""Token accounting for LLM calls and gateway quota lookup.

Every ``LLMProvider.chat_with_retry`` response is recorded in the process-wide
``meter``; the CLI prints the totals after a task and can query the API
gateway for the remaining quota.
"""

from __future__ import annotations

import json
import threading
import urllib.request
from typing import Any

_USAGE_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens")


class UsageMeter:
    """Thread-safe accumulator of LLM token usage, keyed by model."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_model: dict[str, dict[str, int]] = {}

    def record(self, model: str | None, usage: dict[str, Any] | None) -> None:
        if not usage:
            return
        with self._lock:
            entry = self._by_model.setdefault(
                model or "unknown", {"calls": 0, **{key: 0 for key in _USAGE_KEYS}}
            )
            entry["calls"] += 1
            for key in _USAGE_KEYS:
                entry[key] += int(usage.get(key) or 0)
            if not usage.get("total_tokens"):
                entry["total_tokens"] += int(usage.get("prompt_tokens") or 0) + int(
                    usage.get("completion_tokens") or 0
                )

    def reset(self) -> None:
        with self._lock:
            self._by_model.clear()

    def snapshot(self) -> dict[str, dict[str, int]]:
        with self._lock:
            return {model: dict(entry) for model, entry in self._by_model.items()}

    def totals(self) -> dict[str, int]:
        total = {"calls": 0, **{key: 0 for key in _USAGE_KEYS}}
        for entry in self.snapshot().values():
            for key in total:
                total[key] += entry[key]
        return total


meter = UsageMeter()


def fetch_gateway_quota(api_base: str | None, api_key: str | None, timeout: float = 10.0) -> dict[str, Any]:
    """Query a new-api style gateway for the token's quota (``GET /api/usage/token``).

    Returns ``{"granted", "used", "available", "unlimited"}``; raises on failure.
    """
    if not api_base or not api_key:
        raise RuntimeError("provider has no api_base / api_key")
    base = api_base.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    request = urllib.request.Request(
        base + "/api/usage/token",
        headers={"Authorization": f"Bearer {api_key}", "User-Agent": "PhyAgentOS"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    data = payload.get("data")
    if not payload.get("code") or not isinstance(data, dict):
        raise RuntimeError("unexpected quota response")
    return {
        "granted": data.get("total_granted"),
        "used": data.get("total_used"),
        "available": data.get("total_available"),
        "unlimited": bool(data.get("unlimited_quota")),
    }


def format_usage_report(
    snapshot: dict[str, dict[str, int]],
    quota_before: dict[str, Any] | None,
    quota_after: dict[str, Any] | None,
    quota_error: str | None = None,
) -> str:
    lines = ["Token 用量（本次任务，主 Agent 及其调用的所有模型）"]
    if not snapshot:
        lines.append("  没有记录到模型调用用量")
    for model, entry in snapshot.items():
        lines.append(
            f"  {model}: 调用 {entry['calls']} 次，输入 {entry['prompt_tokens']:,}，"
            f"输出 {entry['completion_tokens']:,}，合计 {entry['total_tokens']:,}"
        )
    if len(snapshot) > 1:
        total = sum(entry["total_tokens"] for entry in snapshot.values())
        lines.append(f"  合计 {total:,}")
    if quota_after is not None:
        if quota_after["unlimited"]:
            lines.append("  API 额度: 不限额")
        else:
            text = f"  API 剩余额度: {quota_after['available']:,} / 总额 {quota_after['granted']:,}"
            if quota_before is not None and not quota_before["unlimited"]:
                text += f"（网关记账本次消耗 {quota_before['available'] - quota_after['available']:,}）"
            lines.append(text)
    elif quota_error:
        lines.append(f"  API 剩余额度查询失败: {quota_error}")
    return "\n".join(lines)

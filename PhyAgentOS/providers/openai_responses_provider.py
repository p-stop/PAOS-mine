"""Direct provider for the OpenAI Responses API.

This adapter keeps PhyAgentOS' internal Chat Completions-shaped history while
using ``POST /responses`` on the wire.  In particular, it converts assistant
tool calls and tool results to Responses API ``function_call`` and
``function_call_output`` items.
"""

from __future__ import annotations

import hashlib
from typing import Any

import httpx
import json_repair

from PhyAgentOS.providers.base import LLMProvider, LLMResponse, ToolCallRequest


class OpenAIResponsesProvider(LLMProvider):
    """OpenAI-compatible Responses API provider, without LiteLLM translation."""

    _MAX_CALL_ID_LENGTH = 64

    @staticmethod
    def _normalize_api_base(api_base: str) -> str:
        """Return the API-version root used by OpenAI-compatible gateways.

        Users commonly configure a gateway origin (``https://gateway.example``),
        while the Responses endpoint is conventionally under ``/v1``.  Keep an
        already versioned base untouched and also accept a mistakenly supplied
        ``.../responses`` endpoint without producing ``/responses/responses``.
        """
        base = api_base.rstrip("/")
        if base.endswith("/responses"):
            base = base[: -len("/responses")]
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        return base

    def __init__(
        self,
        api_key: str,
        api_base: str | None = "https://api.openai.com/v1",
        default_model: str = "gpt-6-astra",
        extra_headers: dict[str, str] | None = None,
        timeout_s: float = 180.0,
    ):
        api_base = self._normalize_api_base(api_base or "https://api.openai.com/v1")
        super().__init__(api_key, api_base)
        self.default_model = default_model
        self.extra_headers = extra_headers or {}
        self._client = httpx.AsyncClient(
            base_url=api_base.rstrip("/"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                **self.extra_headers,
            },
            trust_env=False,
            timeout=httpx.Timeout(float(timeout_s), connect=15.0),
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=2),
        )

    @staticmethod
    def _message_content(content: Any) -> Any:
        """Translate common Chat Completions content blocks when necessary."""
        if not isinstance(content, list):
            return content
        converted: list[Any] = []
        for part in content:
            if not isinstance(part, dict):
                converted.append(part)
                continue
            item = dict(part)
            if item.get("type") == "text":
                item["type"] = "input_text"
            elif item.get("type") == "image_url":
                image_url = item.get("image_url")
                if isinstance(image_url, dict):
                    image_url = image_url.get("url")
                item = {"type": "input_image", "image_url": image_url}
            converted.append(item)
        return converted

    @classmethod
    def _wire_call_id(cls, call_id: Any) -> str:
        """Return a Responses-compatible, stable identifier for a tool call.

        The Responses API limits ``call_id`` to 64 characters.  Some upstream
        gateways issue longer IDs, which PhyAgentOS stores in its chat-shaped
        history and sends back on the next tool turn.  Deriving the compact ID
        from its original value (rather than using a per-request counter) makes
        the assistant ``function_call`` and its ``function_call_output`` agree
        even after session history is restored.
        """
        value = str(call_id or "")
        if len(value) <= cls._MAX_CALL_ID_LENGTH:
            return value
        digest_length = cls._MAX_CALL_ID_LENGTH - len("call_")
        return f"call_{hashlib.sha256(value.encode('utf-8')).hexdigest()[:digest_length]}"

    @classmethod
    def _convert_input(cls, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for message in cls._sanitize_empty_content(messages):
            role = message.get("role")
            content = message.get("content")

            if role == "tool":
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": cls._wire_call_id(message.get("tool_call_id")),
                        "output": content if isinstance(content, str) else str(content),
                    }
                )
                continue

            if content is not None:
                items.append(
                    {
                        "role": role,
                        "content": cls._message_content(content),
                    }
                )

            for tool_call in message.get("tool_calls") or []:
                function = tool_call.get("function") or {}
                arguments = function.get("arguments", "{}")
                if not isinstance(arguments, str):
                    import json

                    arguments = json.dumps(arguments, ensure_ascii=False)
                items.append(
                    {
                        "type": "function_call",
                        "call_id": cls._wire_call_id(tool_call.get("id")),
                        "name": function.get("name"),
                        "arguments": arguments,
                    }
                )
        return items

    @staticmethod
    def _convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        for tool in tools:
            if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
                function = tool["function"]
                item = {
                    "type": "function",
                    "name": function.get("name"),
                    "description": function.get("description", ""),
                    "parameters": function.get("parameters", {}),
                }
                if "strict" in function:
                    item["strict"] = function["strict"]
                converted.append(item)
            else:
                converted.append(dict(tool))
        return converted

    @staticmethod
    def _convert_tool_choice(tool_choice: str | dict[str, Any] | None) -> Any:
        if not isinstance(tool_choice, dict):
            return tool_choice
        function = tool_choice.get("function")
        if tool_choice.get("type") == "function" and isinstance(function, dict):
            return {"type": "function", "name": function.get("name")}
        return tool_choice

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        selected_model = model or self.default_model
        payload: dict[str, Any] = {
            "model": selected_model,
            "input": self._convert_input(messages),
            "max_output_tokens": max(1, max_tokens),
            "store": False,
        }
        # GPT-6 Astra does not accept Chat Completions sampling parameters.
        if "gpt-6" not in selected_model.lower() and "astra" not in selected_model.lower():
            payload["temperature"] = temperature
        if reasoning_effort and reasoning_effort != "none":
            payload["reasoning"] = {"effort": reasoning_effort}
        if tools:
            payload["tools"] = self._convert_tools(tools)
            payload["tool_choice"] = self._convert_tool_choice(tool_choice) or "auto"

        try:
            response = await self._client.post("/responses", json=payload)
            response.raise_for_status()
            return self._parse(response.json())
        except httpx.HTTPStatusError as exc:
            # httpx's default exception only reports the status and URL.  The
            # gateway response normally identifies the unsupported field, which
            # is essential for correcting a 400 configuration error.
            detail = exc.response.text.strip()
            if detail:
                detail = detail[:2_000]
                message = f"{exc}; gateway response: {detail}"
            else:
                message = str(exc)
            return LLMResponse(
                content=f"Error calling Responses API: {message}",
                finish_reason="error",
            )
        except Exception as exc:
            return LLMResponse(content=f"Error calling Responses API: {exc}", finish_reason="error")

    @staticmethod
    def _parse(response: dict[str, Any]) -> LLMResponse:
        text_parts: list[str] = []
        tool_calls: list[ToolCallRequest] = []
        reasoning_parts: list[str] = []

        for item in response.get("output") or []:
            item_type = item.get("type")
            if item_type == "message":
                for part in item.get("content") or []:
                    if part.get("type") == "output_text" and part.get("text"):
                        text_parts.append(part["text"])
            elif item_type == "function_call":
                arguments = item.get("arguments") or "{}"
                if isinstance(arguments, str):
                    arguments = json_repair.loads(arguments)
                tool_calls.append(
                    ToolCallRequest(
                        id=item.get("call_id") or item.get("id") or "",
                        name=item.get("name") or "",
                        arguments=arguments,
                    )
                )
            elif item_type == "reasoning":
                for summary in item.get("summary") or []:
                    if isinstance(summary, dict) and summary.get("text"):
                        reasoning_parts.append(summary["text"])

        usage = response.get("usage") or {}
        normalized_usage = {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }
        has_tool_calls = bool(tool_calls)
        status = response.get("status") or "completed"
        finish_reason = "tool_calls" if has_tool_calls else ("stop" if status == "completed" else status)
        return LLMResponse(
            content="".join(text_parts) or None,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=normalized_usage,
            reasoning_content="\n".join(reasoning_parts) or None,
        )

    def get_default_model(self) -> str:
        return self.default_model

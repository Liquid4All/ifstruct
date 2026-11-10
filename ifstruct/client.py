from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import requests


class Refusal(Exception):
    """Model refused to respond."""


@dataclass
class CompletionResult:
    text: str
    latency_ms: float
    finish_reason: str = ""


def _chat_completions_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    return normalized + "/chat/completions"


def _extract_message_text(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("API response missing choices")

    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ValueError("API response missing message")

    content = message.get("content")
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text" and isinstance(item.get("text"), str):
                text_parts.append(item["text"])
        if text_parts:
            return "".join(text_parts)

    finish = choices[0].get("finish_reason") or choices[0].get("native_finish_reason") or ""
    if finish == "refusal" or choices[0].get("native_finish_reason") == "refusal":
        refusal_text = message.get("refusal") or ""
        raise Refusal(f"Model refused to respond{': ' + refusal_text if refusal_text else ''}")

    raise ValueError("API response did not include string content")


def chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int = 8000,
    temperature: float = 0.0,
    timeout: float = 120.0,
    max_retries: int = 40,
    retry_delay: float = 30.0,
) -> CompletionResult:
    """Call an OpenAI-compatible chat completions endpoint."""
    url = _chat_completions_url(base_url)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            start = time.perf_counter()
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            latency_ms = (time.perf_counter() - start) * 1000.0
            response.raise_for_status()
            data = response.json()
            if "error" in data:
                raise ValueError(f"API error: {data['error']}")
            choice = data["choices"][0] if isinstance(data.get("choices"), list) and data["choices"] else {}
            return CompletionResult(
                text=_extract_message_text(data),
                latency_ms=latency_ms,
                finish_reason=str(choice.get("finish_reason") or ""),
            )
        except Refusal:
            raise
        except Exception as exc:  # pragma: no cover - network failure path
            last_error = exc
            if attempt < max_retries:
                print(f"  [retry {attempt + 1}/{max_retries}] {type(exc).__name__}: {str(exc)[:120]}", flush=True)
                time.sleep(retry_delay)

    assert last_error is not None
    raise last_error

from __future__ import annotations

import json
import logging
from typing import Any

import requests


LOGGER = logging.getLogger(__name__)


class OllamaClient:
    def __init__(self, base_url: str, timeout: float = 15.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def status(self) -> dict[str, Any]:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=3)
            response.raise_for_status()
            models = [item.get("name", "") for item in response.json().get("models", [])]
            return {"online": True, "models": models, "error": ""}
        except requests.RequestException as exc:
            return {"online": False, "models": [], "error": str(exc)}

    def embed(self, texts: list[str], model: str) -> list[list[float]]:
        if not model:
            raise ValueError("Embedding 模型名称为空")
        response = requests.post(
            f"{self.base_url}/api/embed", json={"model": model, "input": texts}, timeout=max(self.timeout, 120)
        )
        if response.status_code == 404:  # older Ollama fallback
            vectors = []
            for text in texts:
                old = requests.post(f"{self.base_url}/api/embeddings", json={"model": model, "prompt": text}, timeout=120)
                old.raise_for_status(); vectors.append(old.json()["embedding"])
            return vectors
        response.raise_for_status()
        return response.json().get("embeddings", [])

    def generate_json(self, model: str, prompt: str, retries: int = 2) -> dict[str, Any]:
        if not model:
            raise ValueError("LLM 模型名称为空")
        last_error: Exception | None = None
        current = prompt
        for _ in range(retries + 1):
            try:
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json={"model": model, "prompt": current, "stream": False, "format": "json"},
                    timeout=max(self.timeout, 180),
                )
                response.raise_for_status()
                content = response.json().get("response", "")
                parsed = parse_json_object(content)
                if not isinstance(parsed, dict):
                    raise ValueError("LLM 未返回 JSON 对象")
                return parsed
            except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                LOGGER.warning("Ollama JSON 请求失败：%s", exc)
                current = prompt + "\n上次输出不是有效 JSON。只返回一个 JSON 对象，不要 Markdown。"
        raise RuntimeError(f"本地 LLM 返回内容无法验证：{last_error}")


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(cleaned[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("JSON 根节点必须是对象")
    return value


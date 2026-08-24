from __future__ import annotations

import json
import logging
import mimetypes
import re
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

import requests

from app.integrations.settings import ServiceSettings, ensure_url_allowed
from app.llm.ollama_client import parse_json_object


LOGGER = logging.getLogger(__name__)


def _headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key or 'local-no-key'}", "Content-Type": "application/json"}


class LLMClient:
    def __init__(self, settings: ServiceSettings):
        self.settings = settings
        ensure_url_allowed(settings.llm_base_url, settings.allow_remote)

    def test(self) -> dict[str, Any]:
        model = self.settings.llm_model.strip()
        if not model:
            models = self.models()
            if not models:
                return {"ok": False, "detail": "LLM 服务连接正常，但服务端没有可用模型。请先安装或部署模型。"}
            model = models[0]
        client = self if model == self.settings.llm_model else LLMClient(replace(self.settings, llm_model=model))
        result = client.generate_json("只返回 JSON：{\"ok\":true}", retries=0)
        return {"ok": bool(result.get("ok")), "detail": f"模型 {model} 返回有效 JSON"}

    def models(self) -> list[str]:
        response = requests.get(f"{self.settings.llm_base_url}/models", headers=_headers(self.settings.llm_api_key), timeout=10)
        response.raise_for_status()
        data = response.json().get("data") or []
        return [str(item.get("id", "")) for item in data if item.get("id")]

    def generate_json(self, prompt: str, retries: int = 2) -> dict[str, Any]:
        if not self.settings.llm_model:
            raise ValueError("LLM 模型名称为空")
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                response = requests.post(
                    f"{self.settings.llm_base_url}/chat/completions",
                    headers=_headers(self.settings.llm_api_key),
                    json={
                        "model": self.settings.llm_model,
                        "temperature": self.settings.llm_temperature,
                        "messages": [{"role": "user", "content": prompt + ("\n只输出一个 JSON 对象。" if attempt else "")}],
                    }, timeout=180,
                )
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                return parse_json_object(content)
            except (requests.RequestException, KeyError, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                LOGGER.warning("LLM JSON 响应无效（第 %s 次）: %s", attempt + 1, type(exc).__name__)
        raise RuntimeError(f"LLM 未返回有效 JSON：{type(last_error).__name__ if last_error else '未知错误'}")


class EmbeddingClient:
    def __init__(self, settings: ServiceSettings):
        self.settings = settings
        ensure_url_allowed(settings.embedding_base_url, settings.allow_remote)

    def test(self) -> dict[str, Any]:
        vector = self.embed(["供应商质量文件连接测试"])[0]
        return {"ok": bool(vector), "detail": f"真实向量生成成功，维度 {len(vector)}"}

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not self.settings.embedding_model:
            raise ValueError("Embedding 模型名称为空")
        response = requests.post(
            f"{self.settings.embedding_base_url}/embeddings",
            headers=_headers(self.settings.embedding_api_key),
            json={"model": self.settings.embedding_model, "input": texts}, timeout=180,
        )
        response.raise_for_status()
        data = sorted(response.json().get("data", []), key=lambda item: item.get("index", 0))
        vectors = [item["embedding"] for item in data]
        if len(vectors) != len(texts):
            raise RuntimeError("Embedding 返回数量与输入不一致")
        if vectors and any(len(vector) != self.settings.embedding_dimensions for vector in vectors):
            actual = len(vectors[0])
            raise RuntimeError(f"Embedding 维度不匹配：配置 {self.settings.embedding_dimensions}，实际 {actual}")
        return vectors


class MinerUClient:
    def __init__(self, settings: ServiceSettings):
        self.settings = settings
        ensure_url_allowed(settings.ocr_base_url, settings.allow_remote)

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.settings.ocr_api_key}"} if self.settings.ocr_api_key else {}

    def test(self) -> dict[str, Any]:
        response = requests.get(f"{self.settings.ocr_base_url}/health", headers=self.headers, timeout=10)
        response.raise_for_status()
        return {"ok": True, "detail": f"OCR 服务正常（HTTP {response.status_code}）"}

    def ocr(self, path: Path, timeout_seconds: int = 7200) -> str:
        safe_name = _mineru_safe_name(path.name)
        with path.open("rb") as handle:
            response = requests.post(
                f"{self.settings.ocr_base_url}/tasks", headers=self.headers,
                files={"files": (safe_name, handle, mimetypes.guess_type(path.name)[0] or "application/octet-stream")},
                data={"backend": self.settings.ocr_backend, "lang_list": self.settings.ocr_lang, "return_md": "true"},
                timeout=600,
            )
        response.raise_for_status()
        task_id = response.json().get("task_id")
        if not task_id:
            raise RuntimeError("OCR 服务未返回 task_id")
        waited = 0
        while waited < timeout_seconds:
            state = requests.get(f"{self.settings.ocr_base_url}/tasks/{task_id}", headers=self.headers, timeout=30)
            state.raise_for_status()
            status = str(state.json().get("status", "unknown")).casefold()
            if status in {"completed", "success", "succeeded", "done", "finished"}:
                break
            if status in {"failed", "error"}:
                raise RuntimeError("OCR 任务失败")
            time.sleep(6); waited += 6
        else:
            raise TimeoutError("OCR 任务超时")
        result = requests.get(f"{self.settings.ocr_base_url}/tasks/{task_id}/result", headers=self.headers, timeout=120)
        result.raise_for_status()
        for payload in (result.json().get("results") or {}).values():
            markdown = payload.get("md_content") or payload.get("markdown") or payload.get("md")
            if markdown:
                return str(markdown)
        raise RuntimeError("OCR 结果中没有 Markdown 内容")


def _mineru_safe_name(name: str) -> str:
    suffix = Path(name).suffix.lower() or ".pdf"
    stem = re.sub(r"[^A-Za-z0-9._-]", "_", Path(name).stem).strip("._-") or "document"
    return f"{stem[:80]}_{uuid.uuid4().hex[:8]}{suffix}"

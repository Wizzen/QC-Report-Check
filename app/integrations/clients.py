from __future__ import annotations

import json
import logging
import mimetypes
import re
import time
import uuid
from pathlib import Path
from typing import Any

import requests

from app.integrations.settings import ServiceSettings, ensure_url_allowed
from app.llm.ollama_client import parse_json_object


LOGGER = logging.getLogger(__name__)


def _headers(api_key: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


class LLMClient:
    def __init__(self, settings: ServiceSettings):
        self.settings = settings
        ensure_url_allowed(settings.llm_base_url, settings.allow_remote)

    def ping(self) -> dict[str, Any]:
        try:
            response = requests.get(
                f"{self.settings.llm_base_url}/models",
                headers=_headers(self.settings.llm_api_key),
                timeout=10,
            )
            if response.status_code == 200:
                return {"ok": True, "detail": "连接成功"}
            if response.status_code == 404:
                return {"ok": True, "detail": "服务可达（无 /models 列表端点）"}
            if response.status_code in (401, 403):
                return {"ok": False, "detail": f"密钥无效或无权限（HTTP {response.status_code}）"}
            response.raise_for_status()
            return {"ok": False, "detail": f"HTTP {response.status_code}"}
        except requests.RequestException as exc:
            return {"ok": False, "detail": f"连接失败：{type(exc).__name__}"}

    def test(self) -> dict[str, Any]:
        ping = self.ping()
        if not ping["ok"]:
            return ping
        model = self.settings.llm_model.strip()
        if not model:
            return {"ok": True, "detail": "LLM 服务连接正常（未指定模型，将使用服务端默认模型）"}
        try:
            return self._probe_generation(model)
        except Exception as exc:
            if "timed out" in str(exc).casefold():
                return {"ok": False, "detail": f"服务可达且已发现模型 {model}，但最小对话 20 秒内未返回；模型可能仍在加载或推理队列繁忙"}
            return {"ok": False, "detail": str(exc)}

    def _probe_generation(self, model: str) -> dict[str, Any]:
        """Confirm that a model can generate without waiting for a full reasoning answer."""
        response = requests.post(
            f"{self.settings.llm_base_url}/chat/completions",
            headers=_headers(self.settings.llm_api_key),
            json={
                "model": model, "temperature": 0, "stream": False, "max_tokens": 16,
                "messages": [{"role": "user", "content": '只返回 JSON：{"ok":true}'}],
            },
            timeout=20,
        )
        response.raise_for_status()
        body = response.json()
        if isinstance(body.get("error"), dict):
            return {"ok": False, "detail": f"LLM 服务错误：{body['error'].get('message') or '未知错误'}"}
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            return {"ok": False, "detail": "LLM 响应缺少 choices 字段"}
        message = choices[0].get("message", {})
        content = str(message.get("content") or "").strip()
        if content:
            result = parse_json_object(content)
            return {"ok": bool(result.get("ok")), "detail": f"模型 {model} 返回有效 JSON"}
        reasoning = str(message.get("reasoning_content") or message.get("reasoning") or "").strip()
        if reasoning:
            return {"ok": True, "detail": f"模型 {model} 的思考模式响应正常（连接测试已收到推理 token，不等待最终答案）"}
        return {"ok": False, "detail": f"模型 {model} 返回空响应"}

    def generate_json(self, prompt: str, retries: int = 2, timeout_seconds: int = 180,
                      max_tokens: int | None = None) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                payload: dict[str, Any] = {
                    "temperature": self.settings.llm_temperature,
                    "stream": False,
                    "messages": [{"role": "user", "content": prompt + ("\n只输出一个 JSON 对象。" if attempt else "")}],
                }
                model = self.settings.llm_model.strip()
                if model:
                    payload["model"] = model
                if max_tokens is not None:
                    payload["max_tokens"] = max_tokens
                response = requests.post(
                    f"{self.settings.llm_base_url}/chat/completions",
                    headers=_headers(self.settings.llm_api_key),
                    json=payload,
                    timeout=timeout_seconds,
                )
                response.raise_for_status()
                body = response.json()
                if isinstance(body.get("error"), dict):
                    raise ValueError(f"LLM 服务错误：{body['error'].get('message') or '未知错误'}")
                choices = body.get("choices")
                if not isinstance(choices, list) or not choices:
                    fields = "、".join(sorted(str(key) for key in body))
                    raise ValueError(f"LLM 响应缺少 choices 字段（实际字段：{fields or '无'}）")
                message = choices[0].get("message", {})
                content = message.get("content")
                if content is None or not str(content).strip():
                    reasoning = message.get("reasoning_content") or message.get("reasoning")
                    if reasoning and str(reasoning).strip():
                        raise ValueError("LLM 只返回推理内容，未返回最终 JSON")
                    raise ValueError("LLM 返回空内容")
                return parse_json_object(str(content))
            except (requests.RequestException, KeyError, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                detail = type(exc).__name__
                if isinstance(exc, requests.HTTPError) and exc.response is not None:
                    body = exc.response.text.strip().replace("\n", " ")[:240]
                    detail = f"HTTP {exc.response.status_code}: {body or exc.response.reason}"
                LOGGER.warning("LLM JSON 响应无效（第 %s 次）: %s", attempt + 1, detail)
        if isinstance(last_error, Exception) and str(last_error):
            message = str(last_error)
        elif last_error is not None:
            message = type(last_error).__name__
        else:
            message = "未知错误"
        raise RuntimeError(f"LLM 未返回有效 JSON：{message}")


class EmbeddingClient:
    def __init__(self, settings: ServiceSettings):
        self.settings = settings
        ensure_url_allowed(settings.embedding_base_url, settings.allow_remote)

    def ping(self) -> dict[str, Any]:
        try:
            response = requests.get(
                f"{self.settings.embedding_base_url}/models",
                headers=_headers(self.settings.embedding_api_key),
                timeout=10,
            )
            if response.status_code == 200:
                return {"ok": True, "detail": "连接成功"}
            if response.status_code == 404:
                return {"ok": True, "detail": "服务可达（无 /models 列表端点）"}
            if response.status_code in (401, 403):
                return {"ok": False, "detail": f"密钥无效或无权限（HTTP {response.status_code}）"}
            response.raise_for_status()
            return {"ok": False, "detail": f"HTTP {response.status_code}"}
        except requests.RequestException as exc:
            return {"ok": False, "detail": f"连接失败：{type(exc).__name__}"}

    def test(self) -> dict[str, Any]:
        ping = self.ping()
        if not ping["ok"]:
            return ping
        if not self.settings.embedding_model.strip():
            return {"ok": True, "detail": "Embedding 服务连接正常（未指定模型）"}
        vector = self.embed(["供应商质量文件连接测试"], timeout_seconds=20)[0]
        return {"ok": bool(vector), "detail": f"真实向量生成成功，维度 {len(vector)}"}

    def embed(self, texts: list[str], timeout_seconds: int = 180) -> list[list[float]]:
        if not self.settings.embedding_model:
            raise ValueError("Embedding 模型名称为空")
        response = requests.post(
            f"{self.settings.embedding_base_url}/embeddings",
            headers=_headers(self.settings.embedding_api_key),
            json={"model": self.settings.embedding_model, "input": texts}, timeout=timeout_seconds,
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

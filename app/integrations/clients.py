from __future__ import annotations

import json
import logging
import mimetypes
import re
import time
import uuid
from threading import Lock
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import requests

from app.integrations.settings import ServiceSettings, ensure_url_allowed, is_remote_url
from app.llm.ollama_client import parse_json_object


LOGGER = logging.getLogger(__name__)


def _headers(api_key: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _network_options(url: str) -> dict[str, object]:
    """Keep macOS system proxies from intercepting loopback/LAN AI services."""
    if not is_remote_url(url):
        return {"proxies": {"http": "", "https": ""}}
    return {}


def _request_error(endpoint: str, exc: Exception) -> str:
    message = str(exc).strip().replace("\n", " ")
    return f"请求 {endpoint} 失败：{message[:240] or type(exc).__name__}"


class LLMClient:
    def __init__(self, settings: ServiceSettings):
        self.settings = settings
        self._resolved_model = settings.llm_model.strip()
        self._model_lock = Lock()
        self._generation_lock = Lock()
        self._force_serial_generation = False
        self._project_models: bool | None = None
        self._lm_studio_native: bool | None = None
        ensure_url_allowed(settings.llm_base_url, settings.allow_remote)

    def _native_base_url(self) -> str:
        """Return the server root behind an OpenAI-compatible ``/v1`` URL."""
        parsed = urlsplit(self.settings.llm_base_url.rstrip("/"))
        path = parsed.path.rstrip("/")
        if path.endswith("/v1"):
            path = path[:-3]
        return urlunsplit((parsed.scheme, parsed.netloc, path.rstrip("/"), "", ""))

    def _uses_lm_studio_native_api(self) -> bool:
        """Detect LM Studio once so provider-specific requests never reach other services."""
        if self._lm_studio_native is not None:
            return self._lm_studio_native
        endpoint = f"{self._native_base_url()}/api/v0/models"
        try:
            response = requests.get(
                endpoint, headers=_headers(self.settings.llm_api_key), timeout=3,
                **_network_options(endpoint),
            )
            if response.status_code != 200:
                self._lm_studio_native = False
                return False
            body = response.json()
            rows = body.get("data") if isinstance(body, dict) else None
            self._lm_studio_native = bool(
                isinstance(rows, list)
                and (body.get("object") == "list" or any(
                    isinstance(row, dict) and "compatibility_type" in row for row in rows
                ))
            )
        except (requests.RequestException, ValueError, TypeError):
            self._lm_studio_native = False
        return bool(self._lm_studio_native)

    def generation_concurrency_limit(self) -> int:
        """Return a safe generation limit for the configured local backend.

        The Project Models MLX proxy currently reports continuous batching but
        Qwen3.8 fails inside that path with ``tuple.shape`` when two generations
        overlap. Detect the proxy before submitting audit tasks so the first
        request is serialized instead of waiting for an HTTP 500 to downgrade.
        """
        if self._project_models is None:
            model_hint = self.settings.llm_model.casefold()
            # Limit probing to the affected model family. This avoids adding a
            # provider-detection request to unrelated OpenAI-compatible APIs.
            if "qwen3.8" not in model_hint:
                self._project_models = False
            else:
                endpoint = f"{self._native_base_url()}/health"
                try:
                    response = requests.get(
                        endpoint, headers=_headers(self.settings.llm_api_key), timeout=2,
                        **_network_options(endpoint),
                    )
                    body = response.json() if response.status_code == 200 else {}
                    self._project_models = bool(
                        isinstance(body, dict)
                        and "proxy_model" in body
                        and "continuous_batching_enabled" in body
                    )
                except (requests.RequestException, ValueError, TypeError):
                    # The exact affected Qwen3.8 configuration should fail safe
                    # even if /health is briefly unavailable during model load.
                    self._project_models = True
        if self._project_models:
            self._force_serial_generation = True
            return 1
        return 2

    def _generate_lm_studio_json(self, prompt: str, model: str, thinking: bool | str,
                                 timeout_seconds: int, max_tokens: int | None) -> dict[str, Any]:
        endpoint = f"{self._native_base_url()}/api/v1/chat"
        payload: dict[str, Any] = {
            "model": model,
            "input": prompt,
            "reasoning": thinking if isinstance(thinking, str) else ("on" if thinking else "off"),
            "temperature": self.settings.llm_temperature,
            "stream": False,
            "store": False,
        }
        if max_tokens is not None:
            payload["max_output_tokens"] = max_tokens
        response = requests.post(
            endpoint, headers=_headers(self.settings.llm_api_key), json=payload,
            timeout=timeout_seconds, **_network_options(endpoint),
        )
        response.raise_for_status()
        body = response.json()
        if isinstance(body.get("error"), dict):
            raise ValueError(f"LLM 服务错误：{body['error'].get('message') or '未知错误'}")
        output = body.get("output")
        if not isinstance(output, list):
            raise ValueError("LM Studio 响应缺少 output 字段")
        messages = [
            str(item.get("content") or "").strip()
            for item in output
            if isinstance(item, dict) and item.get("type") == "message"
        ]
        content = "\n".join(part for part in messages if part).strip()
        if not content:
            reasoning = any(
                isinstance(item, dict) and item.get("type") == "reasoning" and str(item.get("content") or "").strip()
                for item in output
            )
            if reasoning:
                raise ValueError("LLM 只返回推理内容，未返回最终 JSON")
            raise ValueError("LLM 返回空内容")
        return parse_json_object(content)

    def available_models(self) -> list[str]:
        response = requests.get(
            f"{self.settings.llm_base_url}/models", headers=_headers(self.settings.llm_api_key), timeout=10,
            **_network_options(self.settings.llm_base_url),
        )
        response.raise_for_status()
        body = response.json()
        rows = body.get("data") if isinstance(body, dict) else None
        if not isinstance(rows, list):
            return []
        return [str(row.get("id") or "").strip() for row in rows if isinstance(row, dict) and str(row.get("id") or "").strip()]

    def resolve_model(self) -> str:
        """Use the configured model or discover the first model exposed by /models once."""
        if self._resolved_model:
            return self._resolved_model
        with self._model_lock:
            if self._resolved_model:
                return self._resolved_model
            try:
                models = self.available_models()
            except requests.RequestException as exc:
                raise RuntimeError(f"LLM 模型名称为空，且无法读取 /models：{_request_error(self.settings.llm_base_url + '/models', exc)}") from exc
            except (ValueError, TypeError) as exc:
                raise RuntimeError("LLM 模型名称为空，且 /models 返回格式无效") from exc
            if not models:
                raise RuntimeError("LLM 模型名称为空，/models 也没有返回可用模型")
            self._resolved_model = models[0]
            return self._resolved_model

    def ping(self) -> dict[str, Any]:
        try:
            response = requests.get(
                f"{self.settings.llm_base_url}/models",
                headers=_headers(self.settings.llm_api_key),
                timeout=10,
                **_network_options(self.settings.llm_base_url),
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
            return {"ok": False, "detail": _request_error(f"{self.settings.llm_base_url}/models", exc)}

    def test(self) -> dict[str, Any]:
        ping = self.ping()
        if not ping["ok"]:
            return ping
        model = self.settings.llm_model.strip() or "尚未识别"
        try:
            model = self.resolve_model()
            return self._probe_generation(model)
        except Exception as exc:
            if "timed out" in str(exc).casefold():
                return {"ok": False, "detail": f"服务可达且已发现模型 {model}，但最小对话 20 秒内未返回；模型可能仍在加载或推理队列繁忙"}
            return {"ok": False, "detail": str(exc)}

    def _probe_generation(self, model: str) -> dict[str, Any]:
        """Confirm that a model can generate without waiting for a full reasoning answer."""
        if self._uses_lm_studio_native_api():
            result = self._generate_lm_studio_json(
                '只返回 JSON：{"ok":true}', model, False, timeout_seconds=20, max_tokens=16,
            )
            return {"ok": bool(result.get("ok")), "detail": f"模型 {model} 返回有效 JSON（LM Studio 快速模式）"}
        response = requests.post(
            f"{self.settings.llm_base_url}/chat/completions",
            headers=_headers(self.settings.llm_api_key),
            json={
                "model": model, "temperature": 0, "stream": False, "max_tokens": 16,
                "messages": [{"role": "user", "content": '只返回 JSON：{"ok":true}'}],
            },
            timeout=20,
            **_network_options(self.settings.llm_base_url),
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
                      max_tokens: int | None = None, thinking: bool | str | None = None) -> dict[str, Any]:
        last_error: Exception | None = None
        last_detail = ""
        attempt = 0
        max_attempts = retries + 1
        serial_recovery_added = False
        while attempt < max_attempts:
            try:
                model = self.resolve_model()
                request_prompt = prompt + ("\n只输出一个 JSON 对象。" if attempt else "")
                if thinking is not None and self._uses_lm_studio_native_api():
                    return self._generate_lm_studio_json(
                        request_prompt, model, thinking, timeout_seconds, max_tokens,
                    )
                payload: dict[str, Any] = {
                    "temperature": self.settings.llm_temperature,
                    "stream": False,
                    "messages": [{"role": "user", "content": request_prompt}],
                }
                payload["model"] = model
                # Qwen3/3.5 exposes a hard thinking switch through its chat
                # template. Avoid sending this provider-specific field to other
                # OpenAI-compatible models that may reject unknown parameters.
                if thinking is not None and re.search(r"qwen[\s._/-]*3", str(payload["model"]), re.IGNORECASE):
                    payload["chat_template_kwargs"] = {
                        "enable_thinking": thinking not in {False, "off"},
                    }
                if max_tokens is not None:
                    payload["max_tokens"] = max_tokens
                request_options = {
                    "headers": _headers(self.settings.llm_api_key),
                    "json": payload,
                    "timeout": timeout_seconds,
                    **_network_options(self.settings.llm_base_url),
                }
                endpoint = f"{self.settings.llm_base_url}/chat/completions"
                if self._force_serial_generation:
                    with self._generation_lock:
                        response = requests.post(endpoint, **request_options)
                else:
                    response = requests.post(endpoint, **request_options)
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
                detail = str(exc).strip() or type(exc).__name__
                if isinstance(exc, requests.HTTPError) and exc.response is not None:
                    body = exc.response.text.strip().replace("\n", " ")[:240]
                    detail = f"HTTP {exc.response.status_code}: {body or exc.response.reason}"
                    if exc.response.status_code == 500 and _is_project_models_batch_failure(body):
                        self._force_serial_generation = True
                        if not serial_recovery_added:
                            max_attempts += 1
                            serial_recovery_added = True
                        LOGGER.warning("Project Models 连续批处理异常，后续请求自动降为串行")
                LOGGER.warning("LLM JSON 响应无效（第 %s 次）: %s", attempt + 1, detail)
                last_detail = detail
                if isinstance(exc, requests.HTTPError) and exc.response is not None \
                        and exc.response.status_code in {400, 401, 403, 404, 422}:
                    break
            attempt += 1
        if last_detail:
            message = last_detail
        elif isinstance(last_error, Exception) and str(last_error):
            message = str(last_error)
        elif last_error is not None:
            message = type(last_error).__name__
        else:
            message = "未知错误"
        raise RuntimeError(f"LLM 未返回有效 JSON：{message}")


def _is_project_models_batch_failure(body: str) -> bool:
    normalized = body.casefold()
    return "tuple" in normalized and "shape" in normalized and "generation failed" in normalized


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
        if not self.settings.embedding_model.strip():
            return {"ok": False, "detail": "Embedding 模型名称为空，无法执行真实向量测试"}
        endpoint = f"{self.settings.embedding_base_url}/embeddings"
        try:
            vector = self.embed(["供应商质量文件连接测试"], timeout_seconds=20)[0]
            return {"ok": bool(vector), "detail": f"真实向量生成成功，维度 {len(vector)}（{endpoint}）"}
        except requests.RequestException as exc:
            return {"ok": False, "detail": _request_error(endpoint, exc)}
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            return {"ok": False, "detail": f"Embedding 响应无效（{endpoint}）：{str(exc)[:240]}"}

    def embed(self, texts: list[str], timeout_seconds: int = 180) -> list[list[float]]:
        if not self.settings.embedding_model:
            raise ValueError("Embedding 模型名称为空")
        response = requests.post(
            f"{self.settings.embedding_base_url}/embeddings",
            headers=_headers(self.settings.embedding_api_key),
            json={"model": self.settings.embedding_model, "input": texts}, timeout=timeout_seconds,
            **_network_options(self.settings.embedding_base_url),
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
        endpoint = f"{self.settings.ocr_base_url}/health"
        try:
            response = requests.get(endpoint, headers=self.headers, timeout=10,
                                    **_network_options(self.settings.ocr_base_url))
            response.raise_for_status()
            return {"ok": True, "detail": f"OCR 服务正常（HTTP {response.status_code}，{endpoint}）"}
        except requests.RequestException as exc:
            return {"ok": False, "detail": _request_error(endpoint, exc)}

    def ocr(self, path: Path, timeout_seconds: int = 7200) -> str:
        safe_name = _mineru_safe_name(path.name)
        with path.open("rb") as handle:
            response = requests.post(
                f"{self.settings.ocr_base_url}/tasks", headers=self.headers,
                files={"files": (safe_name, handle, mimetypes.guess_type(path.name)[0] or "application/octet-stream")},
                data={"backend": self.settings.ocr_backend, "lang_list": self.settings.ocr_lang, "return_md": "true"},
                timeout=600,
                **_network_options(self.settings.ocr_base_url),
            )
        response.raise_for_status()
        task_id = response.json().get("task_id")
        if not task_id:
            raise RuntimeError("OCR 服务未返回 task_id")
        waited = 0
        while waited < timeout_seconds:
            state = requests.get(f"{self.settings.ocr_base_url}/tasks/{task_id}", headers=self.headers, timeout=30,
                                 **_network_options(self.settings.ocr_base_url))
            state.raise_for_status()
            status = str(state.json().get("status", "unknown")).casefold()
            if status in {"completed", "success", "succeeded", "done", "finished"}:
                break
            if status in {"failed", "error"}:
                raise RuntimeError("OCR 任务失败")
            time.sleep(6); waited += 6
        else:
            raise TimeoutError("OCR 任务超时")
        result = requests.get(f"{self.settings.ocr_base_url}/tasks/{task_id}/result", headers=self.headers, timeout=120,
                              **_network_options(self.settings.ocr_base_url))
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

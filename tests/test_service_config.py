from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from app.database import ReviewDatabase
from app.integrations.clients import EmbeddingClient, LLMClient, MinerUClient
from app.integrations.settings import ConfigStore, is_remote_url, normalize_openai_url


def _store(tmp_path: Path) -> ConfigStore:
    return ConfigStore(ReviewDatabase(tmp_path / "review.db"), tmp_path / "service.key")


def test_ollama_url_is_normalized_to_openai_compatible_v1() -> None:
    assert normalize_openai_url("127.0.0.1:11434", ollama=True) == "http://127.0.0.1:11434/v1"
    assert normalize_openai_url("http://127.0.0.1:1234") == "http://127.0.0.1:1234"
    assert normalize_openai_url("http://127.0.0.1:8081", ollama=True) == "http://127.0.0.1:8081"
    assert normalize_openai_url("https://api.example.com/v1") == "https://api.example.com/v1"
    assert not is_remote_url("http://192.168.1.20:11434/v1")
    assert not is_remote_url("http://mineru-server.local:8888")
    assert is_remote_url("https://api.example.com/v1")


def test_remote_service_requires_explicit_opt_in(tmp_path: Path) -> None:
    store = _store(tmp_path)

    with pytest.raises(ValueError, match="公网服务地址"):
        store.save({"allow_remote": False, "llm_base_url": "https://api.example.com/v1"})

    saved = store.save({"allow_remote": True, "llm_base_url": "https://api.example.com/v1"})
    assert saved.allow_remote
    assert saved.uses_remote


def test_api_key_is_encrypted_and_masked_value_preserves_secret(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.save({"llm_api_key": "super-secret"})
    raw = store.db.one("SELECT llm_api_key FROM service_config WHERE id=1")["llm_api_key"]

    assert raw != "super-secret"
    assert "super-secret" not in raw
    assert store.get().llm_api_key == "super-secret"
    store.save({"llm_api_key": "••••••••"})
    assert store.get().llm_api_key == "super-secret"


def test_embedding_validates_the_configured_dimension(tmp_path: Path) -> None:
    store = _store(tmp_path)
    settings = store.save({"embedding_dimensions": 3, "embedding_model": "test-model"})
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"data": [{"index": 0, "embedding": [0.1, 0.2]}]}

    with patch("app.integrations.clients.requests.post", return_value=response):
        with pytest.raises(RuntimeError, match="维度不匹配"):
            EmbeddingClient(settings).embed(["test"])


def test_embedding_test_calls_real_endpoint_without_models_probe(tmp_path: Path) -> None:
    store = _store(tmp_path)
    settings = store.save({"embedding_base_url": "http://127.0.0.1:8081", "embedding_model": "embed",
                           "embedding_dimensions": 2})
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"data": [{"index": 0, "embedding": [0.1, 0.2]}]}

    with patch("app.integrations.clients.requests.get") as get, \
         patch("app.integrations.clients.requests.post", return_value=response) as post:
        result = EmbeddingClient(settings).test()

    assert result["ok"] is True
    get.assert_not_called()
    assert post.call_args.args[0] == "http://127.0.0.1:8081/embeddings"
    assert post.call_args.kwargs["proxies"] == {"http": "", "https": ""}


def test_ocr_local_health_bypasses_system_proxy_and_reports_endpoint(tmp_path: Path) -> None:
    settings = _store(tmp_path).save({"ocr_base_url": "http://mineru-server.local:8888"})
    response = Mock(status_code=200)
    response.raise_for_status.return_value = None

    with patch("app.integrations.clients.requests.get", return_value=response) as get:
        result = MinerUClient(settings).test()

    assert result["ok"] is True
    assert "http://mineru-server.local:8888/health" in result["detail"]
    assert get.call_args.kwargs["proxies"] == {"http": "", "https": ""}


def test_service_presets_keep_keys_encrypted_and_can_be_applied(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.save({"llm_model": "model-a", "llm_api_key": "preset-secret"})
    store.save_preset("llm", "local-a")
    raw = store.db.one("SELECT id,data FROM service_presets WHERE name='local-a'")

    assert "preset-secret" not in raw["data"]
    store.save({"llm_model": "model-b", "llm_api_key": "different"})
    restored = store.apply_preset(raw["id"])
    assert restored.llm_model == "model-a"
    assert restored.llm_api_key == "preset-secret"


def test_llm_ping_reports_reachable_service(tmp_path: Path) -> None:
    settings = _store(tmp_path).get()
    response = Mock()
    response.status_code = 200
    response.raise_for_status.return_value = None

    with patch("app.integrations.clients.requests.get", return_value=response):
        result = LLMClient(settings).ping()

    assert result["ok"] is True
    assert result["detail"] == "连接成功"


def test_llm_test_without_model_skips_json_probe(tmp_path: Path) -> None:
    settings = _store(tmp_path).save({"llm_model": ""})
    response = Mock()
    response.status_code = 200
    response.raise_for_status.return_value = None

    with patch("app.integrations.clients.requests.get", return_value=response):
        result = LLMClient(settings).test()

    assert result["ok"] is True
    assert "默认模型" in result["detail"]


def test_llm_json_request_uses_broad_openai_compatible_payload(tmp_path: Path) -> None:
    settings = _store(tmp_path).save({"llm_model": "qwen-test"})
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"choices": [{"message": {"content": '{"ok": true}'}}]}

    with patch("app.integrations.clients.requests.post", return_value=response) as request:
        result = LLMClient(settings).generate_json("test", retries=0)

    assert result == {"ok": True}
    payload = request.call_args.kwargs["json"]
    assert payload["model"] == "qwen-test"
    assert payload["stream"] is False
    assert "response_format" not in payload


def test_llm_connection_test_has_short_bounded_probe(tmp_path: Path) -> None:
    settings = _store(tmp_path).save({"llm_model": "slow-model"})
    client = LLMClient(settings)

    with patch.object(client, "ping", return_value={"ok": True, "detail": "连接成功"}), \
         patch("app.integrations.clients.requests.post", side_effect=RuntimeError("Read timed out")) as request:
        result = client.test()

    assert result["ok"] is False
    assert "20 秒" in result["detail"]
    assert "模型可能仍在加载" in result["detail"]
    assert request.call_args.kwargs["timeout"] == 20
    assert request.call_args.kwargs["json"]["max_tokens"] == 16


def test_llm_reports_reasoning_only_response_clearly(tmp_path: Path) -> None:
    settings = _store(tmp_path).save({"llm_model": "thinking-model"})
    client = LLMClient(settings)
    response = Mock()
    response.status_code = 200
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "choices": [{"message": {"content": "", "reasoning_content": "internal reasoning"}}]
    }

    with patch.object(client, "ping", return_value={"ok": True, "detail": "连接成功"}), \
         patch("app.integrations.clients.requests.post", return_value=response):
        result = client.test()

    assert result["ok"] is True
    assert "思考模式响应正常" in result["detail"]

    with patch("app.integrations.clients.requests.post", return_value=response):
        with pytest.raises(RuntimeError, match="只返回推理内容"):
            client.generate_json("test", retries=0, max_tokens=16)

from __future__ import annotations

import hashlib
import ipaddress
import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from cryptography.fernet import Fernet, InvalidToken

from app.database import ReviewDatabase
from app.database.v2 import utcnow


@dataclass(frozen=True)
class ServiceSettings:
    allow_remote: bool
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    llm_temperature: float
    llm_concurrency: int
    llm_timeout_seconds: int
    embedding_base_url: str
    embedding_api_key: str
    embedding_model: str
    embedding_dimensions: int
    ocr_base_url: str
    ocr_api_key: str
    ocr_backend: str
    ocr_lang: str
    chunk_size: int
    chunk_overlap: int
    top_k: int

    @property
    def embedding_fingerprint(self) -> str:
        raw = f"{self.embedding_base_url}|{self.embedding_model}|{self.embedding_dimensions}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]

    @property
    def uses_remote(self) -> bool:
        return any(is_remote_url(url) for url in (self.llm_base_url, self.ocr_base_url))


class KeyVault:
    def __init__(self, key_path: Path):
        key_path.parent.mkdir(parents=True, exist_ok=True)
        if not key_path.exists():
            key_path.write_bytes(Fernet.generate_key())
        self.fernet = Fernet(key_path.read_bytes().strip())

    def encrypt(self, value: str) -> str:
        return self.fernet.encrypt(value.encode("utf-8")).decode("ascii") if value else ""

    def decrypt(self, value: str) -> str:
        if not value:
            return ""
        try:
            return self.fernet.decrypt(value.encode("ascii")).decode("utf-8")
        except (InvalidToken, ValueError):
            return ""


class ConfigStore:
    SECRET_FIELDS = {"llm_api_key", "embedding_api_key", "ocr_api_key"}
    PRESET_FIELDS = {
        "llm": ("llm_base_url", "llm_api_key", "llm_model", "llm_temperature", "llm_concurrency", "llm_timeout_seconds"),
        "ocr": ("ocr_base_url", "ocr_api_key", "ocr_backend", "ocr_lang"),
    }

    def __init__(self, db: ReviewDatabase, key_path: Path):
        self.db = db
        self.vault = KeyVault(key_path)

    def get(self) -> ServiceSettings:
        row = self.db.one("SELECT * FROM service_config WHERE id=1")
        if not row:
            raise RuntimeError("服务配置尚未初始化")
        for field in self.SECRET_FIELDS:
            row[field] = self.vault.decrypt(row[field])
        return ServiceSettings(
            allow_remote=bool(row["allow_remote"]),
            llm_base_url=normalize_openai_url(row["llm_base_url"]),
            llm_api_key=row["llm_api_key"], llm_model=row["llm_model"],
            llm_temperature=float(row["llm_temperature"]),
            llm_concurrency=max(1, min(16, int(row.get("llm_concurrency") or 1))),
            llm_timeout_seconds=max(15, min(300, int(row.get("llm_timeout_seconds") or 300))),
            embedding_base_url=normalize_openai_url(row["embedding_base_url"], ollama=True),
            embedding_api_key=row["embedding_api_key"], embedding_model=row["embedding_model"],
            embedding_dimensions=int(row["embedding_dimensions"]),
            ocr_base_url=normalize_url(row["ocr_base_url"]), ocr_api_key=row["ocr_api_key"],
            ocr_backend=row["ocr_backend"], ocr_lang=row["ocr_lang"],
            chunk_size=int(row["chunk_size"]), chunk_overlap=int(row["chunk_overlap"]), top_k=int(row["top_k"]),
        )

    def save(self, values: dict[str, object]) -> ServiceSettings:
        current = self.db.one("SELECT * FROM service_config WHERE id=1") or {}
        allow_remote = bool(values.get("allow_remote", current.get("allow_remote", 0)))
        for field in ("llm_base_url", "embedding_base_url", "ocr_base_url"):
            if field in values:
                normalized = normalize_openai_url(str(values[field]), ollama=field == "embedding_base_url") if field != "ocr_base_url" else normalize_url(str(values[field]))
                ensure_url_allowed(normalized, allow_remote)
                values[field] = normalized
        for field in self.SECRET_FIELDS:
            supplied = values.get(field)
            if supplied is None or str(supplied).startswith("••••"):
                values[field] = current.get(field, "")
            else:
                values[field] = self.vault.encrypt(str(supplied))
        allowed = {
            "allow_remote", "llm_base_url", "llm_api_key", "llm_model", "llm_temperature",
            "llm_concurrency", "llm_timeout_seconds",
            "embedding_base_url", "embedding_api_key", "embedding_model", "embedding_dimensions",
            "ocr_base_url", "ocr_api_key", "ocr_backend", "ocr_lang", "chunk_size", "chunk_overlap", "top_k",
        }
        payload = {key: value for key, value in values.items() if key in allowed}
        payload["allow_remote"] = int(allow_remote)
        payload["updated_at"] = utcnow()
        columns = ",".join(f"{key}=?" for key in payload)
        self.db.execute(f"UPDATE service_config SET {columns} WHERE id=1", list(payload.values()))
        return self.get()

    def presets(self, category: str | None = None) -> list[dict[str, object]]:
        if category:
            return self.db.query("SELECT id,category,name FROM service_presets WHERE category=? ORDER BY name", (category,))
        return self.db.query("SELECT id,category,name FROM service_presets ORDER BY category,name")

    def save_preset(self, category: str, name: str) -> None:
        if category not in self.PRESET_FIELDS:
            raise ValueError("不支持的服务预设类型")
        name = name.strip()
        if not name:
            raise ValueError("预设名称不能为空")
        settings = self.get()
        data: dict[str, object] = {}
        for field in self.PRESET_FIELDS[category]:
            value = getattr(settings, field)
            data[field] = self.vault.encrypt(str(value)) if field in self.SECRET_FIELDS else value
        self.db.execute(
            """INSERT INTO service_presets(category,name,data,created_at) VALUES(?,?,?,?)
               ON CONFLICT(category,name) DO UPDATE SET data=excluded.data""",
            (category, name, json.dumps(data, ensure_ascii=False), utcnow()),
        )

    def apply_preset(self, preset_id: int) -> ServiceSettings:
        row = self.db.one("SELECT data FROM service_presets WHERE id=?", (preset_id,))
        if not row:
            raise ValueError("服务预设不存在")
        data = json.loads(row["data"] or "{}")
        for field in self.SECRET_FIELDS & data.keys():
            data[field] = self.vault.decrypt(str(data[field]))
        return self.save(data)


def normalize_url(value: str) -> str:
    value = (value or "").strip().replace("：//", "://")
    if not value:
        return ""
    if "://" not in value:
        value = "http://" + value
    parts = urlsplit(value)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ValueError("服务地址必须是有效的 HTTP/HTTPS URL")
    return urlunsplit((parts.scheme, parts.netloc, parts.path.rstrip("/"), "", ""))


def normalize_openai_url(value: str, *, ollama: bool = False) -> str:
    normalized = normalize_url(value)
    if not normalized:
        return ""
    parts = urlsplit(normalized)
    path = parts.path.rstrip("/")
    # Only Ollama's native 11434 endpoint is known to require /v1. Other
    # OpenAI-compatible services may expose /embeddings directly at root.
    if path in {"", "/"} and parts.port == 11434:
        path = "/v1"
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def is_remote_url(value: str) -> bool:
    if not value:
        return False
    host = (urlsplit(normalize_url(value)).hostname or "").casefold()
    if host in {"localhost", "host.docker.internal", "docker.for.mac.localhost"} or host.endswith((".local", ".lan")):
        return False
    try:
        address = ipaddress.ip_address(host)
        return not (address.is_loopback or address.is_private or address.is_link_local)
    except ValueError:
        return True


def ensure_url_allowed(value: str, allow_remote: bool) -> None:
    if is_remote_url(value) and not allow_remote:
        raise ValueError("检测到公网服务地址。请先明确启用“允许公网服务”。")


def mask_secret(value: str) -> str:
    return "" if not value else "••••••••"

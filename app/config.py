from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class OllamaConfig:
    base_url: str = "http://127.0.0.1:11434"
    llm_model: str = ""
    embedding_model: str = "bge-m3"


@dataclass(frozen=True)
class StorageConfig:
    database: Path
    database_v2: Path
    standards: Path
    uploads: Path
    vector_db: Path
    exports: Path


@dataclass(frozen=True)
class AuditConfig:
    confidence_threshold: float = 0.70
    max_upload_mb: int = 100


@dataclass(frozen=True)
class AppConfig:
    ollama: OllamaConfig
    storage: StorageConfig
    audit: AuditConfig


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


def load_config(path: Path | None = None) -> AppConfig:
    config_path = path or ROOT / "config.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle) or {}
    ollama = raw.get("ollama", {})
    storage = raw.get("storage", {})
    audit = raw.get("audit", {})
    result = AppConfig(
        ollama=OllamaConfig(
            base_url=str(ollama.get("base_url", "http://127.0.0.1:11434")).rstrip("/"),
            llm_model=str(ollama.get("llm_model", "")),
            embedding_model=str(ollama.get("embedding_model", "bge-m3")),
        ),
        storage=StorageConfig(
            database=_resolve(storage.get("database", "./data/database/app.db")),
            database_v2=_resolve(storage.get("database_v2", "./data/database/qaqc_v2.db")),
            standards=_resolve(storage.get("standards", "./data/standards")),
            uploads=_resolve(storage.get("uploads", "./data/uploads")),
            vector_db=_resolve(storage.get("vector_db", "./data/vector_db")),
            exports=_resolve(storage.get("exports", "./data/exports")),
        ),
        audit=AuditConfig(
            confidence_threshold=float(audit.get("confidence_threshold", 0.70)),
            max_upload_mb=int(audit.get("max_upload_mb", 100)),
        ),
    )
    ensure_directories(result)
    return result


def ensure_directories(config: AppConfig) -> None:
    for directory in (
        config.storage.database.parent,
        config.storage.standards,
        config.storage.uploads,
        config.storage.vector_db,
        config.storage.exports,
        ROOT / "logs",
    ):
        directory.mkdir(parents=True, exist_ok=True)


def save_ollama_config(base_url: str, llm_model: str, embedding_model: str) -> None:
    path = ROOT / "config.yaml"
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    raw["ollama"] = {
        "base_url": base_url.rstrip("/"),
        "llm_model": llm_model.strip(),
        "embedding_model": embedding_model.strip(),
    }
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, allow_unicode=True, sort_keys=False)

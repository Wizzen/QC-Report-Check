from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from app.auditing.v2_service import ReviewService
from app.config import ROOT, AppConfig, load_config
from app.database import ReviewDatabase
from app.integrations import ConfigStore


@dataclass(frozen=True)
class UIContext:
    config: AppConfig
    db: ReviewDatabase
    config_store: ConfigStore
    service: ReviewService


@st.cache_resource
def get_context() -> UIContext:
    config = load_config()
    db = ReviewDatabase(config.storage.database_v2)
    store = ConfigStore(db, ROOT / "data" / "secrets" / "service.key")
    service = ReviewService(db, store, config.storage.uploads, config.storage.standards,
                            config.storage.vector_db, config.audit.max_upload_mb)
    return UIContext(config, db, store, service)


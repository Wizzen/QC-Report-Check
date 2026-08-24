from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app.config import ROOT


def configure_logging() -> None:
    log_path = ROOT / "logs" / "app.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    if any(isinstance(item, RotatingFileHandler) for item in root.handlers):
        return
    root.setLevel(logging.INFO)
    handler = RotatingFileHandler(log_path, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s"))
    root.addHandler(handler)


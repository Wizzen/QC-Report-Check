from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import BinaryIO


ALLOWED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".txt", ".jpg", ".jpeg", ".png"}


def safe_filename(filename: str) -> str:
    """Return a basename safe for local storage; never trust an uploaded path."""
    name = Path(filename.replace("\\", "/")).name
    stem = re.sub(r"[^\w\-.()\u4e00-\u9fff ]+", "_", Path(name).stem, flags=re.UNICODE).strip(" ._")
    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError(f"不支持的文件类型：{suffix or '无扩展名'}")
    return f"{stem or 'document'}{suffix}"


def save_upload(file: BinaryIO, original_name: str, destination: Path, max_bytes: int) -> Path:
    filename = safe_filename(original_name)
    destination.mkdir(parents=True, exist_ok=True)
    data = file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError(f"文件超过大小限制（{max_bytes // 1024 // 1024} MB）")
    if not data:
        raise ValueError("文件内容为空")
    target = destination / f"{uuid.uuid4().hex}_{filename}"
    resolved = target.resolve()
    if destination.resolve() not in resolved.parents:
        raise ValueError("检测到不安全的文件路径")
    target.write_bytes(data)
    return target

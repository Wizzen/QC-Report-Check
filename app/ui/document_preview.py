from __future__ import annotations

import json
from pathlib import Path

import pymupdf
import streamlit as st

from app.evidence import render_pdf_evidence_bytes


@st.cache_data(max_entries=24, show_spinner=False)
def render_pdf_region(path_text: str, modified_ns: int, page_number: int, bbox: tuple) -> bytes:
    del modified_ns
    with pymupdf.open(path_text) as document:
        page = document[max(0, min(page_number-1, len(document)-1))]
        region = pymupdf.Rect(bbox) & page.rect
        if region.is_empty:
            region = page.rect
        return page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5), clip=region).tobytes('png')


def document_pages(page_text: str, raw_text: str) -> list[dict[str, object]]:
    try:
        pages = json.loads(page_text or "[]")
        if isinstance(pages, list) and pages:
            return [
                {"page": int(item.get("page", index + 1)), "text": str(item.get("text", ""))}
                for index, item in enumerate(pages)
                if isinstance(item, dict)
            ]
    except (TypeError, ValueError, json.JSONDecodeError):
        pass
    return [{"page": 1, "text": raw_text or "未提取到可浏览文本。"}]


@st.cache_data(max_entries=6, show_spinner=False)
def read_original_file(path_text: str, modified_ns: int) -> bytes:
    del modified_ns  # Included only to invalidate the cache when the file changes.
    return Path(path_text).read_bytes()


@st.cache_data(max_entries=12, show_spinner=False)
def render_pdf_page(path_text: str, modified_ns: int, page_number: int) -> bytes:
    del modified_ns
    with pymupdf.open(path_text) as document:
        page_index = max(0, min(page_number - 1, document.page_count - 1))
        pixmap = document.load_page(page_index).get_pixmap(matrix=pymupdf.Matrix(1.35, 1.35), alpha=False)
        return pixmap.tobytes("png")


@st.cache_data(max_entries=24, show_spinner=False)
def render_pdf_evidence(
    path_text: str, modified_ns: int, page_number: int, source_text: str, actual: str, item: str,
    bbox: tuple[float, float, float, float] | None = None,
) -> tuple[bytes, bool]:
    """Render a focused PDF screenshot and outline the matched evidence when possible."""
    del modified_ns
    return render_pdf_evidence_bytes(path_text, page_number, source_text, actual, item, bbox)

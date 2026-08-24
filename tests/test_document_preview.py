from __future__ import annotations

from pathlib import Path

import pymupdf

from app.ui.document_preview import document_pages, read_original_file, render_pdf_page


def test_document_pages_uses_page_aware_text_and_fallback() -> None:
    pages = document_pages('[{"page": 2, "text": "report text"}]', "fallback")
    assert pages == [{"page": 2, "text": "report text"}]
    assert document_pages("invalid", "fallback") == [{"page": 1, "text": "fallback"}]


def test_original_download_and_pdf_render(tmp_path: Path) -> None:
    pdf_path = tmp_path / "report.pdf"
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "Supplier report")
    document.save(pdf_path)
    document.close()

    modified = pdf_path.stat().st_mtime_ns
    assert read_original_file(str(pdf_path), modified).startswith(b"%PDF")
    assert render_pdf_page(str(pdf_path), modified, 1).startswith(b"\x89PNG")


def test_macos_launcher_is_self_contained() -> None:
    script = (Path(__file__).resolve().parents[1] / "start_macos.command").read_text(encoding="utf-8")
    assert script.startswith("#!/bin/bash")
    assert ".venv-macos" in script
    assert "pip install -r requirements.txt" in script
    assert "launcher.py" in script

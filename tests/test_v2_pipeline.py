from __future__ import annotations

from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pymupdf
import pytest
from docx import Document

from app.auditing.v2_service import ReviewService
from app.database import ReviewDatabase
from app.integrations import ConfigStore
from app.models import Finding


class NamedBytesIO(BytesIO):
    def __init__(self, data: bytes, name: str):
        super().__init__(data)
        self.name = name


def _pdf(text: str, name: str) -> NamedBytesIO:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), text, fontname="china-s")
    payload = document.tobytes()
    document.close()
    return NamedBytesIO(payload, name)


def _docx(text: str, name: str) -> NamedBytesIO:
    document = Document()
    document.add_paragraph(text)
    payload = BytesIO()
    document.save(payload)
    return NamedBytesIO(payload.getvalue(), name)


def _service(tmp_path: Path) -> ReviewService:
    db = ReviewDatabase(tmp_path / "review.db")
    store = ConfigStore(db, tmp_path / "service.key")
    return ReviewService(db, store, tmp_path / "uploads", tmp_path / "standards", tmp_path / "vectors")


def test_review_requires_supplier_file_before_creating_batch(tmp_path: Path) -> None:
    service = _service(tmp_path)

    with pytest.raises(ValueError, match="至少需要"):
        service.create_review(None, [], [], [])

    assert service.db.one("SELECT COUNT(*) count FROM review_batches")["count"] == 0


def test_full_local_rule_path_uses_basis_not_supplier_archive(tmp_path: Path) -> None:
    service = _service(tmp_path)
    basis = NamedBytesIO("抗拉强度 >= 500 MPa".encode("utf-8"), "采购技术要求.txt")
    with patch("app.auditing.v2_service.DocumentVectorIndex.index", return_value=(0, "embedding_failed")):
        basis_id = service.import_basis(basis, "technical")
        supplier = _docx(
            "Material Test Certificate  Supplier Quality Record  "
            "Heat No: H20260824  Material Grade: Q355B  Rm: 480 MPa  "
            "This certificate contains verified inspection results.",
            "supplier.docx",
        )
        batch_id = service.create_review(None, [basis_id], [supplier], [])
        service.process_batch(batch_id)

    batch = service.db.one("SELECT * FROM review_batches WHERE id=?", (batch_id,))
    findings = service.db.query("SELECT * FROM findings WHERE batch_id=?", (batch_id,))
    assert batch and batch["status"] == "completed"
    assert any(row["item"] == "抗拉强度" and row["severity"] == "Major" for row in findings)
    selected_basis = service.db.query(
        "SELECT d.library_code FROM batch_documents bd JOIN documents d ON d.id=bd.document_id "
        "WHERE bd.batch_id=? AND bd.role IN ('selected_basis','supplemental_basis')",
        (batch_id,),
    )
    assert selected_basis and {row["library_code"] for row in selected_basis} == {"basis"}


def test_llm_explanations_are_batched_instead_of_one_call_per_finding(tmp_path: Path) -> None:
    service = _service(tmp_path)
    service.config_store.save({"llm_model": "test-model"})
    batch_id = service.db.create_batch(None)
    findings = [Finding("数值不符合", "Major", f"item-{index}", "old", actual="1", requirement=">=2")
                for index in range(3)]
    response = {"items": [
        {"index": index, "reason": f"reason-{index}", "suggestion": f"fix-{index}", "confidence": 0.9}
        for index in range(3)
    ]}

    with patch("app.auditing.v2_service.LLMClient.generate_json", return_value=response) as generate:
        result = service._explain(batch_id, findings)

    assert generate.call_count == 1
    assert [item.description for item in result] == ["reason-0", "reason-1", "reason-2"]
    batch = service.db.one("SELECT activity,resource,heartbeat_at FROM review_batches WHERE id=?", (batch_id,))
    assert "批量解释" in batch["activity"]
    assert "LLM" in batch["resource"]
    assert batch["heartbeat_at"]

from __future__ import annotations

import hashlib
from pathlib import Path

import pymupdf

from app.database import ReviewDatabase
from app.database.v2 import utcnow
from app.exporters import export_batch_pdf


def test_pdf_report_contains_summary_details_and_only_matched_screenshot(tmp_path: Path) -> None:
    source_path = tmp_path / "supplier.pdf"
    source_pdf = pymupdf.open()
    page = source_pdf.new_page()
    page.insert_text((72, 120), "Inspection Items  Head Height  Standard 10  Result 9  Sample 20  Pass 19")
    source_pdf.save(source_path)
    source_pdf.close()

    db = ReviewDatabase(tmp_path / "review.db")
    template_id = db.one("SELECT id FROM audit_templates WHERE is_default=1")["id"]
    batch_id = db.create_batch(template_id)
    db.execute("UPDATE review_batches SET supplier_name=?,status='completed' WHERE id=?", ("ACME FASTENERS CO.,LTD", batch_id))
    document_id = db.add_document(
        library="supplier", kind="supplier", original_name="supplier.pdf", stored_path=str(source_path),
        sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(), mime_type="application/pdf",
    )
    db.execute("UPDATE documents SET supplier_name=?,page_count=1,parse_status='completed',ocr_status='not_needed' WHERE id=?",
               ("ACME FASTENERS CO.,LTD", document_id))
    db.attach_document(batch_id, document_id, "supplier", 4)
    for item, evidence in (("Head Height", "Head Height"), ("Missing COC", "batch file list only")):
        db.execute(
            """INSERT INTO findings(batch_id,category,severity,item,description,actual,requirement,source_file,
               source_page,source_text,logic,suggestion,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (batch_id, "检测结果不合格", "Major", item, "发现审核问题", "9", "应符合报告要求",
             "supplier.pdf", 1, evidence, "规则比较", "请供应商整改", utcnow()),
        )

    payload = export_batch_pdf(db, batch_id)

    assert payload.startswith(b"%PDF")
    report = pymupdf.open(stream=payload, filetype="pdf")
    text = "\n".join(page.get_text() for page in report)
    assert "SUPPLIER QUALITY REVIEW" in text
    assert "ACME FASTENERS CO.,LTD" in text
    assert "Head Height" in text and "Missing COC" in text
    assert sum(len(page.get_images(full=True)) for page in report) == 1
    report.close()

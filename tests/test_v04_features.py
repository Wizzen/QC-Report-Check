from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pytest

from app.database import ReviewDatabase
from app.models import ExtractedItem, PageText
from app.rules.generic import GROUPS, GenericDocument, classify_document, run_generic_rules, seed_generic_rules


def test_all_requested_generic_rules_are_versioned(tmp_path: Path) -> None:
    db = ReviewDatabase(tmp_path / "review.db")
    assert seed_generic_rules(db) == sum(len(group) for group in GROUPS.values())
    expected = {code for group in GROUPS.values() for code in group}
    assert {row["code"] for row in db.query("SELECT code FROM audit_rules")} == expected
    assert db.one("SELECT COUNT(*) count FROM audit_rule_versions")["count"] == len(expected)
    template_id = db.one("SELECT id FROM audit_templates WHERE name='通用材料质量审核'")["id"]
    batch_id = db.create_batch(template_id)
    snapshot = json.loads(db.one("SELECT template_snapshot FROM review_batches WHERE id=?", (batch_id,))["template_snapshot"])
    assert {row["rule_code"] for row in snapshot["rules"]} == expected


def test_unknown_document_does_not_emit_blanket_missing_fields() -> None:
    pages = [PageText(1, "Packing list for shipment 123")]
    kind, confidence = classify_document("packing.pdf", pages)
    doc = GenericDocument("d1", "packing.pdf", Path("packing.pdf"), pages, [], kind, confidence)

    findings = run_generic_rules([doc])

    assert [item.rule_code for item in findings] == ["DOC-TYPE"]


def test_generic_rules_find_page_date_and_mechanical_conflicts() -> None:
    pages = [PageText(1, """Material Test Report
Page 1 of 3
Report No: R-1
Manufacturer: Example Steel Co., Ltd
Material Grade: Q355B
Heat No: H01
Specification: 10 mm
Standard: GB/T 1
Manufacturing Date: 2026-08-20
Test Date: 2026-08-19
Issue Date: 2026-08-18
Overall Result: PASS
""")]
    fields = [
        ExtractedItem("抗拉强度", "300 MPa", 300.0, "MPa", 1, "Tensile Strength: 300 MPa", "measurement"),
        ExtractedItem("屈服强度", "350 MPa", 350.0, "MPa", 1, "Yield Strength: 350 MPa", "measurement"),
    ]
    doc = GenericDocument("d1", "mtr.pdf", Path("mtr.pdf"), pages, fields, "MTR", .95, "Example Steel Co., Ltd")

    codes = {item.rule_code for item in run_generic_rules([doc])}

    assert {"DOC-003", "DATE-001", "DATE-002", "MEC-001"} <= codes


def test_cancel_queued_job_is_immediate(tmp_path: Path) -> None:
    db = ReviewDatabase(tmp_path / "review.db")
    batch_id = db.create_batch(None)

    assert db.request_cancel(batch_id)
    assert db.one("SELECT status FROM review_batches WHERE id=?", (batch_id,))["status"] == "cancelled"
    assert db.claim_job() is None


def test_feedback_is_persisted_with_rule_version(tmp_path: Path) -> None:
    db = ReviewDatabase(tmp_path / "review.db")
    batch_id = db.create_batch(None)
    finding_id = db.execute(
        """INSERT INTO findings(batch_id,category,severity,item,description,rule_code,rule_version,created_at)
           VALUES(?,?,?,?,?,?,?,datetime('now'))""",
        (batch_id, "规则不符合", "Major", "Test", "Mismatch", "RES-001", 1),
    )

    db.update_finding_status(finding_id, "人工驳回", note="OCR 错列", correction="实际为 500 MPa")

    feedback = db.one("SELECT * FROM review_feedback WHERE finding_id=?", (finding_id,))
    assert feedback and feedback["action"] == "人工驳回" and feedback["rule_code"] == "RES-001"


def test_trash_can_restore_and_enforces_retention(tmp_path: Path) -> None:
    db = ReviewDatabase(tmp_path / "review.db")
    batch_id = db.create_batch(None)
    db.soft_delete_batch(batch_id)
    assert db.one("SELECT deleted_at FROM review_batches WHERE id=?", (batch_id,))["deleted_at"]
    with pytest.raises(ValueError, match="保留期"):
        db.purge_batch(batch_id)
    db.restore_batch(batch_id)
    assert db.one("SELECT deleted_at FROM review_batches WHERE id=?", (batch_id,))["deleted_at"] == ""

    db.soft_delete_batch(batch_id)
    expired = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(timespec="seconds")
    db.update_batch(batch_id, purge_after=expired)
    db.purge_batch(batch_id)
    assert db.one("SELECT id FROM review_batches WHERE id=?", (batch_id,)) is None

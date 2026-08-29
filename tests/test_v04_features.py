from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.database import ReviewDatabase


def test_withdrawn_versioned_rule_tables_are_removed(tmp_path: Path) -> None:
    db = ReviewDatabase(tmp_path / "review.db")
    names = {row["name"] for row in db.query("SELECT name FROM sqlite_master WHERE type='table'")}
    assert not {"audit_rules", "audit_rule_versions", "template_rule_versions"} & names


def test_windows_launcher_does_not_require_removed_vector_dependency() -> None:
    launcher = (Path(__file__).parents[1] / "start.bat").read_text(encoding="utf-8")

    assert "chromadb" not in launcher.casefold()


def test_cancel_queued_job_is_immediate(tmp_path: Path) -> None:
    db = ReviewDatabase(tmp_path / "review.db")
    batch_id = db.create_batch(None)

    assert db.request_cancel(batch_id)
    assert db.one("SELECT status FROM review_batches WHERE id=?", (batch_id,))["status"] == "cancelled"
    assert db.claim_job() is None


def test_feedback_is_persisted(tmp_path: Path) -> None:
    db = ReviewDatabase(tmp_path / "review.db")
    batch_id = db.create_batch(None)
    finding_id = db.execute(
        """INSERT INTO findings(batch_id,category,severity,item,description,rule_code,rule_version,created_at)
           VALUES(?,?,?,?,?,?,?,datetime('now'))""",
        (batch_id, "规则不符合", "Major", "Test", "Mismatch", "RES-001", 1),
    )

    db.update_finding_status(finding_id, "人工驳回", note="OCR 错列", correction="实际为 500 MPa")

    feedback = db.one("SELECT * FROM review_feedback WHERE finding_id=?", (finding_id,))
    assert feedback and feedback["action"] == "人工驳回"


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

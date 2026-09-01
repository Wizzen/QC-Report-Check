from __future__ import annotations

from pathlib import Path

import pytest

from app.database import ReviewDatabase
from app.ui.pages import _delete_template


def test_v2_initializes_only_the_two_fixed_libraries(tmp_path: Path) -> None:
    db = ReviewDatabase(tmp_path / "review.db")

    libraries = db.query("SELECT code,name FROM document_libraries ORDER BY code")

    assert [row["code"] for row in libraries] == ["basis", "supplier"]
    assert db.one("SELECT name FROM audit_templates WHERE is_default=1") is not None


def test_batch_is_created_automatically_with_persistent_job(tmp_path: Path) -> None:
    db = ReviewDatabase(tmp_path / "review.db")
    template_id = db.one("SELECT id FROM audit_templates WHERE is_default=1")["id"]

    batch_id = db.create_batch(template_id)

    batch = db.one("SELECT * FROM review_batches WHERE id=?", (batch_id,))
    job = db.one("SELECT * FROM jobs WHERE batch_id=?", (batch_id,))
    assert batch and batch["name"].startswith("审核批次 ")
    assert batch["status"] == "queued"
    assert job and job["status"] == "queued"


def test_template_delete_preserves_existing_batch_snapshot_and_protects_default(tmp_path: Path) -> None:
    db = ReviewDatabase(tmp_path / "review.db")
    default_id = db.one("SELECT id FROM audit_templates WHERE is_default=1")["id"]
    template_id = db.execute(
        """INSERT INTO audit_templates(name,required_items,enabled,is_default,created_at)
           VALUES(?, '[]', 1, 0, datetime('now'))""",
        ("临时模板",),
    )
    batch_id = db.create_batch(template_id)

    db.delete_template(template_id)

    batch = db.one("SELECT template_id,template_snapshot FROM review_batches WHERE id=?", (batch_id,))
    assert db.one("SELECT id FROM audit_templates WHERE id=?", (template_id,)) is None
    assert batch["template_id"] is None
    assert "临时模板" in batch["template_snapshot"]
    with pytest.raises(ValueError, match="默认模板"):
        db.delete_template(default_id)


def test_template_delete_supports_database_object_cached_before_method_was_added(tmp_path: Path) -> None:
    db = ReviewDatabase(tmp_path / "review.db")
    template_id = db.execute(
        """INSERT INTO audit_templates(name,required_items,enabled,is_default,created_at)
           VALUES(?, '[]', 1, 0, datetime('now'))""",
        ("旧缓存模板",),
    )

    class CachedDatabase:
        """Represents the pre-update object kept by Streamlit cache_resource."""

        one = db.one
        execute = db.execute

    _delete_template(CachedDatabase(), template_id)

    assert db.one("SELECT id FROM audit_templates WHERE id=?", (template_id,)) is None


def test_claim_job_is_atomic_for_single_worker(tmp_path: Path) -> None:
    db = ReviewDatabase(tmp_path / "review.db")
    db.create_batch(None)

    first = db.claim_job()
    second = db.claim_job()

    assert first and first["status"] == "running"
    assert second is None


def test_interrupted_job_is_requeued_with_visible_activity(tmp_path: Path) -> None:
    db = ReviewDatabase(tmp_path / "review.db")
    batch_id = db.create_batch(None)
    assert db.claim_job() is not None

    assert db.requeue_running_jobs() == 1

    batch = db.one("SELECT * FROM review_batches WHERE id=?", (batch_id,))
    job = db.one("SELECT * FROM jobs WHERE batch_id=?", (batch_id,))
    assert batch and batch["stage"] == "恢复审核"
    assert "重新进入队列" in batch["activity"]
    assert batch["heartbeat_at"]
    assert job and job["status"] == "queued"


def test_cancel_requested_job_is_finished_on_worker_restart(tmp_path: Path) -> None:
    db = ReviewDatabase(tmp_path / "review.db")
    batch_id = db.create_batch(None)
    assert db.claim_job() is not None
    assert db.request_cancel(batch_id)

    visible = db.one("SELECT status,stage FROM review_batches WHERE id=?", (batch_id,))
    assert visible == {"status": "cancelled", "stage": "已取消"}

    assert db.requeue_running_jobs() == 0

    batch = db.one("SELECT status,stage FROM review_batches WHERE id=?", (batch_id,))
    job = db.one("SELECT status FROM jobs WHERE batch_id=?", (batch_id,))
    assert batch == {"status": "cancelled", "stage": "已取消"}
    assert job == {"status": "cancelled"}

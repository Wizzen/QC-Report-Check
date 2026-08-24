from __future__ import annotations

from pathlib import Path

from app.database import ReviewDatabase


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

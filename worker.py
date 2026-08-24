from __future__ import annotations

import argparse
import time

from app.auditing.v2_service import ReviewService
from app.config import ROOT, load_config
from app.database import ReviewDatabase
from app.integrations import ConfigStore
from app.logging_config import configure_logging


def build_service() -> tuple[ReviewDatabase, ReviewService]:
    config = load_config()
    db = ReviewDatabase(config.storage.database_v2)
    store = ConfigStore(db, ROOT / "data" / "secrets" / "service.key")
    return db, ReviewService(db, store, config.storage.uploads, config.storage.standards,
                             config.storage.vector_db, config.audit.max_upload_mb)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    configure_logging()
    db, service = build_service()
    db.requeue_running_jobs()
    while True:
        job = db.claim_job()
        if job:
            service.process_batch(job["batch_id"])
        elif args.once:
            return 0
        else:
            time.sleep(1.5)


if __name__ == "__main__":
    raise SystemExit(main())

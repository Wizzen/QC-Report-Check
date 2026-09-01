from __future__ import annotations

import json
import sqlite3
import shutil
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence


SCHEMA_V2 = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS document_libraries (
  code TEXT PRIMARY KEY CHECK(code IN ('basis','supplier')),
  name TEXT NOT NULL, description TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS audit_templates (
  id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL UNIQUE,
  description TEXT NOT NULL DEFAULT '', required_document_types TEXT NOT NULL DEFAULT '[]',
  required_items TEXT NOT NULL DEFAULT '[]', review_instructions TEXT NOT NULL DEFAULT '', enabled INTEGER NOT NULL DEFAULT 1,
  is_default INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS review_batches (
  id TEXT PRIMARY KEY, name TEXT NOT NULL, template_id INTEGER,
  supplier_name TEXT NOT NULL DEFAULT '',
  review_mode TEXT NOT NULL DEFAULT 'adaptive',
  llm_concurrency INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL DEFAULT 'queued', stage TEXT NOT NULL DEFAULT '等待处理',
  progress INTEGER NOT NULL DEFAULT 0, current_file TEXT NOT NULL DEFAULT '',
  activity TEXT NOT NULL DEFAULT '任务已进入本地队列', resource TEXT NOT NULL DEFAULT 'SQLite 本地任务队列',
  heartbeat_at TEXT NOT NULL DEFAULT '',
  error TEXT NOT NULL DEFAULT '', summary TEXT NOT NULL DEFAULT '{}',
  template_snapshot TEXT NOT NULL DEFAULT '{}', rule_version INTEGER NOT NULL DEFAULT 1,
  cancel_requested INTEGER NOT NULL DEFAULT 0, cancelled_at TEXT NOT NULL DEFAULT '',
  deleted_at TEXT NOT NULL DEFAULT '', purge_after TEXT NOT NULL DEFAULT '',
  started_at TEXT NOT NULL DEFAULT '', completed_at TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
  FOREIGN KEY(template_id) REFERENCES audit_templates(id) ON DELETE SET NULL
);
CREATE TABLE IF NOT EXISTS documents (
  id TEXT PRIMARY KEY, library_code TEXT NOT NULL,
  document_kind TEXT NOT NULL DEFAULT 'other', original_name TEXT NOT NULL,
  supplier_name TEXT NOT NULL DEFAULT '',
  stored_path TEXT NOT NULL, sha256 TEXT NOT NULL, mime_type TEXT NOT NULL DEFAULT '',
  page_count INTEGER NOT NULL DEFAULT 0, page_text TEXT NOT NULL DEFAULT '[]',
  raw_text TEXT NOT NULL DEFAULT '', markdown TEXT NOT NULL DEFAULT '', html TEXT NOT NULL DEFAULT '',
  parse_status TEXT NOT NULL DEFAULT 'pending', ocr_status TEXT NOT NULL DEFAULT 'not_needed',
  index_status TEXT NOT NULL DEFAULT 'pending', index_fingerprint TEXT NOT NULL DEFAULT '',
  index_collection TEXT NOT NULL DEFAULT '', error TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL,
  detected_type TEXT NOT NULL DEFAULT '', type_confidence REAL NOT NULL DEFAULT 0,
  FOREIGN KEY(library_code) REFERENCES document_libraries(code)
);
CREATE INDEX IF NOT EXISTS idx_documents_library ON documents(library_code, created_at);
CREATE TABLE IF NOT EXISTS batch_documents (
  batch_id TEXT NOT NULL, document_id TEXT NOT NULL,
  role TEXT NOT NULL CHECK(role IN ('supplier','selected_basis','supplemental_basis')),
  priority INTEGER NOT NULL DEFAULT 4,
  PRIMARY KEY(batch_id,document_id),
  FOREIGN KEY(batch_id) REFERENCES review_batches(id) ON DELETE CASCADE,
  FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS template_basis (
  template_id INTEGER NOT NULL, document_id TEXT NOT NULL,
  PRIMARY KEY(template_id,document_id),
  FOREIGN KEY(template_id) REFERENCES audit_templates(id) ON DELETE CASCADE,
  FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS requirement_rules (
  id INTEGER PRIMARY KEY AUTOINCREMENT, document_id TEXT, template_id INTEGER,
  item TEXT NOT NULL, operator TEXT NOT NULL, value TEXT, upper_value TEXT,
  unit TEXT NOT NULL DEFAULT '', raw TEXT NOT NULL DEFAULT '', source_page INTEGER NOT NULL DEFAULT 1,
  clause TEXT NOT NULL DEFAULT '', priority INTEGER NOT NULL DEFAULT 4,
  required INTEGER NOT NULL DEFAULT 0, confirmed INTEGER NOT NULL DEFAULT 1,
  created_at TEXT NOT NULL,
  FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE,
  FOREIGN KEY(template_id) REFERENCES audit_templates(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS extracted_data (
  id INTEGER PRIMARY KEY AUTOINCREMENT, document_id TEXT NOT NULL,
  key TEXT NOT NULL, raw_value TEXT NOT NULL DEFAULT '', normalized_value TEXT NOT NULL DEFAULT '',
  unit TEXT NOT NULL DEFAULT '', page INTEGER NOT NULL DEFAULT 1,
  source_text TEXT NOT NULL DEFAULT '', category TEXT NOT NULL DEFAULT 'field',
  FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS findings (
  id INTEGER PRIMARY KEY AUTOINCREMENT, batch_id TEXT NOT NULL,
  category TEXT NOT NULL, severity TEXT NOT NULL, item TEXT NOT NULL,
  description TEXT NOT NULL, actual TEXT NOT NULL DEFAULT '', requirement TEXT NOT NULL DEFAULT '',
  source_file TEXT NOT NULL DEFAULT '', source_page INTEGER NOT NULL DEFAULT 1,
  source_text TEXT NOT NULL DEFAULT '', standard_file TEXT NOT NULL DEFAULT '',
  standard_page INTEGER NOT NULL DEFAULT 1, standard_clause TEXT NOT NULL DEFAULT '',
  logic TEXT NOT NULL DEFAULT '', suggestion TEXT NOT NULL DEFAULT '', confidence REAL NOT NULL DEFAULT 1,
  status TEXT NOT NULL DEFAULT 'AI发现', metadata TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL,
  rule_code TEXT NOT NULL DEFAULT '', rule_version INTEGER NOT NULL DEFAULT 1,
  document_type TEXT NOT NULL DEFAULT '', extraction_confidence REAL NOT NULL DEFAULT 1,
  decision_confidence REAL NOT NULL DEFAULT 1,
  FOREIGN KEY(batch_id) REFERENCES review_batches(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS review_history (
  id INTEGER PRIMARY KEY AUTOINCREMENT, finding_id INTEGER NOT NULL,
  old_status TEXT NOT NULL, new_status TEXT NOT NULL, note TEXT NOT NULL DEFAULT '', changed_at TEXT NOT NULL,
  FOREIGN KEY(finding_id) REFERENCES findings(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS service_config (
  id INTEGER PRIMARY KEY CHECK(id=1), allow_remote INTEGER NOT NULL DEFAULT 0,
  llm_base_url TEXT NOT NULL DEFAULT 'http://127.0.0.1:8080/v1', llm_api_key TEXT NOT NULL DEFAULT '',
  llm_model TEXT NOT NULL DEFAULT '', llm_temperature REAL NOT NULL DEFAULT 0.0,
  llm_concurrency INTEGER NOT NULL DEFAULT 1, llm_timeout_seconds INTEGER NOT NULL DEFAULT 300,
  embedding_base_url TEXT NOT NULL DEFAULT 'http://127.0.0.1:11434/v1', embedding_api_key TEXT NOT NULL DEFAULT '',
  embedding_model TEXT NOT NULL DEFAULT 'bge-m3:latest', embedding_dimensions INTEGER NOT NULL DEFAULT 1024,
  ocr_base_url TEXT NOT NULL DEFAULT 'http://127.0.0.1:8888', ocr_api_key TEXT NOT NULL DEFAULT '',
  ocr_backend TEXT NOT NULL DEFAULT 'pipeline', ocr_lang TEXT NOT NULL DEFAULT 'ch',
  chunk_size INTEGER NOT NULL DEFAULT 1200, chunk_overlap INTEGER NOT NULL DEFAULT 150,
  top_k INTEGER NOT NULL DEFAULT 5, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS service_presets (
  id INTEGER PRIMARY KEY AUTOINCREMENT, category TEXT NOT NULL,
  name TEXT NOT NULL, data TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL,
  UNIQUE(category,name)
);
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY, batch_id TEXT NOT NULL UNIQUE, status TEXT NOT NULL DEFAULT 'queued',
  attempts INTEGER NOT NULL DEFAULT 0, locked_at TEXT, error TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
  FOREIGN KEY(batch_id) REFERENCES review_batches(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS document_fields (
  id INTEGER PRIMARY KEY AUTOINCREMENT, document_id TEXT NOT NULL, field_key TEXT NOT NULL,
  raw_value TEXT NOT NULL DEFAULT '', normalized_value TEXT NOT NULL DEFAULT '', unit TEXT NOT NULL DEFAULT '',
  page INTEGER NOT NULL DEFAULT 1, source_text TEXT NOT NULL DEFAULT '', extraction_method TEXT NOT NULL DEFAULT '',
  confidence REAL NOT NULL DEFAULT 1, bbox TEXT NOT NULL DEFAULT '', row_index INTEGER, column_index INTEGER,
  FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_document_fields_key ON document_fields(document_id,field_key);
CREATE TABLE IF NOT EXISTS document_entities (
  id INTEGER PRIMARY KEY AUTOINCREMENT, batch_id TEXT NOT NULL, entity_type TEXT NOT NULL,
  normalized_value TEXT NOT NULL, display_value TEXT NOT NULL, members TEXT NOT NULL DEFAULT '[]',
  FOREIGN KEY(batch_id) REFERENCES review_batches(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS evidence_regions (
  id INTEGER PRIMARY KEY AUTOINCREMENT, document_id TEXT NOT NULL, page INTEGER NOT NULL,
  source_text TEXT NOT NULL DEFAULT '', bbox TEXT NOT NULL DEFAULT '', method TEXT NOT NULL DEFAULT '', confidence REAL NOT NULL DEFAULT 1,
  FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS finding_evidence (
  id INTEGER PRIMARY KEY AUTOINCREMENT, finding_id INTEGER NOT NULL, document_id TEXT,
  page INTEGER NOT NULL DEFAULT 1, source_text TEXT NOT NULL DEFAULT '', bbox TEXT NOT NULL DEFAULT '',
  evidence_type TEXT NOT NULL DEFAULT 'source', matched INTEGER NOT NULL DEFAULT 0,
  FOREIGN KEY(finding_id) REFERENCES findings(id) ON DELETE CASCADE,
  FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE SET NULL
);
CREATE TABLE IF NOT EXISTS review_feedback (
  id INTEGER PRIMARY KEY AUTOINCREMENT, finding_id INTEGER NOT NULL, batch_id TEXT NOT NULL,
  action TEXT NOT NULL, correction TEXT NOT NULL DEFAULT '', evidence TEXT NOT NULL DEFAULT '', note TEXT NOT NULL DEFAULT '',
  service_fingerprint TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL,
  FOREIGN KEY(finding_id) REFERENCES findings(id) ON DELETE CASCADE,
  FOREIGN KEY(batch_id) REFERENCES review_batches(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS job_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT, batch_id TEXT NOT NULL, stage TEXT NOT NULL,
  activity TEXT NOT NULL, resource TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL,
  FOREIGN KEY(batch_id) REFERENCES review_batches(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS rule_evaluations (
  id INTEGER PRIMARY KEY AUTOINCREMENT, batch_id TEXT NOT NULL, task_index INTEGER NOT NULL,
  task_name TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending', conclusion TEXT NOT NULL DEFAULT '',
  source_file TEXT NOT NULL DEFAULT '', source_page INTEGER NOT NULL DEFAULT 0,
  evidence TEXT NOT NULL DEFAULT '', evidence_type TEXT NOT NULL DEFAULT 'source',
  actual TEXT NOT NULL DEFAULT '', requirement TEXT NOT NULL DEFAULT '', logic TEXT NOT NULL DEFAULT '',
  suggestion TEXT NOT NULL DEFAULT '', confidence REAL NOT NULL DEFAULT 0, error TEXT NOT NULL DEFAULT '',
  started_at TEXT NOT NULL DEFAULT '', completed_at TEXT NOT NULL DEFAULT '',
  UNIQUE(batch_id,task_index), FOREIGN KEY(batch_id) REFERENCES review_batches(id) ON DELETE CASCADE
);
CREATE VIRTUAL TABLE IF NOT EXISTS document_fts USING fts5(document_id UNINDEXED,page UNINDEXED,content,tokenize='unicode61');
"""


class ReviewDatabase:
    """V2 persistence layer. All writes use parameterized SQL and short transactions."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._backup_before_v04()
        self.initialize()

    def _backup_before_v04(self) -> None:
        if not self.path.is_file():
            return
        backup = self.path.with_suffix(self.path.suffix + ".pre-v0.4.0.bak")
        if backup.exists():
            return
        try:
            with sqlite3.connect(self.path) as connection:
                columns = {row[1] for row in connection.execute("PRAGMA table_info(review_batches)").fetchall()}
            if columns and "deleted_at" not in columns:
                shutil.copy2(self.path, backup)
        except sqlite3.DatabaseError:
            return

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def initialize(self) -> None:
        now = utcnow()
        with self.connect() as connection:
            connection.executescript(SCHEMA_V2)
            # v0.4.0 briefly introduced a generic versioned-rule subsystem. It was
            # withdrawn because it suppressed the proven template-specific audit.
            connection.executescript("""
                DROP TABLE IF EXISTS template_rule_versions;
                DROP TABLE IF EXISTS audit_rule_versions;
                DROP TABLE IF EXISTS audit_rules;
            """)
            columns = {row[1] for row in connection.execute("PRAGMA table_info(review_batches)").fetchall()}
            for name, declaration in {
                "supplier_name": "TEXT NOT NULL DEFAULT ''",
                "review_mode": "TEXT NOT NULL DEFAULT 'adaptive'",
                "llm_concurrency": "INTEGER NOT NULL DEFAULT 0",
                "activity": "TEXT NOT NULL DEFAULT '任务已进入本地队列'",
                "resource": "TEXT NOT NULL DEFAULT 'SQLite 本地任务队列'",
                "heartbeat_at": "TEXT NOT NULL DEFAULT ''",
                "template_snapshot": "TEXT NOT NULL DEFAULT '{}'",
                "rule_version": "INTEGER NOT NULL DEFAULT 1",
                "cancel_requested": "INTEGER NOT NULL DEFAULT 0",
                "cancelled_at": "TEXT NOT NULL DEFAULT ''",
                "deleted_at": "TEXT NOT NULL DEFAULT ''",
                "purge_after": "TEXT NOT NULL DEFAULT ''",
                "started_at": "TEXT NOT NULL DEFAULT ''",
                "completed_at": "TEXT NOT NULL DEFAULT ''",
            }.items():
                if name not in columns:
                    connection.execute(f"ALTER TABLE review_batches ADD COLUMN {name} {declaration}")
            document_columns = {row[1] for row in connection.execute("PRAGMA table_info(documents)").fetchall()}
            for name, declaration in {
                "supplier_name": "TEXT NOT NULL DEFAULT ''",
                "detected_type": "TEXT NOT NULL DEFAULT ''",
                "type_confidence": "REAL NOT NULL DEFAULT 0",
            }.items():
                if name not in document_columns:
                    connection.execute(f"ALTER TABLE documents ADD COLUMN {name} {declaration}")
            finding_columns = {row[1] for row in connection.execute("PRAGMA table_info(findings)").fetchall()}
            for name, declaration in {
                "rule_code": "TEXT NOT NULL DEFAULT ''", "rule_version": "INTEGER NOT NULL DEFAULT 1",
                "document_type": "TEXT NOT NULL DEFAULT ''", "extraction_confidence": "REAL NOT NULL DEFAULT 1",
                "decision_confidence": "REAL NOT NULL DEFAULT 1",
            }.items():
                if name not in finding_columns:
                    connection.execute(f"ALTER TABLE findings ADD COLUMN {name} {declaration}")
            template_columns = {row[1] for row in connection.execute("PRAGMA table_info(audit_templates)").fetchall()}
            if "review_instructions" not in template_columns:
                connection.execute("ALTER TABLE audit_templates ADD COLUMN review_instructions TEXT NOT NULL DEFAULT ''")
            service_columns = {row[1] for row in connection.execute("PRAGMA table_info(service_config)").fetchall()}
            for name, declaration in {
                "llm_concurrency": "INTEGER NOT NULL DEFAULT 1",
                "llm_timeout_seconds": "INTEGER NOT NULL DEFAULT 300",
            }.items():
                if name not in service_columns:
                    connection.execute(f"ALTER TABLE service_config ADD COLUMN {name} {declaration}")
            connection.executemany(
                "INSERT OR IGNORE INTO document_libraries(code,name,description) VALUES(?,?,?)",
                [
                    ("basis", "审核依据库", "采购技术要求、图纸、企业及国家/行业/国际标准"),
                    ("supplier", "供应商档案库", "按审核批次保存供应商证明书、报告和证据"),
                ],
            )
            connection.execute(
                """INSERT OR IGNORE INTO service_config(id,updated_at) VALUES(1,?)""", (now,)
            )
            if not connection.execute("SELECT 1 FROM audit_templates LIMIT 1").fetchone():
                connection.execute(
                    """INSERT INTO audit_templates(name,description,required_document_types,required_items,enabled,is_default,created_at)
                       VALUES(?,?,?,?,1,1,?)""",
                    ("通用材料质量审核", "材料、炉号、批次、化学成分与力学性能通用检查", "[]",
                     json.dumps(["化学成分", "抗拉强度", "屈服强度", "延伸率"], ensure_ascii=False), now),
                )

    def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        with self.connect() as connection:
            cursor = connection.execute(sql, params)
            return int(cursor.lastrowid or 0)

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        with self.connect() as connection:
            return [dict(row) for row in connection.execute(sql, params).fetchall()]

    def one(self, sql: str, params: Sequence[Any] = ()) -> dict[str, Any] | None:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def create_batch(self, template_id: int | None, review_mode: str = "adaptive",
                     llm_concurrency: int = 0) -> str:
        batch_id = str(uuid.uuid4())
        now = utcnow()
        # Seconds make repeated uploads visibly distinct while the UUID remains the
        # stable event identity used by storage, jobs and findings.
        name = f"审核批次 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        template = self.one(
            "SELECT name,description,required_items,review_instructions FROM audit_templates WHERE id=?", (template_id,)
        ) if template_id else None
        snapshot = json.dumps(template or {}, ensure_ascii=False)
        review_mode = review_mode if review_mode in {"adaptive", "deep"} else "adaptive"
        llm_concurrency = max(0, min(4, int(llm_concurrency or 0)))
        self.execute(
            """INSERT INTO review_batches(id,name,template_id,review_mode,llm_concurrency,template_snapshot,
               heartbeat_at,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?)""",
            (batch_id, name, template_id, review_mode, llm_concurrency, snapshot, now, now, now),
        )
        self.execute(
            "INSERT INTO jobs(id,batch_id,created_at,updated_at) VALUES(?,?,?,?)",
            (str(uuid.uuid4()), batch_id, now, now),
        )
        return batch_id

    def delete_template(self, template_id: int) -> None:
        template = self.one("SELECT name,is_default FROM audit_templates WHERE id=?", (template_id,))
        if not template:
            raise ValueError("模板不存在或已被删除")
        if template["is_default"]:
            raise ValueError("默认模板不能直接删除；请先将其他模板设为默认模板")
        self.execute("DELETE FROM audit_templates WHERE id=?", (template_id,))

    def add_document(self, *, library: str, kind: str, original_name: str, stored_path: str,
                     sha256: str, mime_type: str = "") -> str:
        document_id = str(uuid.uuid4())
        self.execute(
            """INSERT INTO documents(id,library_code,document_kind,original_name,stored_path,sha256,mime_type,created_at)
               VALUES(?,?,?,?,?,?,?,?)""",
            (document_id, library, kind, original_name, stored_path, sha256, mime_type, utcnow()),
        )
        return document_id

    def attach_document(self, batch_id: str, document_id: str, role: str, priority: int) -> None:
        self.execute(
            "INSERT OR REPLACE INTO batch_documents(batch_id,document_id,role,priority) VALUES(?,?,?,?)",
            (batch_id, document_id, role, priority),
        )

    def update_batch(self, batch_id: str, **values: Any) -> None:
        allowed = {"status", "stage", "progress", "current_file", "activity", "resource", "heartbeat_at", "error", "summary",
                   "cancel_requested", "cancelled_at", "deleted_at", "purge_after", "supplier_name", "started_at", "completed_at"}
        payload = {key: value for key, value in values.items() if key in allowed}
        if not payload:
            return
        if "summary" in payload and not isinstance(payload["summary"], str):
            payload["summary"] = json.dumps(payload["summary"], ensure_ascii=False)
        payload["updated_at"] = utcnow()
        columns = ",".join(f"{key}=?" for key in payload)
        self.execute(f"UPDATE review_batches SET {columns} WHERE id=?", [*payload.values(), batch_id])

    def update_finding_status(self, finding_id: int, new_status: str, note: str = "",
                              correction: str = "", evidence: str = "", service_fingerprint: str = "") -> None:
        row = self.one("SELECT status,batch_id FROM findings WHERE id=?", (finding_id,))
        if not row:
            raise ValueError("问题不存在")
        with self.connect() as connection:
            connection.execute("UPDATE findings SET status=? WHERE id=?", (new_status, finding_id))
            connection.execute(
                "INSERT INTO review_history(finding_id,old_status,new_status,note,changed_at) VALUES(?,?,?,?,?)",
                (finding_id, row["status"], new_status, note, utcnow()),
            )
            connection.execute(
                """INSERT INTO review_feedback(finding_id,batch_id,action,correction,evidence,note,
                   service_fingerprint,created_at) VALUES(?,?,?,?,?,?,?,?)""",
                (finding_id, row["batch_id"], new_status, correction, evidence, note,
                 service_fingerprint, utcnow()),
            )

    def request_cancel(self, batch_id: str) -> bool:
        row = self.one("SELECT status FROM review_batches WHERE id=? AND deleted_at=''", (batch_id,))
        if not row or row["status"] in {"completed", "failed", "cancelled"}:
            return False
        now = utcnow()
        if row["status"] == "queued":
            self.execute("UPDATE jobs SET status='cancelled',updated_at=? WHERE batch_id=?", (now, batch_id))
            self.update_batch(batch_id, status="cancelled", stage="已取消", progress=0, cancel_requested=1,
                              cancelled_at=now, activity="任务在开始前已取消", resource="本机任务队列")
        else:
            self.execute("UPDATE jobs SET status='cancel_requested',updated_at=? WHERE batch_id=?", (now, batch_id))
            # Cancellation is final from the user's perspective as soon as it is
            # accepted: process_batch checks this flag before publishing results.
            # An already-running HTTP request may finish in the background, but
            # must not keep the UI stuck on "正在停止" until its timeout expires.
            self.update_batch(batch_id, status="cancelled", cancel_requested=1,
                              stage="已取消", cancelled_at=now,
                              activity="停止操作已生效；正在释放后台资源", resource="本地 worker")
        return True

    def is_cancel_requested(self, batch_id: str) -> bool:
        row = self.one("SELECT cancel_requested,deleted_at FROM review_batches WHERE id=?", (batch_id,))
        return bool(row and (row["cancel_requested"] or row["deleted_at"]))

    def mark_cancelled(self, batch_id: str) -> None:
        now = utcnow()
        self.update_batch(batch_id, status="cancelled", stage="已取消", cancelled_at=now,
                          activity="审核已在安全检查点停止，未发布不完整结论", resource="本地 worker")
        self.execute("UPDATE jobs SET status='cancelled',updated_at=? WHERE batch_id=?", (now, batch_id))

    def soft_delete_batch(self, batch_id: str) -> None:
        self.request_cancel(batch_id)
        now = datetime.now(timezone.utc)
        self.update_batch(batch_id, deleted_at=now.isoformat(timespec="seconds"),
                          purge_after=(now + timedelta(days=30)).isoformat(timespec="seconds"))

    def restore_batch(self, batch_id: str) -> None:
        self.update_batch(batch_id, deleted_at="", purge_after="")

    def purge_batch(self, batch_id: str, force: bool = False) -> list[dict[str, Any]]:
        """Permanently remove a trashed batch and return private documents for filesystem cleanup."""
        batch = self.one("SELECT deleted_at,purge_after FROM review_batches WHERE id=?", (batch_id,))
        if not batch or not batch["deleted_at"]:
            raise ValueError("仅回收站中的审核记录可以永久删除")
        if not force and batch["purge_after"] and datetime.fromisoformat(batch["purge_after"]) > datetime.now(timezone.utc):
            raise ValueError("回收站保留期尚未结束，当前只能恢复记录")
        private = self.query(
            """SELECT d.* FROM batch_documents bd JOIN documents d ON d.id=bd.document_id
               WHERE bd.batch_id=? AND bd.role IN ('supplier','supplemental_basis')
               AND NOT EXISTS (SELECT 1 FROM batch_documents other WHERE other.document_id=d.id AND other.batch_id<>?)""",
            (batch_id, batch_id),
        )
        with self.connect() as connection:
            connection.execute("DELETE FROM review_batches WHERE id=?", (batch_id,))
            connection.executemany("DELETE FROM documents WHERE id=?", [(row["id"],) for row in private])
        return private

    def claim_job(self) -> dict[str, Any] | None:
        """Atomically claim the oldest queued job for the single local worker."""
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """SELECT j.* FROM jobs j JOIN review_batches b ON b.id=j.batch_id
                   WHERE j.status='queued' AND b.deleted_at='' AND b.cancel_requested=0 ORDER BY j.created_at LIMIT 1"""
            ).fetchone()
            if not row:
                return None
            now = utcnow()
            connection.execute(
                "UPDATE jobs SET status='running',attempts=attempts+1,locked_at=?,updated_at=? WHERE id=?",
                (now, now, row["id"]),
            )
            result = dict(row)
            result["status"] = "running"
            return result

    def requeue_running_jobs(self) -> int:
        """Recover work interrupted when this project's single worker was stopped."""
        now = utcnow()
        with self.connect() as connection:
            cancelled = connection.execute(
                """SELECT j.batch_id FROM jobs j JOIN review_batches b ON b.id=j.batch_id
                   WHERE b.cancel_requested=1 AND b.status IN ('cancel_requested','cancelled')
                     AND j.status='cancel_requested'"""
            ).fetchall()
            if cancelled:
                connection.execute(
                    """UPDATE jobs SET status='cancelled',updated_at=? WHERE batch_id IN
                       (SELECT id FROM review_batches WHERE cancel_requested=1
                        AND status IN ('cancel_requested','cancelled'))""", (now,)
                )
                connection.executemany(
                    """UPDATE review_batches SET status='cancelled',stage='已取消',cancelled_at=?,
                       activity='审核在 worker 重启时完成停止',resource='本地 worker',heartbeat_at=?,updated_at=? WHERE id=?""",
                    [(now, now, now, row["batch_id"]) for row in cancelled],
                )
            rows = connection.execute(
                """SELECT j.batch_id FROM jobs j JOIN review_batches b ON b.id=j.batch_id
                   WHERE j.status='running' AND b.cancel_requested=0 AND b.deleted_at=''"""
            ).fetchall()
            if not rows:
                return 0
            connection.execute(
                """UPDATE jobs SET status='queued',locked_at=NULL,error='',updated_at=? WHERE status='running'
                   AND batch_id IN (SELECT id FROM review_batches WHERE cancel_requested=0 AND deleted_at='')""", (now,)
            )
            connection.executemany(
                """UPDATE review_batches SET status='queued',stage='恢复审核',activity='上次审核被中断，已重新进入队列',
                   resource='SQLite 本地任务队列',heartbeat_at=?,updated_at=? WHERE id=?""",
                [(now, now, row["batch_id"]) for row in rows],
            )
            return len(rows)


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

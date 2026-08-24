from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

from app.models import Finding


SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS projects (
  id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, supplier TEXT DEFAULT '',
  po_number TEXT DEFAULT '', product_name TEXT DEFAULT '', product_model TEXT DEFAULT '',
  material_name TEXT DEFAULT '', material_grade TEXT DEFAULT '', drawing_number TEXT DEFAULT '',
  batch_number TEXT DEFAULT '', heat_number TEXT DEFAULT '', notes TEXT DEFAULT '',
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
  id INTEGER PRIMARY KEY AUTOINCREMENT, project_id INTEGER NOT NULL, kind TEXT NOT NULL,
  original_name TEXT NOT NULL, stored_path TEXT NOT NULL, status TEXT DEFAULT '已上传',
  page_count INTEGER DEFAULT 0, error TEXT DEFAULT '', created_at TEXT NOT NULL,
  FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS standards (
  id INTEGER PRIMARY KEY AUTOINCREMENT, project_id INTEGER NOT NULL, document_id INTEGER NOT NULL,
  name TEXT DEFAULT '', number TEXT DEFAULT '', version TEXT DEFAULT '', priority INTEGER DEFAULT 3,
  FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE,
  FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS standard_chunks (
  id INTEGER PRIMARY KEY AUTOINCREMENT, standard_id INTEGER NOT NULL, content TEXT NOT NULL,
  page INTEGER DEFAULT 1, section TEXT DEFAULT '', clause TEXT DEFAULT '', embedding TEXT DEFAULT '',
  FOREIGN KEY(standard_id) REFERENCES standards(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS extracted_data (
  id INTEGER PRIMARY KEY AUTOINCREMENT, document_id INTEGER NOT NULL, key TEXT NOT NULL,
  raw_value TEXT DEFAULT '', normalized_value TEXT DEFAULT '', unit TEXT DEFAULT '', page INTEGER DEFAULT 1,
  source_text TEXT DEFAULT '', category TEXT DEFAULT 'field',
  FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS requirements (
  id INTEGER PRIMARY KEY AUTOINCREMENT, project_id INTEGER NOT NULL, standard_id INTEGER,
  item TEXT NOT NULL, operator TEXT NOT NULL, value TEXT, upper_value TEXT, unit TEXT DEFAULT '',
  raw TEXT DEFAULT '', source_page INTEGER DEFAULT 1, clause TEXT DEFAULT '', required INTEGER DEFAULT 0,
  FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS findings (
  id INTEGER PRIMARY KEY AUTOINCREMENT, project_id INTEGER NOT NULL, category TEXT NOT NULL,
  severity TEXT NOT NULL, item TEXT NOT NULL, description TEXT NOT NULL, actual TEXT DEFAULT '',
  requirement TEXT DEFAULT '', source_file TEXT DEFAULT '', source_page INTEGER DEFAULT 1,
  source_text TEXT DEFAULT '', standard_file TEXT DEFAULT '', standard_page INTEGER DEFAULT 1,
  standard_clause TEXT DEFAULT '', logic TEXT DEFAULT '', suggestion TEXT DEFAULT '',
  confidence REAL DEFAULT 1.0, status TEXT DEFAULT 'AI发现', metadata TEXT DEFAULT '{}',
  created_at TEXT NOT NULL, FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS review_history (
  id INTEGER PRIMARY KEY AUTOINCREMENT, finding_id INTEGER NOT NULL, old_status TEXT, new_status TEXT NOT NULL,
  note TEXT DEFAULT '', changed_at TEXT NOT NULL,
  FOREIGN KEY(finding_id) REFERENCES findings(id) ON DELETE CASCADE
);
"""


class Database:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(SCHEMA)

    def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        with self.connect() as connection:
            cursor = connection.execute(sql, params)
            return int(cursor.lastrowid or 0)

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        with self.connect() as connection:
            return [dict(row) for row in connection.execute(sql, params).fetchall()]

    def create_project(self, values: dict[str, str]) -> int:
        columns = ["name", "supplier", "po_number", "product_name", "product_model", "material_name",
                   "material_grade", "drawing_number", "batch_number", "heat_number", "notes"]
        return self.execute(
            f"INSERT INTO projects ({','.join(columns)},created_at) VALUES ({','.join('?' for _ in columns)},?)",
            [values.get(column, "").strip() for column in columns] + [_now()],
        )

    def add_finding(self, project_id: int, finding: Finding) -> int:
        columns = ["category", "severity", "item", "description", "actual", "requirement", "source_file",
                   "source_page", "source_text", "standard_file", "standard_page", "standard_clause", "logic",
                   "suggestion", "confidence", "status", "metadata"]
        values = [getattr(finding, column) for column in columns[:-1]] + [json.dumps(finding.metadata, ensure_ascii=False)]
        return self.execute(
            f"INSERT INTO findings (project_id,{','.join(columns)},created_at) VALUES (?,{','.join('?' for _ in columns)},?)",
            [project_id, *values, _now()],
        )

    def update_finding_status(self, finding_id: int, new_status: str, note: str = "") -> None:
        rows = self.query("SELECT status FROM findings WHERE id=?", (finding_id,))
        if not rows:
            raise ValueError("问题记录不存在")
        old = rows[0]["status"]
        with self.connect() as connection:
            connection.execute("UPDATE findings SET status=? WHERE id=?", (new_status, finding_id))
            connection.execute(
                "INSERT INTO review_history(finding_id,old_status,new_status,note,changed_at) VALUES(?,?,?,?,?)",
                (finding_id, old, new_status, note, _now()),
            )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


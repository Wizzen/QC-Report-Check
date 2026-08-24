from pathlib import Path

import pytest

from app.database import Database
from app.llm.ollama_client import parse_json_object
from app.models import PageText
from app.rag.knowledge import KnowledgeBase, split_chunks


class OfflineClient:
    def embed(self, texts: list[str], model: str) -> list[list[float]]:
        raise ConnectionError("offline")


def test_json_parser_accepts_fenced_object_and_rejects_array() -> None:
    assert parse_json_object('```json\n{"result":"FAIL"}\n```')["result"] == "FAIL"
    with pytest.raises(ValueError): parse_json_object("[1, 2]")


def test_standard_retrieval_lexical_fallback(tmp_path: Path) -> None:
    db = Database(tmp_path / "test.db")
    project = db.create_project({"name": "测试项目"})
    document = db.execute("INSERT INTO documents(project_id,kind,original_name,stored_path,created_at) VALUES(?,?,?,?,datetime('now'))", (project, "standard", "协议.txt", "x"))
    standard = db.execute("INSERT INTO standards(project_id,document_id,name,priority) VALUES(?,?,?,?)", (project, document, "采购协议", 1))
    kb = KnowledgeBase(db, OfflineClient(), "bge-m3")  # type: ignore[arg-type]
    count, mode = kb.index(standard, [PageText(12, "5.3.2 Q355B 的抗拉强度不得低于 470 MPa。\n\n其他一般要求。")])
    results = kb.search(project, "Q355B 抗拉强度 470 MPa")
    assert count >= 1 and mode == "词法降级"
    assert results and "抗拉强度" in results[0]["content"]


def test_chunk_keeps_page_and_clause() -> None:
    chunks = split_chunks([PageText(7, "4.2.1 化学成分 P 不得高于 0.035 %")])
    assert chunks[0][0] == 7
    assert chunks[0][2] == "4.2.1"


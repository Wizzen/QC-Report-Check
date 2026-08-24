from __future__ import annotations

import json
import logging
import math
import re
from typing import Any

from app.database import Database
from app.llm import OllamaClient
from app.models import PageText


LOGGER = logging.getLogger(__name__)


class KnowledgeBase:
    """SQLite-backed chunks with Ollama vectors and a deterministic lexical fallback."""

    def __init__(self, db: Database, client: OllamaClient, embedding_model: str):
        self.db, self.client, self.embedding_model = db, client, embedding_model

    def index(self, standard_id: int, pages: list[PageText]) -> tuple[int, str]:
        chunks = split_chunks(pages)
        texts = [chunk[1] for chunk in chunks]
        embeddings: list[list[float]] = []
        mode = "bge-m3"
        try:
            for start in range(0, len(texts), 16):
                embeddings.extend(self.client.embed(texts[start:start + 16], self.embedding_model))
            if len(embeddings) != len(texts):
                raise ValueError("Embedding 返回数量不一致")
        except Exception as exc:
            LOGGER.warning("Embedding 不可用，使用本地词法检索：%s", exc)
            embeddings = [[] for _ in texts]; mode = "词法降级"
        with self.db.connect() as connection:
            connection.execute("DELETE FROM standard_chunks WHERE standard_id=?", (standard_id,))
            connection.executemany(
                "INSERT INTO standard_chunks(standard_id,content,page,section,clause,embedding) VALUES(?,?,?,?,?,?)",
                [(standard_id, text, page, "", clause, json.dumps(vector)) for (page, text, clause), vector in zip(chunks, embeddings)],
            )
        return len(chunks), mode

    def search(self, project_id: int, query: str, limit: int = 5) -> list[dict[str, Any]]:
        rows = self.db.query(
            """SELECT c.*,s.name standard_name,s.number standard_number,d.original_name source_file,s.priority
               FROM standard_chunks c JOIN standards s ON s.id=c.standard_id
               JOIN documents d ON d.id=s.document_id WHERE s.project_id=?""", (project_id,),
        )
        if not rows: return []
        query_vector: list[float] = []
        if any(json.loads(row["embedding"] or "[]") for row in rows):
            try: query_vector = self.client.embed([query], self.embedding_model)[0]
            except Exception as exc: LOGGER.warning("查询向量失败，改用词法检索：%s", exc)
        query_tokens = _tokens(query)
        for row in rows:
            vector = json.loads(row["embedding"] or "[]")
            semantic = _cosine(query_vector, vector) if query_vector and vector else 0.0
            terms = _tokens(row["content"])
            lexical = len(query_tokens & terms) / max(1, len(query_tokens))
            priority_bonus = max(0, 4 - int(row["priority"] or 3)) * 0.03
            row["score"] = round((semantic * 0.75 + lexical * 0.25 if semantic else lexical) + priority_bonus, 6)
        return sorted(rows, key=lambda row: row["score"], reverse=True)[:limit]


def split_chunks(pages: list[PageText], max_chars: int = 1200) -> list[tuple[int, str, str]]:
    result: list[tuple[int, str, str]] = []
    for page in pages:
        blocks = [item.strip() for item in re.split(r"\n\s*\n|(?=^\s*(?:\d+\.)+\d+\s+)", page.text, flags=re.MULTILINE) if item.strip()]
        for block in blocks:
            for start in range(0, len(block), max_chars):
                text = block[start:start + max_chars]
                match = re.match(r"\s*((?:\d+\.)+\d+|\d+\.?)\s*", text)
                result.append((page.page, text, match.group(1).rstrip(".") if match else ""))
    return result


def _tokens(text: str) -> set[str]:
    latin = re.findall(r"[A-Za-z]+\d*(?:\.\d+)?|\d+(?:\.\d+)?", text.casefold())
    chinese = [text[index:index + 2] for index in range(len(text) - 1) if "\u4e00" <= text[index] <= "\u9fff"]
    return set(latin + chinese)


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left: return 0.0
    denominator = math.sqrt(sum(x*x for x in left)) * math.sqrt(sum(x*x for x in right))
    return sum(x*y for x, y in zip(left, right)) / denominator if denominator else 0.0


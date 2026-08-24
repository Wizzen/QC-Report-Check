from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from app.integrations import EmbeddingClient, ServiceSettings
from app.models import PageText


LOGGER = logging.getLogger(__name__)


class DocumentVectorIndex:
    """One persistent Chroma collection per document, with no cross-library mixing."""

    def __init__(self, root: Path, settings: ServiceSettings):
        self.root = root
        self.settings = settings
        self.embedding = EmbeddingClient(settings)

    def index(self, document_id: str, pages: list[PageText]) -> tuple[int, str]:
        chunks = split_document(pages, self.settings.chunk_size, self.settings.chunk_overlap)
        if not chunks:
            return 0, "empty"
        try:
            import chromadb
        except ImportError:
            LOGGER.warning("ChromaDB 未安装，文档 %s 仅保留结构化文本", document_id)
            return 0, "chromadb_unavailable"
        vectors: list[list[float]] = []
        texts = [item["text"] for item in chunks]
        try:
            for offset in range(0, len(texts), 16):
                vectors.extend(self.embedding.embed(texts[offset:offset + 16]))
        except Exception as exc:
            # Vector search is an enhancement. A temporarily unavailable service
            # must not prevent deterministic extraction and rule evaluation.
            LOGGER.warning("文档向量化失败 %s: %s", document_id, type(exc).__name__)
            return 0, "embedding_failed"
        directory = self.root / document_id
        directory.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(directory))
        collection = client.get_or_create_collection(name="document_chunks", metadata={"hnsw:space": "cosine"})
        existing = collection.get()
        if existing.get("ids"):
            collection.delete(ids=existing["ids"])
        collection.add(
            ids=[f"{document_id}:{index}" for index in range(len(chunks))],
            documents=texts,
            embeddings=vectors,
            metadatas=[{"page": item["page"], "clause": item["clause"]} for item in chunks],
        )
        return len(chunks), "ready"

    def search(self, document_ids: list[str], query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        if not document_ids:
            return []
        try:
            import chromadb
        except ImportError:
            return []
        query_vector = self.embedding.embed([query])[0]
        results: list[dict[str, Any]] = []
        for document_id in document_ids:
            directory = self.root / document_id
            if not directory.exists():
                continue
            try:
                collection = chromadb.PersistentClient(path=str(directory)).get_collection("document_chunks")
                answer = collection.query(query_embeddings=[query_vector], n_results=top_k or self.settings.top_k)
                for text, metadata, distance in zip(answer["documents"][0], answer["metadatas"][0], answer["distances"][0]):
                    results.append({"document_id": document_id, "text": text, "page": metadata.get("page", 1),
                                    "clause": metadata.get("clause", ""), "score": 1.0 - float(distance)})
            except Exception as exc:
                LOGGER.warning("文档向量检索失败 %s: %s", document_id, type(exc).__name__)
        return sorted(results, key=lambda item: item["score"], reverse=True)[: top_k or self.settings.top_k]


def split_document(pages: list[PageText], size: int, overlap: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    stride = max(1, size - min(overlap, size - 1))
    for page in pages:
        text = re.sub(r"\n{3,}", "\n\n", page.text).strip()
        for start in range(0, len(text), stride):
            chunk = text[start:start + size].strip()
            if not chunk:
                continue
            clause = ""
            match = re.search(r"(?m)^\s*((?:\d+\.)+\d+)\s+", chunk)
            if match:
                clause = match.group(1)
            output.append({"page": page.page, "clause": clause, "text": chunk})
            if start + size >= len(text):
                break
    return output

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.database import Database
from app.extractors import extract_items, extract_requirements
from app.llm import OllamaClient
from app.models import ExtractedItem, Finding, Requirement
from app.parsers import parse_document
from app.rag import KnowledgeBase
from app.rules import AuditEngine


LOGGER = logging.getLogger(__name__)


class AuditService:
    def __init__(self, db: Database, client: OllamaClient, embedding_model: str, llm_model: str = ""):
        self.db, self.client, self.llm_model = db, client, llm_model
        self.knowledge = KnowledgeBase(db, client, embedding_model)

    def import_document(self, project_id: int, kind: str, original_name: str, path: Path,
                        standard_meta: dict[str, Any] | None = None) -> dict[str, Any]:
        document_id = self.db.execute(
            "INSERT INTO documents(project_id,kind,original_name,stored_path,created_at) VALUES(?,?,?,?,datetime('now'))",
            (project_id, kind, original_name, str(path)),
        )
        try:
            pages = parse_document(path)
            self.db.execute("UPDATE documents SET status='解析完成',page_count=? WHERE id=?", (len(pages), document_id))
            if kind == "supplier":
                items = extract_items(pages)
                with self.db.connect() as connection:
                    connection.executemany(
                        """INSERT INTO extracted_data(document_id,key,raw_value,normalized_value,unit,page,source_text,category)
                           VALUES(?,?,?,?,?,?,?,?)""",
                        [(document_id, item.key, item.raw, str(item.value or ""), item.unit, item.page,
                          item.source_text, item.category) for item in items],
                    )
                return {"document_id": document_id, "pages": len(pages), "items": len(items)}
            meta = standard_meta or {}
            standard_id = self.db.execute(
                "INSERT INTO standards(project_id,document_id,name,number,version,priority) VALUES(?,?,?,?,?,?)",
                (project_id, document_id, meta.get("name", original_name), meta.get("number", ""),
                 meta.get("version", ""), int(meta.get("priority", 3))),
            )
            requirements = extract_requirements(pages, original_name)
            with self.db.connect() as connection:
                connection.executemany(
                    """INSERT INTO requirements(project_id,standard_id,item,operator,value,upper_value,unit,raw,source_page,clause,required)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                    [(project_id, standard_id, req.item, req.operator, req.value, req.upper_value, req.unit,
                      req.raw, req.source_page, req.clause, int(req.required)) for req in requirements],
                )
            chunk_count, mode = self.knowledge.index(standard_id, pages)
            return {"document_id": document_id, "standard_id": standard_id, "pages": len(pages),
                    "requirements": len(requirements), "chunks": chunk_count, "index_mode": mode}
        except Exception as exc:
            self.db.execute("UPDATE documents SET status='解析失败',error=? WHERE id=?", (str(exc)[:1000], document_id))
            raise

    def run(self, project_id: int, use_llm: bool = False) -> list[Finding]:
        documents: dict[str, list[ExtractedItem]] = {}
        rows = self.db.query(
            """SELECT d.original_name,e.* FROM extracted_data e JOIN documents d ON d.id=e.document_id
               WHERE d.project_id=? AND d.kind='supplier'""", (project_id,),
        )
        for row in rows:
            value: float | str | None = row["normalized_value"]
            if row["category"] == "measurement":
                try: value = float(value)
                except (TypeError, ValueError): value = None
            documents.setdefault(row["original_name"], []).append(ExtractedItem(
                row["key"], row["raw_value"], value, row["unit"], row["page"], row["source_text"], row["category"]
            ))
        requirements = [Requirement(
            row["item"], row["operator"], _number_or_text(row["value"]),
            _optional_float(row["upper_value"]), row["unit"], row["raw"], row["source_file"],
            row["source_page"], row["clause"], bool(row["required"])
        ) for row in self.db.query(
            """SELECT r.*,d.original_name source_file FROM requirements r
               LEFT JOIN standards s ON s.id=r.standard_id LEFT JOIN documents d ON d.id=s.document_id
               WHERE r.project_id=? ORDER BY s.priority""", (project_id,))]
        findings = AuditEngine().audit(documents, requirements)
        if not requirements:
            findings.append(Finding("待人工确认", "Review", "审核依据", "没有提取到明确审核要求，未进行推测性判定",
                                    "", "缺少明确审核依据", logic="规则引擎没有可执行的审核条件", confidence=1.0))
        if use_llm and self.llm_model:
            findings = [self._explain(project_id, finding) for finding in findings]
        with self.db.connect() as connection:
            connection.execute("DELETE FROM findings WHERE project_id=? AND status='AI发现'", (project_id,))
        for finding in findings: self.db.add_finding(project_id, finding)
        LOGGER.info("项目 %s 审核完成：%s 个问题", project_id, len(findings))
        return findings

    def _explain(self, project_id: int, finding: Finding) -> Finding:
        evidence = self.knowledge.search(project_id, f"{finding.item} {finding.requirement}", limit=3)
        context = "\n".join(f"[{row['source_file']} 第{row['page']}页 {row['clause']}] {row['content']}" for row in evidence)
        prompt = f"""你是本地质量审核解释器，只能基于下方规则结果与标准证据解释，不得引用训练知识。
规则结果：{finding.logic}\n实际证据：{finding.source_text}\n要求：{finding.requirement}\n标准证据：{context or '无'}
返回 JSON：{{"reason":"简短说明","suggestion":"整改建议","confidence":0到1}}。不得改变规则判定。"""
        try:
            result = self.client.generate_json(self.llm_model, prompt)
            confidence = min(1.0, max(0.0, float(result.get("confidence", 0.5))))
            finding.description = str(result.get("reason") or finding.description)
            finding.suggestion = str(result.get("suggestion") or finding.suggestion)
            finding.confidence = confidence
            if confidence < 0.70: finding.severity, finding.status = "Review", "AI发现"
        except Exception as exc:
            LOGGER.warning("LLM 解释失败，保留规则解释：%s", exc)
        return finding


def _number_or_text(value: Any) -> float | str | None:
    if value is None: return None
    try: return float(value)
    except (TypeError, ValueError): return str(value)


def _optional_float(value: Any) -> float | None:
    if value in (None, ""): return None
    try: return float(value)
    except (TypeError, ValueError): return None


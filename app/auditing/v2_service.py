from __future__ import annotations

import hashlib
import json
import logging
import mimetypes
import shutil
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Iterable
from urllib.parse import urlsplit

from app.database import ReviewDatabase
from app.database.v2 import utcnow
from app.extractors import extract_items, extract_requirements
from app.integrations import ConfigStore, LLMClient, MinerUClient
from app.models import ExtractedItem, Finding, PageText, Requirement
from app.parsers import parse_document
from app.rag.vector_index import DocumentVectorIndex
from app.rules import AuditEngine
from app.utils import save_upload


LOGGER = logging.getLogger(__name__)
PRIORITIES = {"supplemental": 0, "technical": 1, "drawing": 1, "enterprise": 2, "standard": 3, "other": 4}


class ReviewService:
    def __init__(self, db: ReviewDatabase, config_store: ConfigStore, uploads: Path,
                 standards: Path, vector_root: Path, max_upload_mb: int = 100):
        self.db, self.config_store = db, config_store
        self.uploads, self.standards, self.vector_root = uploads, standards, vector_root
        self.max_bytes = max_upload_mb * 1024 * 1024

    def create_review(self, template_id: int | None, selected_basis: list[str],
                      supplier_files: Iterable[object], supplemental_files: Iterable[object]) -> str:
        supplier_files = list(supplier_files)
        supplemental_files = list(supplemental_files)
        if not supplier_files:
            raise ValueError("至少需要上传一份供应商质量文件")
        batch_id = self.db.create_batch(template_id)
        destination = self.uploads / batch_id
        for uploaded in supplier_files:
            document_id = self._save_uploaded(uploaded, "supplier", "supplier", destination)
            self.db.attach_document(batch_id, document_id, "supplier", 4)
        for uploaded in supplemental_files:
            document_id = self._save_uploaded(uploaded, "basis", "technical", self.standards / "supplemental" / batch_id)
            self.db.attach_document(batch_id, document_id, "supplemental_basis", 0)
        basis_ids = selected_basis or self._default_basis(template_id)
        for document_id in basis_ids:
            row = self.db.one("SELECT document_kind FROM documents WHERE id=? AND library_code='basis'", (document_id,))
            if row:
                self.db.attach_document(batch_id, document_id, "selected_basis", PRIORITIES.get(row["document_kind"], 3))
        return batch_id

    def import_basis(self, uploaded: object, kind: str) -> str:
        document_id = self._save_uploaded(uploaded, "basis", kind, self.standards / kind)
        self.process_document(document_id, requirement_priority=PRIORITIES.get(kind, 3))
        return document_id

    def _save_uploaded(self, uploaded: object, library: str, kind: str, destination: Path) -> str:
        name = str(getattr(uploaded, "name", "document.pdf"))
        if hasattr(uploaded, "seek"):
            uploaded.seek(0)
        path = save_upload(uploaded, name, destination, self.max_bytes)  # type: ignore[arg-type]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return self.db.add_document(library=library, kind=kind, original_name=name, stored_path=str(path),
                                    sha256=digest, mime_type=mimetypes.guess_type(name)[0] or "")

    def _default_basis(self, template_id: int | None) -> list[str]:
        if not template_id:
            return []
        return [row["document_id"] for row in self.db.query("SELECT document_id FROM template_basis WHERE template_id=?", (template_id,))]

    def process_batch(self, batch_id: str) -> None:
        try:
            self._activity(batch_id, status="running", stage="解析文件", progress=5, error="",
                           activity="正在读取批次文件清单", resource="SQLite 本地数据库")
            rows = self.db.query(
                """SELECT d.*,bd.role,bd.priority FROM batch_documents bd
                   JOIN documents d ON d.id=bd.document_id WHERE bd.batch_id=? ORDER BY bd.priority,d.created_at""", (batch_id,)
            )
            for index, row in enumerate(rows):
                self._activity(batch_id, current_file=row["original_name"],
                               progress=5 + int(40 * index / max(1, len(rows))),
                               activity=f"准备处理第 {index + 1}/{len(rows)} 份文件",
                               resource="本机文件系统")
                if row["parse_status"] != "completed" or row["index_fingerprint"] != self.config_store.get().embedding_fingerprint:
                    self.process_document(row["id"], requirement_priority=int(row["priority"]) if row["role"] != "supplier" else None,
                                          batch_id=batch_id)
            self._activity(batch_id, stage="执行确定性规则", progress=60, current_file="",
                           activity="正在比较实测值、单位、材料及必检项目", resource="本机 CPU · 确定性规则引擎")
            findings = self._audit(batch_id)
            settings = self.config_store.get()
            self._activity(batch_id, stage="生成问题说明", progress=82,
                           activity=f"准备为 {len(findings)} 个问题生成可读说明",
                           resource=_resource("LLM", settings.llm_base_url, settings.llm_model))
            findings = self._explain(batch_id, findings)
            self._activity(batch_id, stage="保存审核结果", progress=97,
                           activity="正在写入问题证据、判断逻辑和审核汇总", resource="SQLite 本地数据库")
            with self.db.connect() as connection:
                connection.execute("DELETE FROM findings WHERE batch_id=?", (batch_id,))
                connection.executemany(
                    """INSERT INTO findings(batch_id,category,severity,item,description,actual,requirement,source_file,
                       source_page,source_text,standard_file,standard_page,standard_clause,logic,suggestion,confidence,status,metadata,created_at)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    [(batch_id, f.category, f.severity, f.item, f.description, f.actual, f.requirement, f.source_file,
                      f.source_page, f.source_text, f.standard_file, f.standard_page, f.standard_clause, f.logic,
                      f.suggestion, f.confidence, f.status, json.dumps(f.metadata, ensure_ascii=False), utcnow()) for f in findings],
                )
            summary = {level: sum(item.severity == level for item in findings) for level in ("Critical", "Major", "Minor", "Warning", "Review")}
            summary["total"] = len(findings)
            self._activity(batch_id, status="completed", stage="审核完成", progress=100, summary=summary,
                           activity="所有阶段已完成，结果可以查看", resource="本机")
            self.db.execute("UPDATE jobs SET status='completed',updated_at=? WHERE batch_id=?", (utcnow(), batch_id))
        except Exception as exc:
            LOGGER.exception("审核批次失败：%s", batch_id)
            message = str(exc)[:1000]
            self._activity(batch_id, status="failed", stage="处理失败", error=message,
                           activity="审核任务发生错误", resource="请查看失败原因")
            self.db.execute("UPDATE jobs SET status='failed',error=?,updated_at=? WHERE batch_id=?", (message, utcnow(), batch_id))

    def process_document(self, document_id: str, requirement_priority: int | None = None,
                         batch_id: str | None = None) -> None:
        row = self.db.one("SELECT * FROM documents WHERE id=?", (document_id,))
        if not row:
            raise ValueError("文档不存在")
        path = Path(row["stored_path"])
        settings = self.config_store.get()
        pages: list[PageText]
        ocr_status = "not_needed"
        try:
            self._activity(batch_id, activity=f"正在读取 {row['original_name']} 的本地文本层",
                           resource="本机 CPU · PyMuPDF/python-docx/openpyxl")
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                raise ValueError("图像需要 OCR")
            pages = parse_document(path)
            text_chars = sum(len(page.text.strip()) for page in pages)
            if path.suffix.lower() == ".pdf" and text_chars < max(80, len(pages) * 30):
                raise ValueError("PDF 文本层内容不足")
        except ValueError as exc:
            if path.suffix.lower() not in {".pdf", ".jpg", ".jpeg", ".png"}:
                raise
            if not settings.ocr_base_url:
                raise RuntimeError(f"{exc}，且未配置 OCR 服务") from exc
            self._activity(batch_id, activity=f"正在提交 {row['original_name']} 并等待 OCR 识别结果",
                           resource=_resource("OCR", settings.ocr_base_url, settings.ocr_backend))
            markdown = MinerUClient(settings).ocr(path)
            pages = [PageText(1, markdown)]
            ocr_status = "completed"
        raw_text = "\n\n".join(page.text for page in pages)
        self._activity(batch_id, activity=f"正在从 {row['original_name']} 提取结构化字段和审核条款",
                       resource="本机 CPU · 正则提取器")
        with self.db.connect() as connection:
            connection.execute("DELETE FROM extracted_data WHERE document_id=?", (document_id,))
            connection.execute("DELETE FROM requirement_rules WHERE document_id=?", (document_id,))
            if row["library_code"] == "supplier":
                items = extract_items(pages)
                connection.executemany(
                    """INSERT INTO extracted_data(document_id,key,raw_value,normalized_value,unit,page,source_text,category)
                       VALUES(?,?,?,?,?,?,?,?)""",
                    [(document_id, item.key, item.raw, str(item.value if item.value is not None else ""), item.unit,
                      item.page, item.source_text, item.category) for item in items],
                )
            else:
                requirements = extract_requirements(pages, row["original_name"])
                connection.executemany(
                    """INSERT INTO requirement_rules(document_id,item,operator,value,upper_value,unit,raw,source_page,
                       clause,priority,required,confirmed,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    [(document_id, req.item, req.operator, req.value, req.upper_value, req.unit, req.raw,
                      req.source_page, req.clause, requirement_priority or PRIORITIES.get(row["document_kind"], 3),
                      int(req.required), 1, utcnow()) for req in requirements],
                )
        self._activity(batch_id, activity=f"正在为 {row['original_name']} 生成向量并写入独立索引",
                       resource=_resource("Embedding", settings.embedding_base_url, settings.embedding_model))
        count, index_status = DocumentVectorIndex(self.vector_root, settings).index(document_id, pages)
        self.db.execute(
            """UPDATE documents SET page_count=?,page_text=?,raw_text=?,markdown=?,parse_status='completed',
               ocr_status=?,index_status=?,index_fingerprint=?,index_collection=?,error='' WHERE id=?""",
            (len(pages), json.dumps([{"page": p.page, "text": p.text} for p in pages], ensure_ascii=False),
             raw_text, raw_text, ocr_status, index_status, settings.embedding_fingerprint,
             f"{document_id}/document_chunks" if count else "", document_id),
        )
        self._activity(batch_id, activity=f"{row['original_name']} 已完成解析、提取和索引",
                       resource=f"ChromaDB · {index_status}")

    def _audit(self, batch_id: str) -> list[Finding]:
        document_rows = self.db.query(
            """SELECT d.id,d.original_name FROM batch_documents bd JOIN documents d ON d.id=bd.document_id
               WHERE bd.batch_id=? AND bd.role='supplier'""", (batch_id,)
        )
        documents: dict[str, list[ExtractedItem]] = {}
        for document in document_rows:
            rows = self.db.query("SELECT * FROM extracted_data WHERE document_id=?", (document["id"],))
            values: list[ExtractedItem] = []
            for row in rows:
                value: float | str | None = row["normalized_value"]
                if row["category"] == "measurement":
                    try: value = float(value)
                    except ValueError: value = None
                values.append(ExtractedItem(row["key"], row["raw_value"], value, row["unit"], row["page"], row["source_text"], row["category"]))
            documents[document["original_name"]] = values
        rule_rows = self.db.query(
            """SELECT r.*,d.original_name source_file FROM batch_documents bd
               JOIN requirement_rules r ON r.document_id=bd.document_id JOIN documents d ON d.id=r.document_id
               WHERE bd.batch_id=? AND bd.role IN ('selected_basis','supplemental_basis') AND r.confirmed=1
               ORDER BY bd.priority,r.priority""", (batch_id,)
        )
        requirements = [Requirement(row["item"], row["operator"], _number_or_text(row["value"]),
            _optional_float(row["upper_value"]), row["unit"], row["raw"], row["source_file"], row["source_page"],
            row["clause"], bool(row["required"])) for row in rule_rows]
        batch = self.db.one("SELECT template_id FROM review_batches WHERE id=?", (batch_id,))
        if batch and batch["template_id"]:
            template = self.db.one("SELECT required_items FROM audit_templates WHERE id=?", (batch["template_id"],))
            if template:
                for item in json.loads(template["required_items"] or "[]"):
                    if not any(req.item == item and req.operator == "exists" for req in requirements):
                        requirements.append(Requirement(str(item), "exists", None, raw=f"审核模板必检项：{item}", required=True))
        findings = AuditEngine().audit(documents, requirements)
        if not rule_rows:
            findings.append(Finding("待人工确认", "Review", "审核依据", "没有找到明确的结构化审核依据，系统未推测合格性",
                                    requirement="缺少明确审核依据", logic="所选依据未提取出可执行规则"))
        return findings

    def _explain(self, batch_id: str, findings: list[Finding]) -> list[Finding]:
        settings = self.config_store.get()
        if not settings.llm_model or not findings:
            self._activity(batch_id, activity="未配置 LLM，保留确定性规则生成的说明",
                           resource="本机规则引擎")
            return findings
        client = LLMClient(settings)
        group_size = 20
        groups = [findings[offset:offset + group_size] for offset in range(0, len(findings), group_size)]
        for group_index, group in enumerate(groups, start=1):
            offset = (group_index - 1) * group_size
            inputs = [{"index": offset + index, "logic": finding.logic[:500], "actual": finding.actual[:300],
                       "requirement": finding.requirement[:500]} for index, finding in enumerate(group)]
            self._activity(
                batch_id, progress=82 + int(14 * (group_index - 1) / max(1, len(groups))),
                activity=f"正在调用 LLM 批量解释第 {group_index}/{len(groups)} 组（{len(group)} 个问题）",
                resource=_resource("LLM", settings.llm_base_url, settings.llm_model),
            )
            prompt = ("你是质量审核解释器。程序规则结论不可更改，不得引用输入以外的标准。"
                      "为每个输入生成简短原因、整改建议和0到1置信度。"
                      f"\n输入：{json.dumps(inputs, ensure_ascii=False)}"
                      "\n返回 JSON：{\"items\":[{\"index\":0,\"reason\":\"...\",\"suggestion\":\"...\",\"confidence\":0.0}]}")
            try:
                result = client.generate_json(prompt)
                for item in result.get("items", []):
                    index = int(item.get("index", -1))
                    if not 0 <= index < len(findings):
                        continue
                    finding = findings[index]
                    finding.description = str(item.get("reason") or finding.description)
                    finding.suggestion = str(item.get("suggestion") or finding.suggestion)
                    finding.confidence = max(0.0, min(1.0, float(item.get("confidence", finding.confidence))))
            except Exception as exc:
                LOGGER.warning("LLM 解释失败，保留规则结论：%s", type(exc).__name__)
        return findings

    def _activity(self, batch_id: str | None, **values: object) -> None:
        if not batch_id:
            return
        values["heartbeat_at"] = utcnow()
        self.db.update_batch(batch_id, **values)


def _number_or_text(value: object) -> float | str | None:
    if value is None:
        return None
    try: return float(value)
    except (TypeError, ValueError): return str(value)


def _optional_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try: return float(value)
    except (TypeError, ValueError): return None


def _resource(kind: str, base_url: str, model: str) -> str:
    host = urlsplit(base_url).netloc or base_url
    detail = model.strip() or "未指定模型"
    return f"{kind} · {detail} · {host}"

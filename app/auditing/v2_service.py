from __future__ import annotations

import hashlib
import json
import logging
import mimetypes
import re
import shutil
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Iterable
from urllib.parse import urlsplit

import pymupdf

from app.auditing.expert_review import (
    FASTENER_REQUIRED_CHECKS,
    FASTENER_REVIEW_INSTRUCTIONS,
    FASTENER_TEMPLATE_NAME,
    ExpertDocument,
    build_expert_prompt,
    deterministic_fastener_audit,
    extract_supplier_names,
    findings_from_llm,
    supplier_names_from_llm,
)
from app.database import ReviewDatabase
from app.database.v2 import utcnow
from app.extractors import extract_filename_items, extract_items, extract_requirements
from app.integrations import ConfigStore, LLMClient, MinerUClient
from app.models import ExtractedItem, Finding, PageText, Requirement
from app.parsers import parse_document
from app.rag.vector_index import DocumentVectorIndex
from app.rules import AuditEngine, GENERIC_TEMPLATE_NAME, GenericDocument, classify_document, run_generic_rules, seed_generic_rules
from app.utils import save_upload


LOGGER = logging.getLogger(__name__)
PRIORITIES = {"supplemental": 0, "technical": 1, "drawing": 1, "enterprise": 2, "standard": 3, "other": 4}


class ReviewCancelled(RuntimeError):
    pass


class ReviewService:
    def __init__(self, db: ReviewDatabase, config_store: ConfigStore, uploads: Path,
                 standards: Path, vector_root: Path, max_upload_mb: int = 100):
        self.db, self.config_store = db, config_store
        self.uploads, self.standards, self.vector_root = uploads, standards, vector_root
        self.max_bytes = max_upload_mb * 1024 * 1024
        seed_generic_rules(self.db)
        self._ensure_fastener_template()
        self._backfill_supplier_names()
        self._backfill_v04_metadata()

    def _ensure_fastener_template(self) -> None:
        row = self.db.one("SELECT id FROM audit_templates WHERE name=?", (FASTENER_TEMPLATE_NAME,))
        if row:
            self.db.execute(
                "UPDATE audit_templates SET review_instructions=?,required_document_types=?,required_items=? WHERE id=?",
                (FASTENER_REVIEW_INSTRUCTIONS, json.dumps(["COC", "COI/MTR"], ensure_ascii=False),
                 json.dumps(FASTENER_REQUIRED_CHECKS, ensure_ascii=False), row["id"]),
            )
            return
        self.db.execute(
            """INSERT INTO audit_templates(name,description,required_document_types,required_items,review_instructions,
               enabled,is_default,created_at) VALUES(?,?,?,?,?,1,0,?)""",
            (FASTENER_TEMPLATE_NAME, "按文件/WDC关系、表格实测数据和逐页证据审核 CUSTOMER 紧固件质量文件。",
             json.dumps(["COC", "COI/MTR"], ensure_ascii=False),
             json.dumps(FASTENER_REQUIRED_CHECKS, ensure_ascii=False), FASTENER_REVIEW_INSTRUCTIONS, utcnow()),
        )

    def _backfill_supplier_names(self) -> None:
        rows = self.db.query(
            """SELECT id,original_name,stored_path,page_text,ocr_status FROM documents
               WHERE library_code='supplier' AND parse_status='completed'
                 AND (supplier_name='' OR supplier_name LIKE '%�%')"""
        )
        for row in rows:
            supplier_name = _join_names(extract_supplier_names(_expert_document(row)))
            self.db.execute("UPDATE documents SET supplier_name=? WHERE id=?", (supplier_name, row["id"]))
            self._refresh_supplier_names_for_document(row["id"])

    def _backfill_v04_metadata(self) -> None:
        rows = self.db.query(
            """SELECT d.* FROM documents d WHERE d.parse_status='completed' AND
               (d.detected_type='' OR NOT EXISTS (SELECT 1 FROM document_fts f WHERE f.document_id=d.id))"""
        )
        for row in rows:
            pages = _expert_document(row).pages
            kind, confidence = classify_document(row["original_name"], pages)
            path = Path(str(row["stored_path"]))
            fields = self.db.query("SELECT * FROM extracted_data WHERE document_id=?", (row["id"],))
            with self.db.connect() as connection:
                connection.execute("UPDATE documents SET detected_type=?,type_confidence=? WHERE id=?", (kind, confidence, row["id"]))
                connection.execute("DELETE FROM document_fts WHERE document_id=?", (row["id"],))
                connection.executemany("INSERT INTO document_fts(document_id,page,content) VALUES(?,?,?)",
                                       [(row["id"], page.page, page.text) for page in pages if page.text.strip()])
                existing = connection.execute("SELECT 1 FROM document_fields WHERE document_id=? LIMIT 1", (row["id"],)).fetchone()
                if not existing:
                    connection.executemany(
                        """INSERT INTO document_fields(document_id,field_key,raw_value,normalized_value,unit,page,source_text,
                           extraction_method,confidence,bbox) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                        [(row["id"], item["key"], item["raw_value"], item["normalized_value"], item["unit"], item["page"],
                          item["source_text"], "legacy_text_regex", .85,
                          json.dumps(_locate_bbox(path, int(item["page"]), str(item["source_text"] or item["raw_value"])))) for item in fields],
                    )

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

    def retry_review(self, source_batch_id: str) -> str:
        source = self.db.one("SELECT template_id FROM review_batches WHERE id=?", (source_batch_id,))
        if not source:
            raise ValueError("原审核批次不存在")
        batch_id = self.db.create_batch(source["template_id"])
        rows = self.db.query("SELECT document_id,role,priority FROM batch_documents WHERE batch_id=?", (source_batch_id,))
        for row in rows:
            self.db.attach_document(batch_id, row["document_id"], row["role"], row["priority"])
        return batch_id

    def purge_review(self, batch_id: str) -> None:
        private = self.db.purge_batch(batch_id)
        for row in private:
            path = Path(str(row["stored_path"]))
            if path.is_file():
                path.unlink(missing_ok=True)
            vector_path = self.vector_root / str(row["id"])
            if vector_path.is_dir():
                shutil.rmtree(vector_path)

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
            self._check_cancel(batch_id)
            self._activity(batch_id, status="running", stage="解析文件", progress=5, error="", started_at=utcnow(),
                           activity="正在读取批次文件清单", resource="SQLite 本地数据库")
            rows = self.db.query(
                """SELECT d.*,bd.role,bd.priority FROM batch_documents bd
                   JOIN documents d ON d.id=bd.document_id WHERE bd.batch_id=? ORDER BY bd.priority,d.created_at""", (batch_id,)
            )
            for index, row in enumerate(rows):
                self._check_cancel(batch_id)
                self._activity(batch_id, current_file=row["original_name"],
                               progress=5 + int(40 * index / max(1, len(rows))),
                               activity=f"准备处理第 {index + 1}/{len(rows)} 份文件",
                               resource="本机文件系统")
                if row["parse_status"] != "completed" or row["index_fingerprint"] != self.config_store.get().embedding_fingerprint:
                    self.process_document(row["id"], requirement_priority=int(row["priority"]) if row["role"] != "supplier" else None,
                                          batch_id=batch_id)
                self._check_cancel(batch_id)
            self._activity(batch_id, stage="执行确定性规则", progress=60, current_file="",
                           activity="正在比较实测值、单位、材料及必检项目", resource="本机 CPU · 确定性规则引擎")
            findings = self._audit(batch_id)
            self._check_cancel(batch_id)
            settings = self.config_store.get()
            self._activity(batch_id, stage="LLM 专家复核", progress=82,
                           activity=f"准备让 LLM 基于逐页证据复核 {len(findings)} 个规则结果并发现遗漏",
                           resource=_resource("LLM", settings.llm_base_url, settings.llm_model))
            findings = self._expert_review(batch_id, findings)
            self._check_cancel(batch_id)
            self._activity(batch_id, stage="保存审核结果", progress=97,
                           activity="正在写入问题证据、判断逻辑和审核汇总", resource="SQLite 本地数据库")
            for finding in findings:
                self._enrich_finding_evidence(batch_id, finding)
            with self.db.connect() as connection:
                connection.execute("DELETE FROM findings WHERE batch_id=?", (batch_id,))
                connection.executemany(
                    """INSERT INTO findings(batch_id,category,severity,item,description,actual,requirement,source_file,
                       source_page,source_text,standard_file,standard_page,standard_clause,logic,suggestion,confidence,status,metadata,
                       rule_code,rule_version,document_type,extraction_confidence,decision_confidence,created_at)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    [(batch_id, f.category, f.severity, f.item, f.description, f.actual, f.requirement, f.source_file,
                      f.source_page, f.source_text, f.standard_file, f.standard_page, f.standard_clause, f.logic,
                      f.suggestion, f.confidence, f.status, json.dumps(f.metadata, ensure_ascii=False), f.rule_code,
                      f.rule_version, f.document_type, f.extraction_confidence, f.decision_confidence, utcnow()) for f in findings],
                )
                saved = connection.execute("SELECT id,metadata FROM findings WHERE batch_id=? ORDER BY id", (batch_id,)).fetchall()
                for saved_row in saved:
                    metadata = json.loads(saved_row["metadata"] or "{}")
                    for evidence in metadata.get("evidence", []):
                        connection.execute(
                            """INSERT INTO finding_evidence(finding_id,document_id,page,source_text,bbox,evidence_type,matched)
                               VALUES(?,?,?,?,?,?,?)""",
                            (saved_row["id"], evidence.get("document_id"), int(evidence.get("page") or 1),
                             str(evidence.get("source_text") or ""), json.dumps(evidence.get("bbox") or [], ensure_ascii=False),
                             str(evidence.get("evidence_type") or "source"), int(bool(evidence.get("matched")))),
                        )
            summary = {level: sum(item.severity == level for item in findings) for level in ("Critical", "Major", "Minor", "Warning", "Review")}
            summary["total"] = len(findings)
            self._activity(batch_id, status="completed", stage="审核完成", progress=100, summary=summary, completed_at=utcnow(),
                           activity="所有阶段已完成，结果可以查看", resource="本机")
            self.db.execute("UPDATE jobs SET status='completed',updated_at=? WHERE batch_id=?", (utcnow(), batch_id))
        except ReviewCancelled:
            self.db.mark_cancelled(batch_id)
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
        self._check_cancel(batch_id)
        pages: list[PageText]
        ocr_status = "not_needed"
        try:
            self._activity(batch_id, activity=f"正在读取 {row['original_name']} 的本地文本层",
                           resource="本机 CPU · PyMuPDF/python-docx/openpyxl")
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                pages = [PageText(1, "")]
            else:
                pages = parse_document(path)
        except ValueError:
            if path.suffix.lower() not in {".pdf", ".jpg", ".jpeg", ".png"}:
                raise
            pages = [PageText(1, "")]
        ocr_capable = path.suffix.lower() in {".pdf", ".jpg", ".jpeg", ".png"}
        sparse_pages = [page.page for page in pages if ocr_capable and len(re.sub(r"\s+", "", page.text)) < 40]
        if sparse_pages and settings.ocr_base_url:
            self._activity(batch_id, activity=f"正在提交 {row['original_name']} 并等待 OCR 识别结果",
                           resource=_resource("OCR", settings.ocr_base_url, settings.ocr_backend))
            try:
                markdown = MinerUClient(settings).ocr(path)
                self._check_cancel(batch_id)
                pages = _merge_ocr_text(pages, markdown, sparse_pages)
                ocr_status = "completed"
            except ReviewCancelled:
                raise
            except Exception as exc:
                LOGGER.warning("逐页 OCR 不可用，保留已有文本并标记人工复核：%s", type(exc).__name__)
                ocr_status = "failed"
                self._activity(batch_id, activity=f"OCR 未完成；保留 {row['original_name']} 已提取页面并标记空白页",
                               resource=_resource("OCR", settings.ocr_base_url, settings.ocr_backend))
        elif sparse_pages:
            ocr_status = "pending"
        raw_text = "\n\n".join(page.text for page in pages)
        detected_type, type_confidence = classify_document(row["original_name"], pages)
        supplier_name = " / ".join(extract_supplier_names(pages)) if row["library_code"] == "supplier" else ""
        self._activity(batch_id, activity=f"正在从 {row['original_name']} 提取结构化字段和审核条款",
                       resource="本机 CPU · 正则提取器")
        with self.db.connect() as connection:
            connection.execute("DELETE FROM extracted_data WHERE document_id=?", (document_id,))
            connection.execute("DELETE FROM document_fields WHERE document_id=?", (document_id,))
            connection.execute("DELETE FROM document_fts WHERE document_id=?", (document_id,))
            connection.executemany("INSERT INTO document_fts(document_id,page,content) VALUES(?,?,?)",
                                   [(document_id, page.page, page.text) for page in pages if page.text.strip()])
            connection.execute("DELETE FROM requirement_rules WHERE document_id=?", (document_id,))
            if row["library_code"] == "supplier":
                items = [*extract_items(pages), *extract_filename_items(row["original_name"])]
                connection.executemany(
                    """INSERT INTO extracted_data(document_id,key,raw_value,normalized_value,unit,page,source_text,category)
                       VALUES(?,?,?,?,?,?,?,?)""",
                    [(document_id, item.key, item.raw, str(item.value if item.value is not None else ""), item.unit,
                      item.page, item.source_text, item.category) for item in items],
                )
                connection.executemany(
                    """INSERT INTO document_fields(document_id,field_key,raw_value,normalized_value,unit,page,source_text,
                       extraction_method,confidence,bbox) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                    [(document_id, item.key, item.raw, str(item.value if item.value is not None else ""), item.unit,
                      item.page, item.source_text, "text_regex", .92,
                      json.dumps(_locate_bbox(path, item.page, item.source_text or item.raw))) for item in items],
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
        self._check_cancel(batch_id)
        self.db.execute(
            """UPDATE documents SET page_count=?,page_text=?,raw_text=?,markdown=?,supplier_name=?,detected_type=?,type_confidence=?,parse_status='completed',
               ocr_status=?,index_status=?,index_fingerprint=?,index_collection=?,error='' WHERE id=?""",
            (len(pages), json.dumps([{"page": p.page, "text": p.text} for p in pages], ensure_ascii=False),
             raw_text, raw_text, supplier_name, detected_type, type_confidence, ocr_status, index_status, settings.embedding_fingerprint,
             f"{document_id}/document_chunks" if count else "", document_id),
        )
        self._refresh_supplier_names_for_document(document_id)
        self._activity(batch_id, activity=f"{row['original_name']} 已完成解析、提取和索引",
                       resource=f"ChromaDB · {index_status}")

    def _audit(self, batch_id: str) -> list[Finding]:
        document_rows = self.db.query(
            """SELECT d.id,d.original_name,d.stored_path,d.page_text,d.ocr_status,d.supplier_name,d.detected_type,d.type_confidence FROM batch_documents bd JOIN documents d ON d.id=bd.document_id
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
        batch = self.db.one("SELECT template_id,template_snapshot FROM review_batches WHERE id=?", (batch_id,))
        template = self.db.one("SELECT * FROM audit_templates WHERE id=?", (batch["template_id"],)) if batch and batch["template_id"] else None
        findings = AuditEngine().audit(documents, requirements)
        generic_documents: list[GenericDocument] = []
        for document in document_rows:
            fields = documents.get(document["original_name"], [])
            pages = _expert_document(document).pages
            kind, confidence = classify_document(document["original_name"], pages)
            generic_documents.append(GenericDocument(document["id"], document["original_name"], Path(document["stored_path"]),
                                                       pages, fields, kind, confidence, str(document.get("supplier_name") or "")))
        if template and template["name"] == GENERIC_TEMPLATE_NAME:
            try:
                snapshot_rules = json.loads(batch.get("template_snapshot") or "{}").get("rules", []) if batch else []
            except (ValueError, TypeError):
                snapshot_rules = []
            if snapshot_rules:
                enabled_codes = {row["rule_code"] for row in snapshot_rules if row.get("enabled")}
                version_lookup = {row["rule_code"]: int(row.get("rule_version") or 1) for row in snapshot_rules}
            else:
                current_rules = self.db.query(
                    "SELECT rule_code,rule_version FROM template_rule_versions WHERE template_id=? AND enabled=1", (template["id"],)
                )
                enabled_codes = {row["rule_code"] for row in current_rules}
                version_lookup = {row["rule_code"]: int(row["rule_version"]) for row in current_rules}
            generic_findings = run_generic_rules(generic_documents, enabled_codes)
            for finding in generic_findings:
                finding.rule_version = version_lookup.get(finding.rule_code, 1)
            findings.extend(generic_findings)
        if template and template["name"] == FASTENER_TEMPLATE_NAME:
            expert_documents = [_expert_document(row) for row in document_rows]
            findings.extend(deterministic_fastener_audit(expert_documents))
        if not rule_rows and not (template and template.get("review_instructions", "").strip()):
            findings.append(Finding("待人工确认", "Review", "审核依据", "没有找到明确的结构化审核依据，系统未推测合格性",
                                    requirement="缺少明确审核依据", logic="所选依据未提取出可执行规则"))
        return _dedupe_findings(findings)

    def _expert_review(self, batch_id: str, findings: list[Finding]) -> list[Finding]:
        settings = self.config_store.get()
        if not settings.llm_base_url:
            self._activity(batch_id, activity="未配置 LLM Base URL，保留确定性专家规则结果", resource="本机规则引擎")
            return findings
        rows = self.db.query(
            """SELECT d.id,d.original_name,d.stored_path,d.page_text,d.ocr_status FROM batch_documents bd
               JOIN documents d ON d.id=bd.document_id WHERE bd.batch_id=? AND bd.role='supplier'""", (batch_id,)
        )
        documents = [_expert_document(row) for row in rows]
        self._check_cancel(batch_id)
        if not documents:
            return findings
        batch = self.db.one("SELECT template_id FROM review_batches WHERE id=?", (batch_id,))
        template = self.db.one("SELECT review_instructions FROM audit_templates WHERE id=?", (batch["template_id"],)) if batch and batch["template_id"] else None
        instructions = str(template.get("review_instructions") or "") if template else ""
        basis_rows = self.db.query(
            """SELECT d.original_name,d.raw_text FROM batch_documents bd JOIN documents d ON d.id=bd.document_id
               WHERE bd.batch_id=? AND bd.role IN ('selected_basis','supplemental_basis') ORDER BY bd.priority""", (batch_id,)
        )
        basis = self._hybrid_basis_context(batch_id, basis_rows)
        self._activity(batch_id, activity="正在调用 LLM 逐页复核文件类型、追溯关系和表格证据",
                       resource=_resource("LLM", settings.llm_base_url, settings.llm_model))
        try:
            payload = LLMClient(settings).generate_json(build_expert_prompt(documents, instructions, findings, basis))
            self._check_cancel(batch_id)
            discovered = findings_from_llm(payload, documents)
            llm_names = supplier_names_from_llm(payload, documents)
            for row in rows:
                names = [*extract_supplier_names(_expert_document(row)), *llm_names.get(row["original_name"], [])]
                combined = _join_names(names)
                if combined:
                    self.db.execute("UPDATE documents SET supplier_name=? WHERE id=?", (combined, row["id"]))
                    self._refresh_supplier_names_for_document(row["id"])
            self._activity(batch_id, progress=94, activity=f"LLM 专家复核新增 {len(discovered)} 个有原页证据的问题",
                           resource=_resource("LLM", settings.llm_base_url, settings.llm_model))
            return _dedupe_findings([*findings, *discovered])
        except ReviewCancelled:
            raise
        except Exception as exc:
            LOGGER.warning("LLM 专家复核失败，保留确定性结果：%s", type(exc).__name__)
            self._activity(batch_id, progress=94, activity="LLM 专家复核未完成，已保留确定性规则结果",
                           resource=_resource("LLM", settings.llm_base_url, settings.llm_model))
            return findings

    def _hybrid_basis_context(self, batch_id: str, basis_rows: list[dict[str, object]]) -> str:
        """Combine exact FTS and vector hits, restricted to the basis selected for this batch."""
        ids = [row["id"] for row in self.db.query(
            """SELECT d.id FROM batch_documents bd JOIN documents d ON d.id=bd.document_id
               WHERE bd.batch_id=? AND bd.role IN ('selected_basis','supplemental_basis')""", (batch_id,)
        )]
        if not ids:
            return ""
        names = {row["id"]: row["original_name"] for row in self.db.query(
            f"SELECT id,original_name FROM documents WHERE id IN ({','.join('?' for _ in ids)})", ids
        )}
        ranked: dict[tuple[str, int, str], float] = {}
        placeholders = ",".join("?" for _ in ids)
        try:
            lexical = self.db.query(
                f"""SELECT document_id,page,content,bm25(document_fts) score FROM document_fts
                    WHERE document_fts MATCH ? AND document_id IN ({placeholders}) ORDER BY score LIMIT 12""",
                ["标准 OR 要求 OR specification OR requirement OR chemical OR mechanical", *ids],
            )
            for rank, row in enumerate(lexical, 1):
                ranked[(row["document_id"], int(row["page"]), str(row["content"])[:2500])] = 1 / (60 + rank)
        except Exception:
            pass
        try:
            vector = DocumentVectorIndex(self.vector_root, self.config_store.get()).search(
                ids, "材料牌号 炉号 化学成分 机械性能 尺寸 检验标准 合格要求", top_k=12
            )
            for rank, row in enumerate(vector, 1):
                key = (row["document_id"], int(row["page"]), str(row["text"])[:2500])
                ranked[key] = ranked.get(key, 0) + 1 / (60 + rank)
        except Exception:
            pass
        if ranked:
            items = sorted(ranked, key=ranked.get, reverse=True)[:12]
            return "\n\n".join(f"[依据={names.get(doc_id, doc_id)}][页={page}]\n{text}" for doc_id, page, text in items)
        return "\n".join(f"[依据={row['original_name']}]\n{str(row['raw_text'])[:12000]}" for row in basis_rows)

    def _refresh_supplier_names_for_document(self, document_id: str) -> None:
        batch_rows = self.db.query("SELECT batch_id FROM batch_documents WHERE document_id=? AND role='supplier'", (document_id,))
        for batch_row in batch_rows:
            rows = self.db.query(
                """SELECT d.supplier_name FROM batch_documents bd JOIN documents d ON d.id=bd.document_id
                   WHERE bd.batch_id=? AND bd.role='supplier' ORDER BY d.created_at,d.id""",
                (batch_row["batch_id"],),
            )
            supplier_name = _join_names(
                part.strip() for row in rows for part in str(row.get("supplier_name") or "").split("/") if part.strip()
            )
            self.db.execute("UPDATE review_batches SET supplier_name=?,updated_at=? WHERE id=?",
                            (supplier_name, utcnow(), batch_row["batch_id"]))

    def _enrich_finding_evidence(self, batch_id: str, finding: Finding) -> None:
        evidence = finding.metadata.get("evidence") if isinstance(finding.metadata, dict) else None
        if not isinstance(evidence, list):
            evidence = [{"file": finding.source_file, "page": finding.source_page,
                         "source_text": finding.source_text or finding.actual, "evidence_type": "source"}]
        for item in evidence:
            if not isinstance(item, dict):
                continue
            if not item.get("document_id"):
                row = self.db.one(
                    """SELECT d.id FROM batch_documents bd JOIN documents d ON d.id=bd.document_id
                       WHERE bd.batch_id=? AND d.original_name=? LIMIT 1""", (batch_id, item.get("file") or finding.source_file)
                )
                if row: item["document_id"] = row["id"]
            if item.get("document_id") and not item.get("bbox") and item.get("evidence_type") != "absence":
                field = self.db.one(
                    """SELECT bbox FROM document_fields WHERE document_id=? AND page=? AND bbox<>''
                       AND (source_text=? OR raw_value=?) ORDER BY confidence DESC LIMIT 1""",
                    (item["document_id"], int(item.get("page") or 1), str(item.get("source_text") or ""), finding.actual),
                )
                if field:
                    try: item["bbox"] = json.loads(field["bbox"] or "[]")
                    except ValueError: item["bbox"] = []
            item["matched"] = bool(item.get("bbox"))
        finding.metadata["evidence"] = evidence

    def _explain(self, batch_id: str, findings: list[Finding]) -> list[Finding]:
        settings = self.config_store.get()
        if not settings.llm_base_url or not findings:
            self._activity(batch_id, activity="未配置 LLM Base URL，保留确定性规则生成的说明",
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
        self.db.execute("INSERT INTO job_events(batch_id,stage,activity,resource,created_at) VALUES(?,?,?,?,?)",
                        (batch_id, str(values.get("stage") or ""), str(values.get("activity") or ""),
                         str(values.get("resource") or ""), utcnow()))

    def _check_cancel(self, batch_id: str | None) -> None:
        if batch_id and self.db.is_cancel_requested(batch_id):
            raise ReviewCancelled("审核已由用户停止")


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


def _locate_bbox(path: Path, page_number: int, source_text: str) -> list[float]:
    """Best-effort PDF coordinates; empty means the UI must not draw a synthetic box."""
    if path.suffix.casefold() != ".pdf" or not path.is_file() or not source_text.strip():
        return []
    candidates = [source_text.removeprefix("[TABLE_ROW] "), *source_text.split(" || ")]
    try:
        with pymupdf.open(path) as document:
            page = document.load_page(max(0, min(page_number - 1, document.page_count - 1)))
            for candidate in sorted({" ".join(item.split()) for item in candidates if len(item.strip()) >= 3}, key=len):
                matches = page.search_for(candidate)
                if matches:
                    rect = matches[0]
                    return [round(rect.x0, 2), round(rect.y0, 2), round(rect.x1, 2), round(rect.y1, 2)]
    except Exception:
        return []
    return []


def _join_names(names: Iterable[str]) -> str:
    output: list[str] = []
    seen: set[str] = set()
    for value in names:
        name = re.sub(r"\s+", " ", str(value)).strip(" \t:：,;，；-|_")
        key = re.sub(r"[^0-9a-z\u3400-\u9fff]+", "", name.casefold())
        if len(key) >= 5 and key not in seen:
            seen.add(key)
            output.append(name)
    return " / ".join(output)


def _resource(kind: str, base_url: str, model: str) -> str:
    host = urlsplit(base_url).netloc or base_url
    detail = model.strip() or "未指定模型"
    return f"{kind} · {detail} · {host}"


def _merge_ocr_text(pages: list[PageText], markdown: str, sparse_pages: list[int]) -> list[PageText]:
    """Preserve good text-layer pages and attach OCR output to pages that were empty."""
    if not sparse_pages:
        return pages
    target = sparse_pages[0]
    output: list[PageText] = []
    for page in pages:
        if page.page == target:
            merged = "\n\n".join(part for part in (page.text.strip(), "[OCR]\n" + markdown.strip()) if part.strip())
            output.append(PageText(page.page, merged))
        else:
            output.append(page)
    return output


def _expert_document(row: dict[str, object]) -> ExpertDocument:
    try:
        payload = json.loads(str(row.get("page_text") or "[]"))
    except json.JSONDecodeError:
        payload = []
    pages = [PageText(int(item.get("page", 1)), str(item.get("text", ""))) for item in payload if isinstance(item, dict)]
    if not pages:
        pages = [PageText(1, "")]
    return ExpertDocument(str(row.get("original_name") or "未命名文件"), Path(str(row.get("stored_path") or "")),
                          pages, str(row.get("ocr_status") or "not_needed"))


def _dedupe_findings(items: list[Finding]) -> list[Finding]:
    seen: set[tuple[str, str, int, str]] = set()
    output: list[Finding] = []
    for item in items:
        key = (item.category, item.source_file, item.source_page,
               re.sub(r"\s+", "", item.source_text or item.actual).casefold())
        if key not in seen:
            seen.add(key)
            output.append(item)
    return output

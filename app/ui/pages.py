from __future__ import annotations

import html
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from app.exporters import export_batch, export_batch_pdf
from app.integrations import EmbeddingClient, LLMClient, MinerUClient
from app.integrations.settings import mask_secret
from app.ui.context import get_context
from app.ui.document_preview import document_pages, read_original_file, render_pdf_evidence, render_pdf_page


def start_review_page() -> None:
    ctx = get_context()
    settings = ctx.config_store.get()
    transport = '<span class="qaqc-badge remote">外部服务已启用</span>' if settings.uses_remote else '<span class="qaqc-badge">本机 / 局域网服务</span>'
    st.html(f"""<section class="qaqc-hero"><h1>上传质量文件，<br>把问题和依据一次找全</h1>
      <p>选择审核模板和依据，上传供应商文件。程序先做逐页 OCR 与表格规则检查，再由 AI 基于原页证据复核遗漏。</p>
      <div class="qaqc-privacy">所有文件默认保留在本机　{transport}</div></section>""")
    templates = ctx.db.query("SELECT * FROM audit_templates WHERE enabled=1 ORDER BY is_default DESC,name")
    template_options = {row["name"]: row["id"] for row in templates}
    chosen_name = st.selectbox("审核模板", list(template_options), key="review_template")
    template_id = template_options.get(chosen_name)
    basis = ctx.db.query("SELECT id,original_name,document_kind FROM documents WHERE library_code='basis' AND parse_status='completed' ORDER BY created_at DESC")
    built_in_label = f"{chosen_name} · 内置模板规则（自动启用）"
    basis_options = {built_in_label: None, **{
        f"{row['original_name']} · {_kind_label(row['document_kind'])}": row["id"] for row in basis
    }}
    defaults = [row["document_id"] for row in ctx.db.query("SELECT document_id FROM template_basis WHERE template_id=?", (template_id,))]
    default_labels = [built_in_label, *[label for label, doc_id in basis_options.items() if doc_id in defaults]]
    selected_labels = st.multiselect("审核依据", list(basis_options), default=default_labels,
                                     placeholder="可多选；模板绑定的依据会自动带出")
    selected_ids = [basis_options[label] for label in selected_labels if basis_options[label]]
    if not basis:
        st.caption("当前使用审核模板内置规则；审核依据库暂无额外标准，可稍后导入采购要求、图纸或标准。")
    left, right = st.columns(2, gap="large")
    with left:
        st.html('<div class="qaqc-file-label">供应商质量文件　<span style="color:#b42318">必填</span></div>')
        supplier_files = st.file_uploader("供应商质量文件", type=["pdf", "docx", "xlsx", "jpg", "jpeg", "png"],
                                          accept_multiple_files=True, label_visibility="collapsed", key="upload_supplier")
        st.caption("支持 PDF、DOCX、XLSX、JPG、PNG；可一次上传多份证明书和报告。")
    with right:
        st.html('<div class="qaqc-file-label">本次补充审核依据　<span style="color:#667085">可选</span></div>')
        supplemental = st.file_uploader("本次补充依据", type=["pdf", "docx", "xlsx", "txt"],
                                        accept_multiple_files=True, label_visibility="collapsed", key="upload_supplemental")
        st.caption("临时采购要求、图纸或协议优先级最高，并自动存入审核依据库。")
    st.space("small")
    if settings.uses_remote:
        st.warning("当前配置包含公网服务。本次文件内容可能发送到已配置的外部 LLM、Embedding 或 OCR 地址。", icon=":material/cloud_upload:")
    if st.button("开始审核", type="primary", width="stretch", icon=":material/play_arrow:", key="start_review"):
        if not supplier_files:
            st.error("请至少上传一份供应商质量文件。", icon=":material/error:")
        else:
            try:
                batch_id = ctx.service.create_review(template_id, selected_ids, supplier_files, supplemental or [])
                st.session_state["active_batch"] = batch_id
                st.toast("审核任务已创建", icon=":material/check_circle:")
                st.session_state["record_batch"] = batch_id
            except Exception as exc:
                st.error(f"创建审核任务失败：{exc}")
    batch_id = st.session_state.get("active_batch")
    if batch_id:
        batch_status_card(str(batch_id))


@st.fragment(run_every="2s")
def batch_status_card(batch_id: str) -> None:
    ctx = get_context()
    batch = ctx.db.one("SELECT * FROM review_batches WHERE id=?", (batch_id,))
    if not batch:
        return
    with st.container(border=True):
        top = st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center")
        with top:
            st.subheader(batch["name"])
            st.badge(_status_label(batch["status"]), color=_status_color(batch["status"]))
        st.caption(f"供应商：{batch.get('supplier_name') or ('正在根据 OCR/LLM 识别' if batch['status'] in {'queued', 'running'} else '未识别')}")
        st.progress(int(batch["progress"]), text=f"{batch['stage']} {('· ' + batch['current_file']) if batch['current_file'] else ''}")
        age = _heartbeat_age(batch.get("heartbeat_at") or batch.get("updated_at"))
        activity = batch.get("activity") or "等待 worker 更新当前操作"
        resource = batch.get("resource") or "SQLite 本地任务队列"
        st.caption(f"正在审核：{activity}　·　调用资源：{resource}　·　最近活动：{_age_label(age)}")
        if batch["status"] in {"running", "cancel_requested"} and age >= 90:
            st.warning(
                f"此阶段已 {age} 秒没有新进度，可能仍在等待 {resource} 响应；如果时间继续增长，可检查对应服务状态。",
                icon=":material/hourglass_top:",
            )
        if batch["status"] in {"queued", "running", "cancel_requested"}:
            if st.button("停止审核", icon=":material/stop_circle:", key=f"cancel_{batch_id}", type="secondary"):
                if ctx.db.request_cancel(batch_id):
                    st.toast("停止请求已提交", icon=":material/stop_circle:")
                    st.rerun(scope="fragment")
        if batch["status"] == "failed":
            st.error(batch["error"] or "处理失败")
        elif batch["status"] == "cancelled":
            st.warning("审核已停止，未发布不完整结论。已解析文件仍然保留。", icon=":material/pause_circle:")
            if st.button("重新审核", icon=":material/replay:", key=f"retry_{batch_id}"):
                new_batch = ctx.service.retry_review(batch_id)
                st.session_state["active_batch"] = new_batch
                st.session_state["record_batch"] = new_batch
                st.rerun(scope="fragment")
        elif batch["status"] == "completed":
            summary = json.loads(batch["summary"] or "{}")
            st.success(f"审核完成：共发现 {summary.get('total', 0)} 个问题或待确认项。", icon=":material/task_alt:")
            if st.button("查看完整结果", icon=":material/visibility:", key=f"open_result_{batch_id}"):
                st.session_state["record_batch"] = batch_id
                review_page = st.session_state.get("_review_page")
                if review_page:
                    st.switch_page(review_page, query_params={"batch": batch_id})


def review_records_page() -> None:
    ctx = get_context()
    _section_header("审核中心", "从批次总览进入问题、原文件、处理过程和导出；供应商档案已经整合到每个批次中。")
    all_batches = ctx.db.query(
        """SELECT b.*,t.name template_name,
           (SELECT COUNT(*) FROM findings f WHERE f.batch_id=b.id) finding_count,
           (SELECT COUNT(*) FROM findings f WHERE f.batch_id=b.id AND f.severity IN ('Critical','Major')) major_count,
           (SELECT COUNT(*) FROM findings f WHERE f.batch_id=b.id AND f.severity='Review' AND f.status='AI发现') review_count
           FROM review_batches b LEFT JOIN audit_templates t ON t.id=b.template_id ORDER BY b.created_at DESC"""
    )
    active_rows = [row for row in all_batches if not row.get("deleted_at")]
    feedback = ctx.db.one(
        """SELECT COUNT(*) total,SUM(CASE WHEN action='人工确认' THEN 1 ELSE 0 END) confirmed,
           SUM(CASE WHEN action='人工驳回' THEN 1 ELSE 0 END) rejected FROM review_feedback"""
    ) or {"total": 0, "confirmed": 0, "rejected": 0}
    durations = []
    for row in active_rows:
        if row["status"] == "completed" and row.get("started_at") and row.get("completed_at"):
            try:
                durations.append((datetime.fromisoformat(row["completed_at"]) - datetime.fromisoformat(row["started_at"])).total_seconds())
            except ValueError:
                pass
    total_feedback = int(feedback.get("total") or 0)
    with st.container(horizontal=True):
        st.metric("审核批次", len(active_rows), border=True)
        st.metric("处理中", sum(row["status"] in {"queued", "running", "cancel_requested"} for row in active_rows), border=True)
        st.metric("待人工复核", sum(int(row["review_count"] or 0) for row in active_rows), border=True)
        st.metric("严重/主要问题", sum(int(row["major_count"] or 0) for row in active_rows), border=True)
        st.metric("平均耗时", f"{sum(durations) / len(durations) / 60:.1f} 分" if durations else "-", border=True)
        st.metric("确认 / 驳回", f"{int(feedback.get('confirmed') or 0)} / {int(feedback.get('rejected') or 0)}",
                  help=f"累计人工反馈 {total_feedback} 条", border=True)
    view_mode = st.segmented_control("记录范围", ["当前审核", "回收站"], default="当前审核", key="review_scope")
    batches = active_rows if view_mode == "当前审核" else [row for row in all_batches if row.get("deleted_at")]
    if not batches:
        st.info("暂无记录。" if view_mode == "回收站" else "暂无审核记录。请先在“开始审核”上传文件。")
        return
    with st.container(border=True):
        filters = st.container(horizontal=True, vertical_alignment="bottom")
        with filters:
            supplier_filter = st.text_input("供应商/批次搜索", placeholder="输入供应商、批次或模板", key="review_search")
            status_options = sorted({_status_label(row["status"]) for row in batches})
            status_filter = st.multiselect("状态", status_options, placeholder="全部状态", key="review_status_filter")
            severity_filter = st.selectbox("问题范围", ["全部", "有严重/主要问题", "有待复核", "无问题"], key="review_issue_filter")
        query = supplier_filter.casefold().strip()
        visible_batches = [row for row in batches if (not query or query in f"{row['name']} {row.get('supplier_name','')} {row.get('template_name','')}".casefold())]
        if status_filter:
            visible_batches = [row for row in visible_batches if _status_label(row["status"]) in status_filter]
        if severity_filter == "有严重/主要问题": visible_batches = [row for row in visible_batches if int(row["major_count"] or 0) > 0]
        elif severity_filter == "有待复核": visible_batches = [row for row in visible_batches if int(row["review_count"] or 0) > 0]
        elif severity_filter == "无问题": visible_batches = [row for row in visible_batches if int(row["finding_count"] or 0) == 0]
        if not visible_batches:
            st.info("没有符合筛选条件的审核批次。")
            return
        table = pd.DataFrame([{
            "批次": row["name"], "供应商": row.get("supplier_name") or "未识别", "状态": _status_label(row["status"]),
            "模板": row.get("template_name") or "-", "问题": int(row["finding_count"] or 0),
            "严重/主要": int(row["major_count"] or 0), "待复核": int(row["review_count"] or 0), "创建时间": row["created_at"],
        } for row in visible_batches])
        event = st.dataframe(table, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
                             key=f"review_batch_table_{view_mode}",
                             column_config={"问题": st.column_config.NumberColumn(format="%d"),
                                            "严重/主要": st.column_config.NumberColumn(format="%d"),
                                            "待复核": st.column_config.NumberColumn(format="%d")})
    preferred = str(st.query_params.get("batch") or st.session_state.get("record_batch") or "")
    if event.selection.rows:
        batch_id = str(visible_batches[event.selection.rows[0]]["id"])
    elif any(str(row["id"]) == preferred for row in visible_batches):
        batch_id = preferred
    else:
        batch_id = str(visible_batches[0]["id"])
    st.session_state["record_batch"] = batch_id
    batch = next(row for row in batches if row["id"] == batch_id)
    with st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center"):
        st.subheader(f"{batch['name']} · {batch.get('supplier_name') or '供应商未识别'}")
        if view_mode == "当前审核":
            if st.button("移到回收站", icon=":material/delete:", key=f"trash_{batch_id}"):
                _confirm_delete(batch_id, str(batch["name"]), permanent=False)
        else:
            with st.container(horizontal=True):
                if st.button("恢复", icon=":material/restore:", key=f"restore_{batch_id}"):
                    ctx.db.restore_batch(batch_id); st.toast("审核记录已恢复"); st.rerun()
                purge_due = _purge_due(str(batch.get("purge_after") or ""))
                if st.button("永久删除", icon=":material/delete_forever:", key=f"purge_{batch_id}", type="primary",
                             disabled=not purge_due, help=None if purge_due else "30天保留期结束后才能永久删除"):
                    _confirm_delete(batch_id, str(batch["name"]), permanent=True)
    if view_mode == "回收站":
        st.caption(f"将在 {batch.get('purge_after') or '30天后'} 到期；永久删除前仍可恢复。")
    if batch["status"] != "completed":
        batch_status_card(batch_id)
    detail_view = st.segmented_control("批次详情", ["问题", "文件", "处理过程", "导出"], default="问题", key=f"batch_view_{batch_id}")
    if detail_view == "问题": _render_findings(batch_id)
    elif detail_view == "文件": _render_batch_files(batch_id)
    elif detail_view == "处理过程": _render_job_events(batch_id)
    else: _render_exports(batch_id)


@st.dialog("确认删除审核记录")
def _confirm_delete(batch_id: str, name: str, permanent: bool) -> None:
    ctx = get_context()
    if permanent:
        st.warning(f"将永久删除 **{name}** 的问题、反馈、专属文件和向量索引，无法恢复。")
        confirmed = st.checkbox("我确认永久删除以上内容")
        if st.button("永久删除", type="primary", disabled=not confirmed, icon=":material/delete_forever:"):
            ctx.service.purge_review(batch_id); st.toast("审核记录已永久删除"); st.rerun()
    else:
        st.write(f"“{name}”将进入回收站，30天内可以恢复。运行中的审核会先安全停止。")
        if st.button("移到回收站", type="primary", icon=":material/delete:"):
            ctx.db.soft_delete_batch(batch_id); st.toast("已移到回收站"); st.rerun()


def _render_findings(batch_id: str) -> None:
    ctx = get_context()
    findings = ctx.db.query("SELECT * FROM findings WHERE batch_id=? ORDER BY CASE severity WHEN 'Critical' THEN 1 WHEN 'Major' THEN 2 WHEN 'Minor' THEN 3 WHEN 'Warning' THEN 4 ELSE 5 END,id", (batch_id,))
    summary_cols = st.columns(5)
    for column, level in zip(summary_cols, ["Critical", "Major", "Minor", "Warning", "Review"]):
        column.metric(level, sum(row["severity"] == level for row in findings), border=True)
    if not findings:
        st.success("未发现问题。请仍按企业流程完成人工抽查。")
        return
    selected_levels = st.pills("严重程度", ["Critical", "Major", "Minor", "Warning", "Review"],
                               default=["Critical", "Major", "Minor", "Warning", "Review"], selection_mode="multi")
    visible = [row for row in findings if row["severity"] in (selected_levels or [])]
    frame = pd.DataFrame(visible)
    table_event = st.dataframe(frame[["id", "rule_code", "severity", "category", "item", "source_file", "source_page", "actual", "requirement", "status"]],
                 width="stretch", hide_index=True, on_select="rerun", selection_mode="single-row", key=f"finding_table_{batch_id}",
                 column_config={"id": "ID", "severity": "等级", "category": "类别", "item": "检查项目",
                                "rule_code": "规则",
                                "source_file": "供应商文件", "source_page": "页码", "actual": "实际",
                                "requirement": "要求", "status": "状态"})
    labels = {f"#{row['id']} [{row['severity']}] {row['description']}": row for row in visible}
    if not labels:
        return
    preferred_finding = st.session_state.get(f"selected_finding_{batch_id}")
    if table_event.selection.rows:
        selected = visible[table_event.selection.rows[0]]
    elif preferred_finding and any(row["id"] == preferred_finding for row in visible):
        selected = next(row for row in visible if row["id"] == preferred_finding)
    else:
        selected = labels[st.selectbox("问题详情", list(labels), key=f"finding_select_{batch_id}")]
    st.session_state[f"selected_finding_{batch_id}"] = selected["id"]
    left, right = st.columns(2, gap="large")
    with left:
        source = html.escape(selected["source_text"] or selected["actual"] or "未提供")
        st.html(f'<div class="qaqc-evidence"><strong>供应商文件证据</strong><div class="meta">{html.escape(selected["source_file"])} · 第 {selected["source_page"]} 页</div><pre>{source}</pre></div>')
    with right:
        requirement = html.escape(selected["requirement"] or "缺少明确审核依据")
        st.html(f'<div class="qaqc-evidence"><strong>审核依据</strong><div class="meta">{html.escape(selected["standard_file"])} · 第 {selected["standard_page"]} 页 · 条款 {html.escape(selected["standard_clause"] or "-")}</div><pre>{requirement}</pre></div>')
    with st.container(border=True):
        st.caption(f"规则：{selected.get('rule_code') or '历史规则'} · 版本：v{selected.get('rule_version') or 1} · 文档类型：{selected.get('document_type') or '未记录'} · 判定置信度：{float(selected.get('decision_confidence') or selected['confidence']):.0%}")
        st.markdown(f"**判断逻辑**　{selected['logic'] or '待人工确认'}")
        st.markdown(f"**问题说明**　{selected['description']}")
        st.markdown(f"**整改建议**　{selected['suggestion']}")
        with st.container(horizontal=True):
            for label, status, icon in [("确认问题", "人工确认", ":material/check:"), ("驳回", "人工驳回", ":material/close:"),
                                        ("已整改", "已整改", ":material/build:"), ("关闭", "已关闭", ":material/done_all:")]:
                if st.button(label, icon=icon, key=f"finding_{selected['id']}_{status}"):
                    current_settings = ctx.config_store.get()
                    fingerprint = f"{current_settings.embedding_fingerprint}|{current_settings.llm_base_url}|{current_settings.llm_model}"
                    ctx.db.update_finding_status(selected["id"], status, service_fingerprint=fingerprint); st.rerun()
    if selected.get("rule_code"):
        similar = ctx.db.query(
            """SELECT rf.action,rf.note,rf.correction,rf.created_at,b.name batch_name
               FROM review_feedback rf JOIN findings f ON f.id=rf.finding_id
               JOIN review_batches b ON b.id=rf.batch_id
               WHERE rf.rule_code=? AND rf.finding_id<>? ORDER BY rf.id DESC LIMIT 3""",
            (selected["rule_code"], selected["id"]),
        )
        if similar:
            with st.expander("历史人工复核案例", icon=":material/history:"):
                st.caption("仅作审核提示，历史案例不能替代本批次的正式审核依据。")
                st.dataframe(pd.DataFrame(similar), hide_index=True, width="stretch",
                             column_config={"action": "人工处理", "note": "原因", "correction": "修正内容",
                                            "created_at": "时间", "batch_name": "历史批次"})
    evidence_rows = ctx.db.query(
        """SELECT fe.*,d.stored_path,d.original_name FROM finding_evidence fe
           LEFT JOIN documents d ON d.id=fe.document_id WHERE fe.finding_id=? ORDER BY fe.id""", (selected["id"],)
    )
    if not evidence_rows:
        fallback = ctx.db.one(
            """SELECT d.id document_id,d.stored_path,d.original_name FROM batch_documents bd JOIN documents d ON d.id=bd.document_id
               WHERE bd.batch_id=? AND bd.role='supplier' AND d.original_name=? ORDER BY d.created_at LIMIT 1""",
            (batch_id, selected["source_file"]),
        )
        if fallback:
            evidence_rows = [{**fallback, "page": selected["source_page"], "source_text": selected["source_text"], "evidence_type": "source"}]
    with st.container(border=True):
        st.markdown("**原报告证据组合**")
        if not evidence_rows:
            st.info("未找到与该问题关联的原报告文件。")
        else:
            cols = st.columns(min(2, len(evidence_rows)))
            for index, evidence in enumerate(evidence_rows):
                with cols[index % len(cols)]:
                    path = Path(str(evidence.get("stored_path") or ""))
                    page = int(evidence.get("page") or 1)
                    kind = str(evidence.get("evidence_type") or "source")
                    if kind == "absence":
                        st.info(f"缺失证明：{evidence.get('source_text') or '系统已扫描适用页面但未命中该字段'}", icon=":material/search_off:")
                        if path.is_file() and path.suffix.casefold() == ".pdf":
                            st.image(render_pdf_page(str(path), path.stat().st_mtime_ns, page), caption=f"代表页：{evidence.get('original_name')} · 第 {page} 页")
                    elif path.is_file() and path.suffix.casefold() == ".pdf":
                        try:
                            bbox_values = json.loads(str(evidence.get("bbox") or "[]"))
                            bbox = tuple(float(value) for value in bbox_values) if len(bbox_values) == 4 else None
                        except (TypeError, ValueError, json.JSONDecodeError):
                            bbox = None
                        screenshot, matched = render_pdf_evidence(str(path), path.stat().st_mtime_ns, page,
                            str(evidence.get("source_text") or ""), str(selected["actual"] or ""), str(selected["item"] or ""), bbox)
                        st.image(screenshot, caption=(f"{evidence.get('original_name')} · 第 {page} 页 · 红框为定位证据" if matched
                                                     else f"{evidence.get('original_name')} · 第 {page} 页 · 未精确定位，不绘制红框"))
                    elif path.is_file() and path.suffix.casefold() in {".jpg", ".jpeg", ".png"}:
                        st.image(str(path), caption=str(evidence.get("original_name") or "原始报告"))
                    else:
                        st.caption(f"{evidence.get('original_name') or selected['source_file']} · 第 {page} 页：{evidence.get('source_text') or '-'}")
                    if st.button("在文件中查看", icon=":material/find_in_page:", key=f"open_evidence_{selected['id']}_{index}"):
                        st.session_state[f"batch_file_{batch_id}"] = evidence.get("document_id")
                        st.session_state[f"batch_file_page_{batch_id}"] = page
                        st.session_state[f"batch_view_{batch_id}"] = "文件"
                        st.rerun()


@st.cache_data(max_entries=12, show_spinner="正在生成带原页证据的审核报告…")
def _cached_pdf_export(batch_id: str, export_signature: str, *, _db: object) -> bytes:
    del export_signature
    return export_batch_pdf(_db, batch_id)  # type: ignore[arg-type]


def _render_batch_files(batch_id: str) -> None:
    ctx = get_context()
    rows = ctx.db.query(
        """SELECT d.*,bd.role FROM batch_documents bd JOIN documents d ON d.id=bd.document_id
           WHERE bd.batch_id=? ORDER BY CASE bd.role WHEN 'supplier' THEN 1 ELSE 2 END,d.created_at""", (batch_id,)
    )
    if not rows:
        st.info("该批次没有关联文件。")
        return
    lookup = {str(row["id"]): row for row in rows}
    preferred = str(st.session_state.get(f"batch_file_{batch_id}") or rows[0]["id"])
    if preferred not in lookup: preferred = str(rows[0]["id"])
    file_id = st.selectbox("选择文件", list(lookup), index=list(lookup).index(preferred), key=f"batch_file_select_{batch_id}",
                           format_func=lambda value: f"{'供应商文件' if lookup[value]['role']=='supplier' else '审核依据'} · {lookup[value]['original_name']}")
    st.session_state[f"batch_file_{batch_id}"] = file_id
    row = lookup[file_id]
    path = Path(str(row["stored_path"]))
    pages = document_pages(str(row.get("page_text") or "[]"), str(row.get("raw_text") or ""))
    page_numbers = [int(item["page"]) for item in pages] or [1]
    requested_page = int(st.session_state.get(f"batch_file_page_{batch_id}") or page_numbers[0])
    if requested_page not in page_numbers: requested_page = page_numbers[0]
    page = st.selectbox("页码", page_numbers, index=page_numbers.index(requested_page), key=f"batch_page_select_{batch_id}_{file_id}",
                        format_func=lambda value: f"第 {value} 页")
    st.session_state[f"batch_file_page_{batch_id}"] = page
    related = ctx.db.query("SELECT id,rule_code,severity,item,description FROM findings WHERE batch_id=? AND source_file=? AND source_page=? ORDER BY id",
                           (batch_id, row["original_name"], page))
    if related:
        st.caption(f"本页关联 {len(related)} 条问题")
        with st.container(horizontal=True):
            for finding in related[:6]:
                if st.button(f"{finding['rule_code'] or '#'+str(finding['id'])} · {finding['item']}", key=f"page_finding_{finding['id']}"):
                    st.session_state[f"selected_finding_{batch_id}"] = finding["id"]
                    st.session_state[f"batch_view_{batch_id}"] = "问题"
                    st.rerun()
    left, right = st.columns([3, 2], gap="large")
    with left:
        if path.is_file() and path.suffix.casefold() == ".pdf":
            st.image(render_pdf_page(str(path), path.stat().st_mtime_ns, page), caption=f"{row['original_name']} · 第 {page} 页")
        elif path.is_file() and path.suffix.casefold() in {".jpg", ".jpeg", ".png"}:
            st.image(str(path), caption=str(row["original_name"]))
        else:
            st.info("该格式暂不支持原版页面渲染，请查看提取内容或下载原文件。")
    with right:
        page_text = next((str(item["text"]) for item in pages if int(item["page"]) == page), "")
        st.text_area("提取内容", page_text, height=420, disabled=True, key=f"batch_text_{batch_id}_{file_id}_{page}")
        st.caption(f"文档类型：{row.get('detected_type') or '未识别'} ({float(row.get('type_confidence') or 0):.0%}) · OCR：{row['ocr_status']} · 索引：{row['index_status']}")
        if path.is_file():
            st.download_button("下载原文件", read_original_file(str(path), path.stat().st_mtime_ns), str(row["original_name"]),
                               str(row.get("mime_type") or "application/octet-stream"), icon=":material/download:",
                               key=f"batch_download_{batch_id}_{file_id}")


def _render_job_events(batch_id: str) -> None:
    ctx = get_context()
    events = ctx.db.query("SELECT stage,activity,resource,created_at FROM job_events WHERE batch_id=? ORDER BY id", (batch_id,))
    if not events:
        st.info("该批次暂无详细处理事件。历史版本创建的批次可能只有当前状态。")
        return
    st.dataframe(pd.DataFrame(events), hide_index=True, width="stretch",
                 column_config={"stage": "阶段", "activity": "操作", "resource": "调用资源", "created_at": "时间"})


def _render_exports(batch_id: str) -> None:
    ctx = get_context()
    findings = ctx.db.query("SELECT id,status FROM findings WHERE batch_id=? ORDER BY id", (batch_id,))
    excel_data = export_batch(ctx.db, batch_id)
    signature = json.dumps([(row["id"], row["status"]) for row in findings], ensure_ascii=False)
    pdf_data = _cached_pdf_export(batch_id, signature, _db=ctx.db)
    st.caption("导出文件包含批次、供应商、模板、规则版本、问题详情和可定位的原页证据。")
    with st.container(horizontal=True):
        st.download_button("导出问题清单.xlsx", excel_data, f"供应商质量问题清单-{batch_id[:8]}.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", icon=":material/table_view:")
        st.download_button("导出审核报告.pdf", pdf_data, f"供应商质量审核报告-{batch_id[:8]}.pdf",
                           "application/pdf", icon=":material/picture_as_pdf:")


def basis_library_page() -> None:
    ctx = get_context()
    _section_header("审核依据库", "统一保存采购技术要求、图纸、企业标准和国家/行业/国际标准。")
    with st.container(border=True):
        category = st.segmented_control("依据类别", ["采购/技术要求", "图纸", "企业标准", "国家/行业/国际标准"], default="采购/技术要求")
        uploaded = st.file_uploader("导入审核依据", type=["pdf", "docx", "xlsx", "txt"], accept_multiple_files=True, key="basis_import")
        if st.button("解析并加入依据库", type="primary", disabled=not uploaded, icon=":material/library_add:"):
            kind = {"采购/技术要求": "technical", "图纸": "drawing", "企业标准": "enterprise", "国家/行业/国际标准": "standard"}[category]
            for item in uploaded or []:
                with st.status(f"正在处理 {item.name}") as status:
                    try:
                        ctx.service.import_basis(item, kind); status.update(label=f"{item.name} 已加入依据库", state="complete")
                    except Exception as exc:
                        status.update(label=f"{item.name} 处理失败：{exc}", state="error")
    rows = ctx.db.query("""SELECT d.*,(SELECT COUNT(*) FROM requirement_rules r WHERE r.document_id=d.id) rule_count
                           FROM documents d WHERE library_code='basis' ORDER BY created_at DESC""")
    if rows:
        frame = pd.DataFrame(rows)
        st.dataframe(frame[["original_name", "document_kind", "page_count", "parse_status", "index_status", "rule_count", "created_at"]],
                     width="stretch", hide_index=True,
                     column_config={"original_name": "文件名称", "document_kind": "类别", "page_count": "页数",
                                    "parse_status": "解析", "index_status": "索引", "rule_count": "规则数", "created_at": "导入时间"})
    else:
        st.info("依据库为空。导入第一份技术协议或标准后即可在审核页选择。")


def supplier_library_page() -> None:
    ctx = get_context()
    _section_header("供应商档案", "供应商文件按自动审核批次永久保留，仅作为被审证据，不会被当作标准依据。")
    rows = ctx.db.query("""SELECT d.id,d.original_name,d.supplier_name,d.stored_path,d.mime_type,d.sha256,d.page_count,d.page_text,d.raw_text,
        d.document_kind,d.parse_status,d.ocr_status,d.index_status,d.error,d.created_at,
        GROUP_CONCAT(DISTINCT b.name) batches FROM documents d LEFT JOIN batch_documents bd ON bd.document_id=d.id
        LEFT JOIN review_batches b ON b.id=bd.batch_id WHERE d.library_code='supplier' GROUP BY d.id ORDER BY d.created_at DESC""")
    if rows:
        document_ids = [str(row["id"]) for row in rows]
        selected_id = str(st.session_state.get("supplier_document_preview_id") or document_ids[0])
        if selected_id not in document_ids:
            selected_id = document_ids[0]
            st.session_state["supplier_document_preview_id"] = selected_id
        table = pd.DataFrame(rows).set_index("id")[["supplier_name", "original_name", "document_kind", "page_count",
            "parse_status", "ocr_status", "index_status", "created_at", "batches"]]
        table.insert(0, "selected", [str(index) == selected_id for index in table.index])
        edited = st.data_editor(
            table, width="stretch", hide_index=True, key="supplier_archive_table",
            disabled=[column for column in table.columns if column != "selected"],
            column_config={"selected": st.column_config.CheckboxColumn("查看", help="勾选后，下方立即显示该报告"),
                           "supplier_name": "供应商名称", "original_name": "文件名称", "document_kind": "类型",
                           "page_count": "页数", "parse_status": "解析", "ocr_status": "OCR",
                           "index_status": "索引", "created_at": "上传时间", "batches": "审核批次"},
        )
        checked = [str(index) for index in edited.index[edited["selected"]]]
        newly_checked = next((value for value in checked if value != selected_id), None)
        if newly_checked:
            st.session_state["supplier_document_preview_id"] = newly_checked
            st.rerun()
        st.subheader("浏览报告内容")
        row_lookup = {str(row["id"]): row for row in rows}
        selected_id = st.selectbox(
            "选择报告", document_ids, key="supplier_document_preview_id",
            format_func=lambda value: f"{row_lookup[value].get('supplier_name') or '供应商未识别'} · {row_lookup[value]['original_name']} · {row_lookup[value]['created_at']}",
        )
        selected = row_lookup[str(selected_id)]
        _supplier_document_detail(selected)
    else:
        st.info("暂无供应商档案。")


@st.fragment
def _supplier_document_detail(document: dict[str, object]) -> None:
    path = Path(str(document["stored_path"]))
    if not path.is_file():
        st.error("原文件不存在，无法预览或下载。")
        return
    pages = document_pages(str(document.get("page_text") or "[]"), str(document.get("raw_text") or ""))
    page_options = {f"第 {item['page']} 页": item for item in pages}
    with st.container(border=True):
        heading = st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center")
        with heading:
            st.markdown(f"**{document['original_name']}**")
            st.badge(f"{document['page_count']} 页", color="gray")
        st.caption(f"供应商：{document.get('supplier_name') or '未识别（重新审核时将结合 OCR 与 LLM 复核）'}")
        view = st.segmented_control(
            "浏览方式", ["提取内容", "原文件预览", "处理信息"], default="提取内容",
            key=f"preview_mode_{document['id']}",
        )
        if view == "提取内容":
            label = st.selectbox("页码", list(page_options), key=f"preview_page_{document['id']}")
            st.text_area("报告文字内容", str(page_options[label]["text"]), height=430, disabled=True)
        elif view == "原文件预览":
            suffix = path.suffix.casefold()
            if suffix in {".jpg", ".jpeg", ".png"}:
                st.image(str(path), caption=str(document["original_name"]), width="stretch")
            elif suffix == ".pdf":
                label = st.selectbox("预览页码", list(page_options), key=f"visual_page_{document['id']}")
                page_number = int(page_options[label]["page"])
                st.image(render_pdf_page(str(path), path.stat().st_mtime_ns, page_number),
                         caption=f"{document['original_name']} · 第 {page_number} 页", width="stretch")
            else:
                st.info("DOCX/XLSX 暂以“提取内容”方式浏览；可下载原文件查看完整排版。")
        else:
            st.json({"文件名": document["original_name"], "类型": document["mime_type"] or path.suffix,
                     "文件大小": _file_size(path.stat().st_size), "SHA-256": document["sha256"],
                     "解析状态": document["parse_status"], "OCR 状态": document["ocr_status"],
                     "索引状态": document["index_status"], "审核批次": document["batches"] or "-",
                     "错误": document["error"] or "-"})
        st.download_button(
            "下载原始报告", read_original_file(str(path), path.stat().st_mtime_ns), str(document["original_name"]),
            str(document["mime_type"] or "application/octet-stream"), icon=":material/download:",
            on_click="ignore", key=f"download_supplier_{document['id']}",
        )


def templates_page() -> None:
    ctx = get_context()
    _section_header("规则模板", "模板定义必检项目并可绑定默认审核依据；它是结构化配置，不是第三个文档库。")
    templates = ctx.db.query("SELECT * FROM audit_templates ORDER BY is_default DESC,name")
    names = {row["name"]: row for row in templates}
    selected_name = st.selectbox("选择模板", [*names, "新建模板"])
    selected = names.get(selected_name)
    basis = ctx.db.query("SELECT id,original_name FROM documents WHERE library_code='basis' AND parse_status='completed' ORDER BY created_at DESC")
    basis_labels = {row["original_name"]: row["id"] for row in basis}
    attached = [] if not selected else [row["document_id"] for row in ctx.db.query("SELECT document_id FROM template_basis WHERE template_id=?", (selected["id"],))]
    with st.form("template_editor"):
        name = st.text_input("模板名称", value=selected["name"] if selected else "")
        description = st.text_area("说明", value=selected["description"] if selected else "")
        required = st.text_area("必检项目（每行一个）", value="\n".join(json.loads(selected["required_items"] or "[]")) if selected else "")
        instructions = st.text_area(
            "专家审核说明",
            value=str(selected.get("review_instructions") or "") if selected else "",
            height=260,
            help="这里保存判断规则和提示说明，不会再把每一行误当成供应商报告中必须出现的字段。",
        )
        default_basis = st.multiselect("默认审核依据", list(basis_labels), default=[label for label, doc_id in basis_labels.items() if doc_id in attached])
        enabled = st.toggle("启用", value=bool(selected["enabled"]) if selected else True)
        is_default = st.toggle("设为默认模板", value=bool(selected["is_default"]) if selected else False)
        if st.form_submit_button("保存模板", type="primary", icon=":material/save:"):
            if not name.strip():
                st.error("模板名称不能为空。")
            else:
                items = [line.strip() for line in required.splitlines() if line.strip()]
                if selected:
                    template_id = selected["id"]
                    ctx.db.execute("UPDATE audit_templates SET name=?,description=?,required_items=?,review_instructions=?,enabled=?,is_default=? WHERE id=?",
                                   (name.strip(), description.strip(), json.dumps(items, ensure_ascii=False), instructions.strip(), int(enabled), int(is_default), template_id))
                else:
                    template_id = ctx.db.execute("""INSERT INTO audit_templates(name,description,required_document_types,required_items,review_instructions,enabled,is_default,created_at)
                        VALUES(?,?,?,?,?,?,?,datetime('now'))""", (name.strip(), description.strip(), "[]", json.dumps(items, ensure_ascii=False), instructions.strip(), int(enabled), int(is_default)))
                if is_default:
                    ctx.db.execute("UPDATE audit_templates SET is_default=0 WHERE id<>?", (template_id,))
                with ctx.db.connect() as connection:
                    connection.execute("DELETE FROM template_basis WHERE template_id=?", (template_id,))
                    connection.executemany("INSERT INTO template_basis(template_id,document_id) VALUES(?,?)",
                                           [(template_id, basis_labels[label]) for label in default_basis])
                st.success("模板已保存。"); st.rerun()
    if selected:
        st.subheader("版本化通用规则")
        rule_rows = ctx.db.query(
            """SELECT r.code,r.group_name,r.title,r.applies_to,r.severity,r.current_version,
               COALESCE(tr.enabled,0) enabled FROM audit_rules r
               LEFT JOIN template_rule_versions tr ON tr.rule_code=r.code AND tr.template_id=?
               ORDER BY r.group_name,r.code""", (selected["id"],)
        )
        if rule_rows:
            metrics = ctx.db.query(
                """SELECT r.code,COUNT(DISTINCT f.id) triggers,
                   COUNT(DISTINCT CASE WHEN rf.action='人工确认' THEN rf.id END) confirmed,
                   COUNT(DISTINCT CASE WHEN rf.action='人工驳回' THEN rf.id END) rejected
                   FROM audit_rules r LEFT JOIN findings f ON f.rule_code=r.code
                   LEFT JOIN review_feedback rf ON rf.finding_id=f.id GROUP BY r.code"""
            )
            metric_lookup = {row["code"]: row for row in metrics}
            for row in rule_rows:
                metric = metric_lookup.get(row["code"], {})
                row["triggers"] = int(metric.get("triggers") or 0)
                row["confirmed"] = int(metric.get("confirmed") or 0)
                row["rejected"] = int(metric.get("rejected") or 0)
                reviewed = row["confirmed"] + row["rejected"]
                row["false_positive_rate"] = row["rejected"] / reviewed if reviewed else 0.0
            rules_frame = pd.DataFrame(rule_rows).set_index("code")
            rules_frame["enabled"] = rules_frame["enabled"].astype(bool)
            with st.form(f"template_rules_{selected['id']}"):
                edited = st.data_editor(
                    rules_frame[["enabled", "group_name", "title", "severity", "current_version", "triggers", "confirmed", "rejected", "false_positive_rate", "applies_to"]],
                    width="stretch", disabled=["group_name", "title", "severity", "current_version", "triggers", "confirmed", "rejected", "false_positive_rate", "applies_to"],
                    column_config={"enabled": "启用", "group_name": "分组", "title": "规则说明", "severity": "等级",
                                   "current_version": "版本", "triggers": "触发", "confirmed": "确认", "rejected": "驳回",
                                   "false_positive_rate": st.column_config.NumberColumn("驳回率", format="percent"),
                                   "applies_to": "适用文档类型"}, key=f"rule_editor_{selected['id']}"
                )
                change_reason = st.text_input("变更理由", placeholder="例如：减少 MTR 字段缺失误报")
                save_rules = st.form_submit_button("保存为新规则版本", type="primary", icon=":material/save:")
            if save_rules:
                changed_codes = [code for code in edited.index if bool(edited.loc[code, "enabled"]) != bool(rules_frame.loc[code, "enabled"])]
                if not changed_codes:
                    st.info("规则启用状态没有变化。")
                elif not change_reason.strip():
                    st.error("版本变更必须填写修改理由。")
                else:
                    with ctx.db.connect() as connection:
                        for code in changed_codes:
                            version = int(rules_frame.loc[code, "current_version"]) + 1
                            connection.execute("INSERT INTO audit_rule_versions(rule_code,version,parameters,change_reason,created_at) VALUES(?,?,?,?,datetime('now'))",
                                               (code, version, json.dumps({"enabled": bool(edited.loc[code, "enabled"])}), change_reason.strip()))
                            connection.execute("UPDATE audit_rules SET current_version=? WHERE code=?", (version, code))
                            connection.execute("""INSERT INTO template_rule_versions(template_id,rule_code,rule_version,enabled) VALUES(?,?,?,?)
                                ON CONFLICT(template_id,rule_code) DO UPDATE SET rule_version=excluded.rule_version,enabled=excluded.enabled""",
                                               (selected["id"], code, version, int(bool(edited.loc[code, "enabled"]))))
                    st.success(f"已保存 {len(changed_codes)} 条规则的新版本。")
                    st.rerun()
            suggestions = [row for row in rule_rows if row["confirmed"] + row["rejected"] >= 3 and row["false_positive_rate"] >= .4]
            if suggestions:
                st.warning("规则优化建议：以下规则的人工驳回率较高，建议检查适用文档类型、字段别名或阈值；系统不会自动修改。")
                st.dataframe(pd.DataFrame(suggestions)[["code", "title", "triggers", "confirmed", "rejected", "false_positive_rate"]],
                             hide_index=True, width="stretch",
                             column_config={"code": "规则", "title": "说明", "triggers": "触发", "confirmed": "确认",
                                            "rejected": "驳回", "false_positive_rate": st.column_config.NumberColumn("驳回率", format="percent")})


def settings_page() -> None:
    ctx = get_context()
    _section_header("系统设置", "分别配置 LLM、向量和 OCR 服务。保存后下一次任务立即生效。")
    current = ctx.config_store.get()
    presets = ctx.config_store.presets()
    if presets:
        preset_options = {f"{_service_label(row['category'])} · {row['name']}": row["id"] for row in presets}
        preset_col, apply_col = st.columns([4, 1], vertical_alignment="bottom")
        selected_preset = preset_col.selectbox("连接预设", list(preset_options), key="service_preset")
        if apply_col.button("应用预设", icon=":material/settings_backup_restore:", width="stretch"):
            try:
                ctx.config_store.apply_preset(int(preset_options[selected_preset]))
                st.success("预设已应用。")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
    with st.form("service_settings"):
        allow_remote = st.toggle("允许公网服务", value=current.allow_remote,
                                 help="启用后，文档内容可能发送到配置的公网地址。默认关闭。")
        st.subheader("LLM 大模型")
        llm_url = st.text_input("LLM Base URL", current.llm_base_url)
        llm_key = st.text_input("LLM API Key", mask_secret(current.llm_api_key), type="password")
        llm_model = st.text_input("LLM 模型", current.llm_model, placeholder="留空则使用服务端默认模型，例如 Qwen3.8-27B-4bit")
        llm_temp = st.number_input("温度", 0.0, 2.0, current.llm_temperature, 0.1)
        st.subheader("向量模型")
        emb_url = st.text_input("Embedding Base URL", current.embedding_base_url)
        emb_key = st.text_input("Embedding API Key", mask_secret(current.embedding_api_key), type="password")
        emb_model = st.text_input("Embedding 模型", current.embedding_model)
        emb_dim = st.number_input("向量维度", 1, 100000, current.embedding_dimensions)
        st.subheader("MinerU OCR")
        ocr_url = st.text_input("OCR Base URL", current.ocr_base_url)
        ocr_key = st.text_input("OCR API Key", mask_secret(current.ocr_api_key), type="password")
        ocr_backend = st.text_input("Backend", current.ocr_backend)
        ocr_lang = st.text_input("语言", current.ocr_lang)
        st.subheader("检索参数")
        c1, c2, c3 = st.columns(3)
        chunk_size = c1.number_input("分块大小", 200, 8000, current.chunk_size)
        overlap = c2.number_input("分块重叠", 0, 2000, current.chunk_overlap)
        top_k = c3.number_input("Top K", 1, 50, current.top_k)
        submitted = st.form_submit_button("保存配置", type="primary", icon=":material/save:")
    if submitted:
        try:
            updated = ctx.config_store.save({"allow_remote": allow_remote, "llm_base_url": llm_url, "llm_api_key": llm_key,
                "llm_model": llm_model, "llm_temperature": llm_temp, "embedding_base_url": emb_url,
                "embedding_api_key": emb_key, "embedding_model": emb_model, "embedding_dimensions": emb_dim,
                "ocr_base_url": ocr_url, "ocr_api_key": ocr_key, "ocr_backend": ocr_backend, "ocr_lang": ocr_lang,
                "chunk_size": chunk_size, "chunk_overlap": overlap, "top_k": top_k})
            st.success("配置已保存。")
        except Exception as exc:
            st.error(str(exc))
    st.subheader("连接测试")
    st.caption("连接测试使用已保存的配置。修改地址、模型或 API Key 后，请先点击上方“保存配置”。Ollama 11434 会自动补 `/v1`；其他服务请按其真实接口路径填写。")
    if st.button("一键测试全部服务", type="primary", icon=":material/network_check:", width="stretch"):
        _test_all_services(ctx.config_store.get())
    with st.container(horizontal=True):
        if st.button("测试 LLM", icon=":material/smart_toy:"):
            _show_test(lambda: LLMClient(ctx.config_store.get()).test())
        if st.button("测试向量", icon=":material/hub:"):
            _show_test(lambda: EmbeddingClient(ctx.config_store.get()).test())
        if st.button("测试 OCR", icon=":material/document_scanner:"):
            _show_test(lambda: MinerUClient(ctx.config_store.get()).test())
    st.caption("API Key 加密保存在 data/secrets，页面和日志不会显示明文。")
    st.caption("提示：思考型模型的连接测试只确认已经产生推理响应；正式审核会继续等待思考完成并读取最终 JSON。")
    with st.expander("保存当前连接为预设"):
        preset_category = st.segmented_control(
            "服务类型", ["LLM", "Embedding", "OCR"], default="LLM", key="preset_category"
        )
        preset_name = st.text_input("预设名称", placeholder="例如：本机 Ollama / 公司内网 MinerU")
        if st.button("保存预设", icon=":material/bookmark_add:"):
            try:
                ctx.config_store.save_preset({"LLM": "llm", "Embedding": "embedding", "OCR": "ocr"}[preset_category], preset_name)
                st.success("连接预设已保存。")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))


def _show_test(callback) -> None:
    with st.spinner("正在执行真实连接测试…"):
        try:
            result = callback()
            if result.get("ok"):
                st.success(result["detail"])
            else:
                st.warning(result.get("detail") or "连接测试未通过")
        except Exception as exc:
            st.error(f"连接失败：{exc}")


def _test_all_services(settings) -> None:
    checks = [
        ("LLM", lambda: LLMClient(settings).test()),
        ("Embedding", lambda: EmbeddingClient(settings).test()),
        ("OCR", lambda: MinerUClient(settings).test()),
    ]
    passed = 0
    with st.status("正在并行测试 LLM、Embedding 和 OCR…", expanded=True) as status:
        st.caption("三项测试同时开始；已完成的服务会立即显示，不再等待前一项结束。")
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=len(checks), thread_name_prefix="service-test") as executor:
            futures = {executor.submit(callback): (name, time.perf_counter()) for name, callback in checks}
            for future in as_completed(futures):
                name, item_started = futures[future]
                elapsed = time.perf_counter() - item_started
                try:
                    result = future.result()
                    if result.get("ok"):
                        passed += 1
                        st.success(f"{name}（{elapsed:.1f} 秒）：{result['detail']}")
                    else:
                        st.warning(f"{name}（{elapsed:.1f} 秒）：{result.get('detail') or '测试未通过'}")
                except Exception as exc:
                    st.error(f"{name}（{elapsed:.1f} 秒）：连接失败 — {exc}")
        total = time.perf_counter() - started
        final_state = "complete" if passed == len(checks) else "error"
        status.update(label=f"服务测试完成：{passed}/{len(checks)} 项通过 · 总耗时 {total:.1f} 秒",
                      state=final_state, expanded=True)


def _section_header(title: str, subtitle: str) -> None:
    st.html(f'<h1 class="qaqc-section-title">{html.escape(title)}</h1><p class="qaqc-section-sub">{html.escape(subtitle)}</p>')


def _kind_label(kind: str) -> str:
    return {"technical": "采购/技术要求", "drawing": "图纸", "enterprise": "企业标准", "standard": "标准规范"}.get(kind, "其它")


def _status_label(status: str) -> str:
    return {"queued": "等待处理", "running": "审核中", "cancel_requested": "正在停止", "cancelled": "已停止",
            "completed": "已完成", "failed": "失败"}.get(status, status)


def _status_color(status: str) -> str:
    return {"queued": "gray", "running": "blue", "cancel_requested": "orange", "cancelled": "gray",
            "completed": "green", "failed": "red"}.get(status, "gray")


def _service_label(category: str) -> str:
    return {"llm": "LLM", "embedding": "Embedding", "ocr": "OCR"}.get(category, category)


def _heartbeat_age(value: str | None) -> int:
    if not value:
        return 0
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return max(0, int((datetime.now(timezone.utc) - timestamp).total_seconds()))
    except ValueError:
        return 0


def _purge_due(value: str) -> bool:
    if not value:
        return False
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return timestamp <= datetime.now(timezone.utc)
    except ValueError:
        return False


def _age_label(seconds: int) -> str:
    if seconds < 5:
        return "刚刚"
    if seconds < 60:
        return f"{seconds} 秒前"
    return f"{seconds // 60} 分 {seconds % 60} 秒前"


def _file_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from app.exporters import export_batch
from app.integrations import EmbeddingClient, LLMClient, MinerUClient
from app.integrations.settings import mask_secret
from app.ui.context import get_context
from app.ui.document_preview import document_pages, read_original_file, render_pdf_page


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
    basis_options = {f"{row['original_name']} · {_kind_label(row['document_kind'])}": row["id"] for row in basis}
    defaults = [row["document_id"] for row in ctx.db.query("SELECT document_id FROM template_basis WHERE template_id=?", (template_id,))]
    default_labels = [label for label, doc_id in basis_options.items() if doc_id in defaults]
    selected_labels = st.multiselect("审核依据", list(basis_options), default=default_labels,
                                     placeholder="可多选；模板绑定的依据会自动带出")
    selected_ids = [basis_options[label] for label in selected_labels]
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
        st.progress(int(batch["progress"]), text=f"{batch['stage']} {('· ' + batch['current_file']) if batch['current_file'] else ''}")
        age = _heartbeat_age(batch.get("heartbeat_at") or batch.get("updated_at"))
        activity = batch.get("activity") or "等待 worker 更新当前操作"
        resource = batch.get("resource") or "SQLite 本地任务队列"
        st.caption(f"正在审核：{activity}　·　调用资源：{resource}　·　最近活动：{_age_label(age)}")
        if batch["status"] == "running" and age >= 90:
            st.warning(
                f"此阶段已 {age} 秒没有新进度，可能仍在等待 {resource} 响应；如果时间继续增长，可检查对应服务状态。",
                icon=":material/hourglass_top:",
            )
        if batch["status"] == "failed":
            st.error(batch["error"] or "处理失败")
        elif batch["status"] == "completed":
            summary = json.loads(batch["summary"] or "{}")
            st.success(f"审核完成：共发现 {summary.get('total', 0)} 个问题或待确认项。", icon=":material/task_alt:")
            if st.button("查看完整结果", icon=":material/visibility:", key=f"open_result_{batch_id}"):
                st.session_state["record_batch"] = batch_id
                st.toast("请打开顶部“审核记录”查看结果")


def review_records_page() -> None:
    ctx = get_context()
    _section_header("审核记录", "每次上传自动生成一个批次，无需创建项目。")
    batches = ctx.db.query("SELECT b.*,t.name template_name FROM review_batches b LEFT JOIN audit_templates t ON t.id=b.template_id ORDER BY b.created_at DESC")
    if not batches:
        st.info("暂无审核记录。请先在“开始审核”上传文件。")
        return
    options = {f"{row['name']} · {_status_label(row['status'])}": row["id"] for row in batches}
    preferred = st.session_state.get("record_batch")
    index = next((i for i, value in enumerate(options.values()) if value == preferred), 0)
    label = st.selectbox("选择审核批次", list(options), index=index)
    batch_id = options[label]
    batch = next(row for row in batches if row["id"] == batch_id)
    if batch["status"] != "completed":
        batch_status_card(batch_id)
        return
    _render_findings(batch_id)


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
    st.dataframe(frame[["id", "severity", "category", "item", "source_file", "source_page", "actual", "requirement", "status"]],
                 width="stretch", hide_index=True,
                 column_config={"id": "ID", "severity": "等级", "category": "类别", "item": "检查项目",
                                "source_file": "供应商文件", "source_page": "页码", "actual": "实际",
                                "requirement": "要求", "status": "状态"})
    labels = {f"#{row['id']} [{row['severity']}] {row['description']}": row for row in visible}
    if not labels:
        return
    selected = labels[st.selectbox("问题详情", list(labels))]
    left, right = st.columns(2, gap="large")
    with left:
        source = html.escape(selected["source_text"] or selected["actual"] or "未提供")
        st.html(f'<div class="qaqc-evidence"><strong>供应商文件证据</strong><div class="meta">{html.escape(selected["source_file"])} · 第 {selected["source_page"]} 页</div><pre>{source}</pre></div>')
    with right:
        requirement = html.escape(selected["requirement"] or "缺少明确审核依据")
        st.html(f'<div class="qaqc-evidence"><strong>审核依据</strong><div class="meta">{html.escape(selected["standard_file"])} · 第 {selected["standard_page"]} 页 · 条款 {html.escape(selected["standard_clause"] or "-")}</div><pre>{requirement}</pre></div>')
    with st.container(border=True):
        st.markdown(f"**判断逻辑**　{selected['logic'] or '待人工确认'}")
        st.markdown(f"**问题说明**　{selected['description']}")
        st.markdown(f"**整改建议**　{selected['suggestion']}")
        with st.container(horizontal=True):
            for label, status, icon in [("确认问题", "人工确认", ":material/check:"), ("驳回", "人工驳回", ":material/close:"),
                                        ("已整改", "已整改", ":material/build:"), ("关闭", "已关闭", ":material/done_all:")]:
                if st.button(label, icon=icon, key=f"finding_{selected['id']}_{status}"):
                    ctx.db.update_finding_status(selected["id"], status); st.rerun()
    data = export_batch(ctx.db, batch_id)
    st.download_button("导出问题清单.xlsx", data, "供应商质量文件问题清单.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       icon=":material/download:")


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
    rows = ctx.db.query("""SELECT d.id,d.original_name,d.stored_path,d.mime_type,d.sha256,d.page_count,d.page_text,d.raw_text,
        d.document_kind,d.parse_status,d.ocr_status,d.index_status,d.error,d.created_at,
        GROUP_CONCAT(DISTINCT b.name) batches FROM documents d LEFT JOIN batch_documents bd ON bd.document_id=d.id
        LEFT JOIN review_batches b ON b.id=bd.batch_id WHERE d.library_code='supplier' GROUP BY d.id ORDER BY d.created_at DESC""")
    if rows:
        table = pd.DataFrame(rows)[["original_name", "document_kind", "page_count", "parse_status", "ocr_status",
                                    "index_status", "created_at", "batches"]]
        st.dataframe(table, width="stretch", hide_index=True,
                     column_config={"original_name": "文件名称", "document_kind": "类型", "page_count": "页数",
                                    "parse_status": "解析", "ocr_status": "OCR", "index_status": "索引",
                                    "created_at": "上传时间", "batches": "审核批次"})
        st.subheader("浏览报告内容")
        options = {f"{row['original_name']} · {row['created_at']} · {row['id'][:8]}": row for row in rows}
        selected = options[st.selectbox("选择报告", list(options), key="supplier_document_preview")]
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
    selected_name = st.selectbox("选择模板", ["新建模板", *names])
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
    with st.status("正在依次测试 LLM、Embedding 和 OCR…", expanded=True) as status:
        for name, callback in checks:
            st.write(f"正在测试 **{name}**…")
            try:
                result = callback()
                if result.get("ok"):
                    passed += 1
                    st.success(f"{name}：{result['detail']}")
                else:
                    st.warning(f"{name}：{result.get('detail') or '测试未通过'}")
            except Exception as exc:
                st.error(f"{name}：连接失败 — {exc}")
        final_state = "complete" if passed == len(checks) else "error"
        status.update(label=f"服务测试完成：{passed}/{len(checks)} 项通过", state=final_state, expanded=True)


def _section_header(title: str, subtitle: str) -> None:
    st.html(f'<h1 class="qaqc-section-title">{html.escape(title)}</h1><p class="qaqc-section-sub">{html.escape(subtitle)}</p>')


def _kind_label(kind: str) -> str:
    return {"technical": "采购/技术要求", "drawing": "图纸", "enterprise": "企业标准", "standard": "标准规范"}.get(kind, "其它")


def _status_label(status: str) -> str:
    return {"queued": "等待处理", "running": "审核中", "completed": "已完成", "failed": "失败"}.get(status, status)


def _status_color(status: str) -> str:
    return {"queued": "gray", "running": "blue", "completed": "green", "failed": "red"}.get(status, "gray")


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

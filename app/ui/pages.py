from __future__ import annotations

import html
import inspect
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from app.exporters import export_batch, export_batch_pdf
from app.config import ROOT
from app.auditing.expert_review import parse_template_tasks
from app.auditing.bolt_template import BOLT_ENGINE, SCOPE_LABELS, SINGLE_NOTICE, SIGNATURE_NOTICE
from app.integrations import LLMClient, MinerUClient
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
    chosen_template = next((row for row in templates if row['id'] == template_id), {})
    bolt_mode = chosen_template.get('engine_binding') == BOLT_ENGINE
    scope_label = st.selectbox('审核范围', list(SCOPE_LABELS.values()), disabled=not bolt_mode,
                               help='新螺栓检验模版支持单文件预检查；完整包审核为默认。')
    audit_scope = next(key for key, value in SCOPE_LABELS.items() if value == scope_label) if bolt_mode else 'full_package'
    if audit_scope == 'single_document':
        st.warning(SINGLE_NOTICE)
    if bolt_mode:
        st.caption(SIGNATURE_NOTICE)
    review_mode_label = st.segmented_control(
        "审核方式", ["自适应审核", "思考模式深度复核"], default="自适应审核",
        key="review_mode", width="stretch",
    )
    review_mode = "deep" if review_mode_label == "思考模式深度复核" else "adaptive"
    if bolt_mode:
        st.caption("新模板按文件与 WDC 分配规则；确定性检查不调用模型，签章单独读取原页图片。深度选项仅影响文字规则。")
    elif review_mode == "adaptive":
        st.caption("每个必检项只调用一次模型，使用紧凑 JSON；无法定位的证据自动转人工复核。服务支持连续批处理时可启用双路并行。")
    else:
        st.caption("每一条必检项都启用思考模式，速度较慢，并会占用更多上下文和显存。")
        if "qwen3.5-4b" in settings.llm_model.casefold():
            st.warning("当前 qwen3.5-4b 的完整思考实测可能超过 300 秒。建议本机日常审核选择“自适应审核”。", icon=":material/timer:")
    project_models_safe_mode = "qwen3.8" in settings.llm_model.casefold()
    parallel_enabled = st.toggle(
        "启用并行 LLM 审核", value=False if project_models_safe_mode else settings.llm_concurrency > 1,
        key="review_parallel_enabled", disabled=project_models_safe_mode or bolt_mode,
        help="Project Models 的 Qwen3.8 连续批处理当前不稳定，已自动锁定安全单路。" if project_models_safe_mode
        else "只并行执行彼此独立的模板规则；文件解析、确定性规则与结果写入仍按安全顺序执行。",
    )
    if bolt_mode:
        review_concurrency = 1
        st.caption("新模板当前采用单路执行，避免文字与视觉请求竞争本机显存；并行加速需另做压力测试。")
    elif project_models_safe_mode:
        review_concurrency = 1
        st.caption("已识别 Project Models Qwen3.8：自动使用安全单路，避免生成服务出现 tuple.shape HTTP 500。")
    elif parallel_enabled:
        review_concurrency = 2
        st.caption("本次会同时执行最多 2 条独立规则。")
    else:
        review_concurrency = 1
        st.caption("本次按单路调用执行，稳定性最高。")
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
    upload_generation = int(st.session_state.get("upload_event_generation", 0))
    left, right = st.columns(2, gap="large")
    with left:
        st.html('<div class="qaqc-file-label">供应商质量文件　<span style="color:#b42318">必填</span></div>')
        supplier_files = st.file_uploader("供应商质量文件", type=["pdf", "docx", "xlsx", "jpg", "jpeg", "png"],
                                          accept_multiple_files=True, label_visibility="collapsed",
                                          key=f"upload_supplier_{upload_generation}")
        st.caption("支持 PDF、DOCX、XLSX、JPG、PNG；可一次上传多份证明书和报告。")
    with right:
        st.html('<div class="qaqc-file-label">本次补充审核依据　<span style="color:#667085">可选</span></div>')
        supplemental = st.file_uploader("本次补充依据", type=["pdf", "docx", "xlsx", "txt"],
                                        accept_multiple_files=True, label_visibility="collapsed",
                                        key=f"upload_supplemental_{upload_generation}")
        st.caption("临时采购要求、图纸或协议优先级最高，并自动存入审核依据库。")
    if settings.uses_remote:
        st.warning("当前配置包含公网服务。本次文件内容可能发送到已配置的外部 LLM 或 OCR 地址。", icon=":material/cloud_upload:")
    if st.button("开始审核", type="primary", width="stretch", icon=":material/play_arrow:", key="start_review"):
        if not supplier_files:
            st.error("请至少上传一份供应商质量文件。", icon=":material/error:")
        else:
            try:
                create_parameters = inspect.signature(ctx.service.create_review).parameters
                if "llm_concurrency" in create_parameters:
                    batch_id = ctx.service.create_review(
                        template_id, selected_ids, supplier_files, supplemental or [],
                        review_mode=review_mode, llm_concurrency=review_concurrency,
                        **({'audit_scope': audit_scope} if 'audit_scope' in create_parameters else {}),
                    )
                else:
                    # Streamlit may retain a ReviewService instance created before
                    # a hot reload. Preserve the selected concurrency globally for
                    # that legacy instance and use its older method signature.
                    ctx.config_store.save({"llm_concurrency": review_concurrency})
                    batch_id = ctx.service.create_review(
                        template_id, selected_ids, supplier_files, supplemental or [], review_mode=review_mode
                    )
                st.session_state["active_batch"] = batch_id
                st.toast("审核任务已创建", icon=":material/check_circle:")
                st.session_state["record_batch"] = batch_id
                # A completed submission starts a fresh upload event. This clears
                # the uploader widgets without changing the batch being monitored.
                st.session_state["upload_event_generation"] = upload_generation + 1
                st.rerun()
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
        concurrency = max(1, int(batch.get("llm_concurrency") or 1))
        st.caption(
            f"审核方式：{'思考模式深度复核' if batch.get('review_mode') == 'deep' else '自适应审核'}"
            f" · LLM {'并行' if concurrency > 1 else '单路'} {concurrency}"
        )
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
                    st.toast("审核已停止，正在释放后台资源", icon=":material/stop_circle:")
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
    _section_header("审核中心", "按批次浏览，在同一工作台完成证据复核、人工反馈和整改流转。")
    active_batches = ctx.db.query(
        """SELECT b.*,t.name template_name,
           (SELECT COUNT(*) FROM findings f WHERE f.batch_id=b.id) finding_count,
           (SELECT COUNT(*) FROM findings f WHERE f.batch_id=b.id AND f.severity<>'Review'
              AND f.status NOT IN ('误报驳回','不适用')) formal_count,
           (SELECT COUNT(*) FROM findings f WHERE f.batch_id=b.id AND f.severity IN ('Critical','Major')) major_count,
           (SELECT COUNT(*) FROM findings f WHERE f.batch_id=b.id AND f.severity='Review' AND f.status='AI发现') review_count
           FROM review_batches b LEFT JOIN audit_templates t ON t.id=b.template_id
           WHERE b.deleted_at='' ORDER BY b.created_at DESC"""
    )
    selected_task_batch = ""
    trash_count = int(ctx.db.one(
        "SELECT COUNT(*) count FROM review_batches WHERE deleted_at<>''"
    )["count"])
    try:
        worker_state = json.loads((ROOT / ".worker-status.json").read_text(encoding="utf-8"))
        worker_label = {"online": "在线", "restarting": "恢复中", "failed": "异常", "stopped": "停止"}.get(
            str(worker_state.get("status")), "未知"
        )
    except (OSError, ValueError, TypeError):
        worker_label = "未知"
    dashboard_filter = "all"
    totals = {
        "processing": sum(row["status"] in {"queued", "running"} for row in active_batches),
        "formal": sum(int(row["formal_count"] or 0) for row in active_batches),
        "review": sum(int(row["review_count"] or 0) for row in active_batches),
    }
    metric_columns = st.columns(4)
    metric_columns[0].metric("Worker", worker_label, border=True)
    metric_columns[1].metric("审核中", totals["processing"], border=True)
    metric_columns[2].metric("正式问题", totals["formal"], border=True)
    metric_columns[3].metric("待人工复核", totals["review"], border=True)
    top_actions = st.container(horizontal=True, horizontal_alignment="right", vertical_alignment="center")
    with top_actions:
        if st.button(f"回收站 {trash_count}", icon=":material/recycling:", key="browse_review_trash"):
            _review_trash_dialog()
    _render_learning_panel()
    batches = active_batches
    if not batches:
        st.info("暂无审核记录。请先在“开始审核”上传文件；已删除记录可通过上方小块浏览。")
        return
    st.subheader("审核批次")
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
        if dashboard_filter == "completed": visible_batches = [row for row in visible_batches if row["status"] == "completed"]
        elif dashboard_filter == "processing": visible_batches = [row for row in visible_batches if row["status"] in {"queued", "running"}]
        elif dashboard_filter == "issues": visible_batches = [row for row in visible_batches if int(row["finding_count"] or 0) > 0]
        elif dashboard_filter == "major": visible_batches = [row for row in visible_batches if int(row["major_count"] or 0) > 0]
        elif dashboard_filter == "review": visible_batches = [row for row in visible_batches if int(row["review_count"] or 0) > 0]
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
                             key="review_batch_table_active",
                             column_config={"问题": st.column_config.NumberColumn(format="%d"),
                                            "严重/主要": st.column_config.NumberColumn(format="%d"),
                                            "待复核": st.column_config.NumberColumn(format="%d")})
    preferred = str(selected_task_batch or st.query_params.get("batch") or st.session_state.get("record_batch") or "")
    if event.selection.rows:
        batch_id = str(visible_batches[event.selection.rows[0]]["id"])
    elif any(str(row["id"]) == preferred for row in visible_batches):
        batch_id = preferred
    else:
        batch_id = str(next((row for row in visible_batches if row["status"] == "completed"), visible_batches[0])["id"])
    st.session_state["record_batch"] = batch_id
    batch = next(row for row in batches if row["id"] == batch_id)
    with st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center"):
        st.subheader(f"{batch['name']} · {batch.get('supplier_name') or '供应商未识别'}")
        if st.button("移到回收站", icon=":material/delete:", key=f"trash_{batch_id}"):
            _confirm_delete(batch_id, str(batch["name"]), permanent=False)
    if batch["status"] != "completed":
        batch_status_card(batch_id)
    _render_download_card(batch_id)
    batch_summary = json.loads(batch.get('summary') or '{}')
    st.caption('审核范围：' + SCOPE_LABELS.get(batch.get('audit_scope', 'full_package'), '完整文件包审核'))
    if batch.get('audit_scope') == 'single_document':
        st.warning(SINGLE_NOTICE)
    if batch_summary.get('uncovered_wdcs'):
        st.warning('未覆盖WDC（未判合格）：' + '、'.join(batch_summary['uncovered_wdcs']))
    if batch_summary.get('vision_status'):
        st.caption('视觉状态：' + batch_summary['vision_status'] + '。' + SIGNATURE_NOTICE)
    if batch_summary.get('model_identity'):
        st.caption('本批次实际模型：' + batch_summary['model_identity'] +
                   ' · 指纹 ' + str(batch_summary.get('model_fingerprint') or '未记录'))
    detail_view = st.segmented_control("批次详情", ["人工复核", "全部规则", "文件证据", "处理记录"], default="人工复核", key=f"batch_view_{batch_id}")
    if detail_view == "人工复核": _render_findings(batch_id)
    elif detail_view == "全部规则": _render_rule_evaluations(batch_id)
    elif detail_view == "文件证据": _render_batch_files(batch_id)
    elif detail_view == "处理记录": _render_job_events(batch_id)
    else: _render_job_events(batch_id)


def _render_all_tasks_dashboard(tasks: list[dict[str, object]]) -> str:
    """Render every persistent queue task and return the selected batch id."""
    status_path = ROOT / ".worker-status.json"
    try:
        worker = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        worker = {"status": "unknown", "message": "尚未收到 worker 状态"}
    if worker.get("updated_at") and _heartbeat_age(str(worker["updated_at"])) > 10:
        worker["status"] = "stale"
        worker["message"] = "worker 状态超过10秒未更新，启动器可能已经退出"
    status_labels = {"online": "在线", "restarting": "正在恢复", "failed": "异常", "stopped": "已停止",
                     "stale": "失联", "unknown": "未知"}
    worker_label = status_labels.get(str(worker.get("status")), str(worker.get("status") or "未知"))
    queued = sum(row["job_status"] == "queued" for row in tasks)
    running = sum(row["job_status"] in {"running", "cancel_requested"} for row in tasks)
    completed = sum(row["job_status"] == "completed" for row in tasks)
    abnormal = sum(row["job_status"] in {"failed", "cancelled"} for row in tasks)
    with st.container(border=True):
        st.markdown("**全部任务 Dashboard**")
        with st.container(horizontal=True):
            st.metric("任务总数", len(tasks), border=True)
            st.metric("排队", queued, border=True)
            st.metric("运行中", running, border=True)
            st.metric("已完成", completed, border=True)
            st.metric("失败/取消", abnormal, border=True)
            st.metric("本地 worker", worker_label, border=True)
        worker_message = str(worker.get("message") or "")
        if worker.get("status") in {"failed", "restarting", "stopped", "stale", "unknown"}:
            st.warning(
                f"worker：{worker_label}。{worker_message or '排队任务暂时无法领取。'}",
                icon=":material/build_circle:",
            )
        elif queued and any(_heartbeat_age(str(row.get("updated_at") or "")) >= 15 for row in tasks if row["job_status"] == "queued"):
            st.warning(
                "存在等待超过15秒仍未被领取的任务。worker 虽显示在线，但队列可能异常，请查看任务的尝试次数和 logs/worker.log。",
                icon=":material/hourglass_disabled:",
            )
        elif queued:
            st.caption(f"worker 在线，当前还有 {queued} 个任务等待领取；状态每秒更新。")
        elif int(worker.get("restart_count") or 0) > 0:
            st.caption(f"worker 在线，本次启动已自动恢复 {worker['restart_count']} 次。{worker_message}")
        else:
            st.caption("worker 在线，SQLite 本地任务队列运行正常。")
        if not tasks:
            st.info("当前还没有任务。")
            return ""
        frame = pd.DataFrame([{
            "任务": str(row["id"])[:8], "批次": row["name"], "供应商": row.get("supplier_name") or "未识别",
            "队列状态": _status_label(str(row["job_status"])), "批次状态": _status_label(str(row["batch_status"])),
            "阶段": row["stage"], "进度": int(row["progress"] or 0), "当前操作": row["activity"],
            "调用资源": row["resource"], "尝试次数": int(row["attempts"] or 0), "最近更新": row["updated_at"],
            "失败原因": row["job_error"],
        } for row in tasks])
        event = st.dataframe(
            frame, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
            key="all_job_dashboard_table",
            column_config={
                "批次": st.column_config.TextColumn(pinned=True),
                "进度": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d%%"),
                "尝试次数": st.column_config.NumberColumn(format="%d"),
            },
        )
        st.caption("点击任一任务，下面的审核批次表和详情会同步定位到对应批次。")
    if event.selection.rows:
        return str(tasks[event.selection.rows[0]]["batch_id"])
    return ""


@st.dialog("审核回收站")
def _review_trash_dialog() -> None:
    """Keep deleted batches isolated from the audit-center layout and metrics."""
    ctx = get_context()
    rows = ctx.db.query(
        """SELECT b.id,b.name,b.supplier_name,b.status,b.deleted_at,b.purge_after,b.created_at,
                  (SELECT COUNT(*) FROM findings f WHERE f.batch_id=b.id) finding_count,
                  (SELECT COUNT(*) FROM batch_documents bd WHERE bd.batch_id=b.id) file_count
           FROM review_batches b WHERE b.deleted_at<>'' ORDER BY b.deleted_at DESC"""
    )
    if not rows:
        st.info("回收站为空。")
        return

    st.caption("这里的记录不会出现在审核中心主表、任务 Dashboard 或任何统计卡片中。")
    st.dataframe(
        pd.DataFrame([{
            "批次": row["name"], "供应商": row.get("supplier_name") or "未识别",
            "文件": int(row["file_count"] or 0), "问题": int(row["finding_count"] or 0),
            "移入时间": row["deleted_at"],
        } for row in rows]),
        hide_index=True, width="stretch",
        column_config={"文件": st.column_config.NumberColumn(format="%d"),
                       "问题": st.column_config.NumberColumn(format="%d")},
    )
    options = {
        f"{row['name']} · {row.get('supplier_name') or '未识别'} · {str(row['id'])[:8]}": row
        for row in rows
    }
    selected_label = st.selectbox("选择要处理的记录", list(options), key="trash_selected_batch")
    selected = options[selected_label]
    st.caption(
        f"移入回收站：{selected['deleted_at']}　·　原保留期截至：{selected.get('purge_after') or '未设置'}"
    )
    confirmed = st.checkbox(
        "我确认立即永久删除所选记录及其专属文件（无法恢复）",
        key=f"force_purge_confirm_{selected['id']}",
    )
    learning_choice = st.radio(
        "学习记录处理", ["保留匿名学习模式", "同时删除学习记录"],
        key=f"purge_learning_{selected['id']}",
        help="保留模式只留下规则类别和统计，不保留文件、原文或业务值。",
    )
    with st.container(horizontal=True):
        if st.button("恢复到审核中心", icon=":material/restore:", key=f"trash_restore_{selected['id']}"):
            ctx.db.restore_batch(str(selected["id"]))
            st.toast("审核记录已恢复", icon=":material/restore:")
            st.rerun()
        if st.button(
            "立即永久删除", icon=":material/delete_forever:", type="primary",
            disabled=not confirmed, key=f"trash_force_purge_{selected['id']}",
        ):
            _force_purge_review(ctx, str(selected["id"]), retain_learning=learning_choice == "保留匿名学习模式")
            st.toast("审核记录已永久删除", icon=":material/delete_forever:")
            st.rerun()


def _force_purge_review(ctx: object, batch_id: str, retain_learning: bool = True) -> None:
    """Force deletion while remaining compatible with a pre-update cached service."""
    purge = ctx.service.purge_review  # type: ignore[attr-defined]
    parameters = inspect.signature(purge).parameters
    if "force" in parameters:
        kwargs = {"force": True}
        if "retain_learning" in parameters:
            kwargs["retain_learning"] = retain_learning
        purge(batch_id, **kwargs)
        return
    # Streamlit may still hold the previous ReviewService object. Expiring the
    # retention timestamp lets that implementation perform its normal, safe
    # filesystem cleanup without requiring a process restart.
    ctx.db.update_batch(  # type: ignore[attr-defined]
        batch_id, purge_after=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    purge(batch_id)


@st.dialog("确认删除审核记录")
def _confirm_delete(batch_id: str, name: str, permanent: bool) -> None:
    ctx = get_context()
    if permanent:
        st.warning(f"将永久删除 **{name}** 的问题、反馈、专属文件和本地检索缓存，无法恢复。")
        confirmed = st.checkbox("我确认永久删除以上内容")
        if st.button("永久删除", type="primary", disabled=not confirmed, icon=":material/delete_forever:"):
            _force_purge_review(ctx, batch_id); st.toast("审核记录已永久删除"); st.rerun()
    else:
        st.write(f"“{name}”将进入回收站，30天内可以恢复。运行中的审核会先安全停止。")
        if st.button("移到回收站", type="primary", icon=":material/delete:"):
            ctx.db.soft_delete_batch(batch_id); st.toast("已移到回收站"); st.rerun()


def _finding_review_rank(row: dict) -> tuple:
    try:
        priority = json.loads(row.get('metadata') or '{}').get('review_priority') == 'high'
    except (TypeError, ValueError, AttributeError):
        priority = False
    # Learning changes order within a severity, never the severity itself.
    severity = {'Critical': 0, 'Major': 1, 'Minor': 2, 'Warning': 3, 'Review': 4}.get(row['severity'], 5)
    return severity, not priority, row['id']


def _render_findings(batch_id: str) -> None:
    ctx = get_context()
    findings = ctx.db.query("SELECT * FROM findings WHERE batch_id=? ORDER BY CASE severity WHEN 'Critical' THEN 1 WHEN 'Major' THEN 2 WHEN 'Minor' THEN 3 WHEN 'Warning' THEN 4 ELSE 5 END,id", (batch_id,))
    findings.sort(key=_finding_review_rank)
    documents = ctx.db.query(
        """SELECT d.original_name,d.detected_type FROM batch_documents bd JOIN documents d ON d.id=bd.document_id
           WHERE bd.batch_id=? AND bd.role='supplier' ORDER BY d.created_at""", (batch_id,),
    )
    with st.expander("补充漏检问题", icon=":material/add_circle:"):
        if documents:
            with st.form(f"manual_finding_{batch_id}", clear_on_submit=True):
                manual_item = st.text_input("问题名称")
                manual_description = st.text_area("问题说明")
                manual_severity = st.selectbox("严重程度", ["Major", "Minor", "Warning", "Review"])
                manual_file = st.selectbox("证据文件", [row["original_name"] for row in documents])
                manual_page = st.number_input("页码", min_value=1, step=1)
                manual_evidence = st.text_area("逐字证据")
                manual_requirement = st.text_area("判断依据")
                if st.form_submit_button("添加漏检问题", type="primary"):
                    try:
                        ctx.db.add_manual_finding(
                            batch_id, item=manual_item, description=manual_description, severity=manual_severity,
                            source_file=manual_file, source_page=int(manual_page), evidence=manual_evidence,
                            requirement=manual_requirement, service_fingerprint=_service_fingerprint(ctx),
                        )
                        st.toast("漏检问题已补充并留存审计记录；可可靠匹配规则版本的反馈才参与自动校准", icon=":material/check_circle:")
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))
        else:
            st.info("该批次没有可关联的供应商文件。")
    if not findings:
        batch = ctx.db.one("SELECT status FROM review_batches WHERE id=?", (batch_id,))
        if batch and batch["status"] == "completed":
            st.success("未发现问题。请仍按企业流程完成人工抽查。")
        else:
            st.info("该批次未完成，因此没有发布问题结果。可重新审核后查看完整结论。")
        return
    queue_scope = st.segmented_control(
        "问题队列", ["待处理", "全部问题"], default="待处理", key=f"finding_scope_{batch_id}",
    )
    visible = findings if queue_scope == "全部问题" else [
        row for row in findings if row["status"] in {"AI发现", "待人工复核", "已整改", "重新打开"}
    ]
    if not visible:
        st.success("当前没有待处理问题；可切换到“全部问题”查看历史结论。")
        return
    queue, workspace = st.columns([2, 3], gap="large")
    preferred_finding = st.session_state.get(f"selected_finding_{batch_id}")
    with queue:
        st.markdown("**待复核问题队列**")
        queue_frame = pd.DataFrame([{
            "ID": row["id"], "等级": row["severity"], "检查项": row["item"],
            "复核顺序": "优先" if not _finding_review_rank(row)[1] else "常规",
            "文件/页": f"{row['source_file']} / {row['source_page']}", "状态": row["status"],
        } for row in visible])
        table_event = st.dataframe(
            queue_frame, width="stretch", hide_index=True, on_select="rerun",
            selection_mode="single-row", key=f"finding_queue_{batch_id}", height=420,
            column_config={"ID": st.column_config.NumberColumn(width="small"),
                           "等级": st.column_config.TextColumn(width="small")},
        )
    if table_event.selection.rows:
        selected = visible[table_event.selection.rows[0]]
    elif preferred_finding and any(row["id"] == preferred_finding for row in visible):
        selected = next(row for row in visible if row["id"] == preferred_finding)
    else:
        selected = visible[0]
    st.session_state[f"selected_finding_{batch_id}"] = selected["id"]
    with workspace:
        st.markdown(f"**#{selected['id']} · {selected['item']}**")
        source = html.escape(selected["source_text"] or selected["actual"] or "未提供")
        requirement = html.escape(selected["requirement"] or "缺少明确审核依据")
        st.html(f'<div class="qaqc-evidence"><strong>供应商文件证据</strong><div class="meta">{html.escape(selected["source_file"])} · 第 {selected["source_page"]} 页</div><pre>{source}</pre></div>')
        st.html(f'<div class="qaqc-evidence"><strong>审核依据</strong><div class="meta">{html.escape(selected["standard_file"])} · 第 {selected["standard_page"]} 页 · 条款 {html.escape(selected["standard_clause"] or "-")}</div><pre>{requirement}</pre></div>')
        st.caption(f"判定置信度：{float(selected.get('decision_confidence') or selected['confidence']):.0%}")
        st.markdown(f"**判断逻辑**　{selected['logic'] or '待人工确认'}")
        st.markdown(f"**问题说明**　{selected['description']}")
        st.markdown(f"**整改建议**　{selected['suggestion']}")
        _render_finding_actions(ctx, batch_id, selected)


def _render_finding_actions(ctx: object, batch_id: str, selected: dict[str, object]) -> None:
    finding_id = int(selected["id"])
    st.markdown("**AI 结论处理**")
    action_columns = st.columns(2)
    if action_columns[0].button("确认问题", type="primary", icon=":material/check:",
                                key=f"confirm_finding_{finding_id}", width="stretch"):
        ctx.db.record_finding_feedback(  # type: ignore[attr-defined]
            finding_id, action="确认问题", new_status="人工确认",
            service_fingerprint=_service_fingerprint(ctx),
        )
        st.toast("已确认问题，学习样本已更新", icon=":material/check_circle:")
        st.rerun()
    if action_columns[1].button("撤销上次人工操作", icon=":material/undo:",
                                key=f"undo_finding_{finding_id}", width="stretch"):
        if ctx.db.undo_last_feedback(finding_id):  # type: ignore[attr-defined]
            st.toast("已撤销上次人工操作", icon=":material/undo:")
            st.rerun()
        st.warning("没有可撤销的人工操作。")
    with st.expander("误报驳回", icon=":material/close:"):
        with st.form(f"reject_finding_{finding_id}"):
            reason = st.selectbox("驳回原因", ["OCR识别错误", "证据页错误", "依据引用错误", "上下文不足",
                                                   "规则不适用", "阈值问题", "其他"])
            corrected_value = st.text_input("正确值（可选）")
            corrected_evidence = st.text_area("正确证据（可选）")
            reject_note = st.text_area("备注（不会写入 LLM 提示）")
            if st.form_submit_button("确认驳回"):
                ctx.db.record_finding_feedback(  # type: ignore[attr-defined]
                    finding_id, action="误报驳回", new_status="误报驳回", reason_code=reason,
                    correction=corrected_value, evidence=corrected_evidence, note=reject_note,
                    service_fingerprint=_service_fingerprint(ctx),
                )
                st.toast("误报已驳回并形成结构化学习样本", icon=":material/check_circle:")
                st.rerun()
    with st.expander("修正结论", icon=":material/edit_note:"):
        with st.form(f"correct_finding_{finding_id}"):
            corrected_status = st.selectbox("正确状态", ["人工确认", "待人工复核", "不适用"])
            corrected_severity = st.selectbox("正确严重程度", ["Critical", "Major", "Minor", "Warning", "Review"])
            correction = st.text_area("正确结论")
            correction_basis = st.text_area("正确依据/证据")
            correction_note = st.text_area("备注（不会写入 LLM 提示）")
            if st.form_submit_button("保存修正"):
                try:
                    ctx.db.record_finding_feedback(  # type: ignore[attr-defined]
                        finding_id, action="修正结论", new_status=corrected_status,
                        corrected_status=corrected_status, corrected_severity=corrected_severity,
                        correction=correction, evidence=correction_basis, note=correction_note,
                        service_fingerprint=_service_fingerprint(ctx),
                    )
                    st.toast("修正结论已保存", icon=":material/check_circle:")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
    st.markdown("**整改流程**")
    workflow = st.columns(3)
    for column, label, status, icon in [
        (workflow[0], "标记已整改", "已整改", ":material/build:"),
        (workflow[1], "关闭问题", "已关闭", ":material/done_all:"),
        (workflow[2], "重新打开", "重新打开", ":material/replay:"),
    ]:
        if column.button(label, icon=icon, key=f"workflow_{finding_id}_{status}", width="stretch"):
            ctx.db.record_finding_feedback(  # type: ignore[attr-defined]
                finding_id, action=label, new_status=status, learning_eligible=False,
                service_fingerprint=_service_fingerprint(ctx),
            )
            st.toast(f"已{label}", icon=":material/check_circle:")
            st.rerun()
    if st.button("在文件证据中查看此页", icon=":material/find_in_page:", key=f"open_finding_page_{finding_id}"):
        document = ctx.db.one(  # type: ignore[attr-defined]
            """SELECT d.id FROM batch_documents bd JOIN documents d ON d.id=bd.document_id
               WHERE bd.batch_id=? AND d.original_name=? LIMIT 1""",
            (batch_id, selected.get("source_file") or ""),
        )
        if document:
            st.session_state[f"batch_file_{batch_id}"] = document["id"]
            st.session_state[f"batch_file_page_{batch_id}"] = int(selected.get("source_page") or 1)
            st.session_state[f"batch_view_{batch_id}"] = "文件证据"
            st.rerun()


def _render_learning_panel() -> None:
    ctx = get_context()
    with st.expander("学习与准确率", icon=":material/model_training:"):
        summary = ctx.db.learning_summary()
        metrics = st.columns(4)
        metrics[0].metric("确认率", f"{summary['confirmation_rate']:.0%}")
        metrics[1].metric("误报率", f"{summary['rejection_rate']:.0%}")
        metrics[2].metric("漏检补充", summary["missed"])
        metrics[3].metric("有效样本", summary["total"])
        patterns = summary["patterns"]
        if patterns:
            st.dataframe(pd.DataFrame([{
                "模板": row["template"], "规则": row["rule_code"], "文档类型": row["document_type"] or "通用",
                "模型/规则指纹": row['model_fingerprint'][:12],
                "样本": row["sample_count"], "距激活": row["remaining"],
                "模式": "高误报降级" if row["downgrade_llm_issue"] else ("高确认优先" if row["prioritize_review"] else "观察中"),
                "状态": "启用" if row["enabled"] else "已暂停",
            } for row in patterns]), hide_index=True, width="stretch")
            lookup = {f"{row['template']} · {row['rule_code']} · {row['document_type'] or '通用'} · {row['sample_count']}条 · 模式{index}": row
                      for index, row in enumerate(patterns, 1)}
            chosen = lookup[st.selectbox("管理学习模式", list(lookup), key="learning_pattern")]
            controls = st.columns(2)
            if controls[0].button("暂停" if chosen["enabled"] else "恢复", key="toggle_learning_pattern", width="stretch"):
                ctx.db.set_feedback_pattern_enabled(chosen["pattern_key"], not chosen["enabled"])
                st.rerun()
            if controls[1].button("清空所选模式", key="clear_learning_pattern", width="stretch"):
                ctx.db.clear_learning_feedback(chosen["pattern_key"])
                st.toast("所选学习模式已清空")
                st.rerun()
        else:
            st.caption("尚无符合结构化要求的新反馈。旧反馈保留在审计历史中，但不参与自动校准。")
        st.download_button("导出匿名 JSONL", ctx.db.export_learning_feedback(), "qaqc-learning.jsonl",
                           "application/x-ndjson", icon=":material/download:")
        remaining_feedback = max(0, 200 - int(summary["total"]))
        remaining_reports = max(0, 50 - int(summary["report_count"]))
        st.caption(f"LoRA 数据导出尚未启用：还需 {remaining_feedback} 条高质量反馈、{remaining_reports} 份报告。")


def _service_fingerprint(ctx: object) -> str:
    settings = ctx.config_store.get()  # type: ignore[attr-defined]
    return str(settings.llm_model or "")


def _render_rule_evaluations(batch_id: str) -> None:
    """Show every template task, including passes and isolated call failures."""
    ctx = get_context()
    rows = ctx.db.query(
        """SELECT task_index,task_name,status,conclusion,source_file,source_page,evidence,
                  logic,confidence,error FROM rule_evaluations
           WHERE batch_id=? ORDER BY task_index""",
        (batch_id,),
    )
    if not rows:
        st.caption("该批次没有逐条 LLM 规则记录；可能使用旧版本创建，或模板未配置必检项目。")
        return
    with st.container(border=True):
        st.markdown("**逐条规则审核汇总**")
        counts = {status: sum(row["status"] == status for row in rows)
                  for status in ("合格", "不合格", "存疑", "不适用", "调用失败", "审核中", "pending")}
        st.caption(
            f"共 {len(rows)} 条 · 合格 {counts['合格']} · 不合格 {counts['不合格']} · "
            f"存疑 {counts['存疑']} · 不适用 {counts['不适用']} · 调用失败 {counts['调用失败']}"
        )
        display_rows = [{
            "序号": row["task_index"], "审核任务": row["task_name"], "结论": row["status"],
            "说明": row["conclusion"] or row["error"], "文件": row["source_file"],
            "页码": row["source_page"] or "", "置信度": float(row["confidence"] or 0),
        } for row in rows]
        event = st.dataframe(
            pd.DataFrame(display_rows), hide_index=True, width="stretch", on_select="rerun",
            selection_mode="single-row", key=f"rule_evaluations_{batch_id}",
            column_config={"置信度": st.column_config.ProgressColumn("置信度", min_value=0.0, max_value=1.0, format="percent")},
        )
        if event.selection.rows:
            selected = rows[event.selection.rows[0]]
            status = selected["status"]
            message = selected["error"] if status == "调用失败" else selected["conclusion"]
            st.markdown(f"**{selected['task_index']}. {selected['task_name']} — {status}**")
            if message:
                st.write(message)
            if selected["evidence"]:
                location = f"{selected['source_file'] or '检查范围'} · 第 {selected['source_page']} 页" if selected["source_page"] else "检查范围"
                st.caption(location)
                st.code(str(selected["evidence"]), language=None)
            if selected["logic"]:
                st.caption(f"判断逻辑：{selected['logic']}")


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
                    st.session_state[f"batch_view_{batch_id}"] = "人工复核"
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
        st.caption(f"OCR：{row['ocr_status']} · 依据检索：本地关键词")
        observations = ctx.db.query('SELECT * FROM visual_evidence WHERE batch_id=? AND document_id=? AND page=?',
                                    (batch_id, file_id, page))
        if observations:
            with st.expander('签章检查区域与状态', expanded=True):
                st.caption(SIGNATURE_NOTICE)
                for observation in observations:
                    detail = json.loads(observation['details'])
                    state_label = {'present': '已识别', 'absent': '明确未见', 'unknown': '需人工确认', 'unfilled': '未填字段'}.get(observation['state'], observation['state'])
                    st.write(f"{observation['kind']}：{state_label} · {detail.get('description', '')}")
                    if path.suffix.lower() == '.pdf' and path.is_file() and len(detail.get('bbox', [])) == 4:
                        from app.ui.document_preview import render_pdf_region
                        st.image(render_pdf_region(str(path), path.stat().st_mtime_ns, page, tuple(detail['bbox'])),
                                 caption='程序检查区域（不代表已验证签署身份）')
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
    _render_download_card(batch_id)


def _render_download_card(batch_id: str) -> None:
    ctx = get_context()
    findings = ctx.db.query("SELECT id,status,severity FROM findings WHERE batch_id=? ORDER BY id", (batch_id,))
    feedback = ctx.db.query(
        "SELECT id,action,corrected_status,corrected_severity,undone_at FROM review_feedback WHERE batch_id=? ORDER BY id",
        (batch_id,),
    )
    excel_data = export_batch(ctx.db, batch_id)
    signature = json.dumps({"findings": findings, "feedback": feedback}, ensure_ascii=False, sort_keys=True)
    pdf_data = _cached_pdf_export(batch_id, signature, _db=ctx.db)
    processed = sum(row["status"] != "AI发现" for row in findings)
    with st.container(border=True):
        st.markdown("**下载最新审核结果**")
        st.caption(f"当前人工状态：已处理 {processed}/{len(findings)} 条。每次反馈后会自动刷新导出缓存。")
        download_columns = st.columns(2)
        download_columns[0].download_button(
            "下载完整审核报告 PDF", pdf_data, f"供应商质量审核报告-{batch_id[:8]}.pdf",
            "application/pdf", icon=":material/picture_as_pdf:", type="primary", width="stretch",
            key=f"download_pdf_{batch_id}",
        )
        download_columns[1].download_button(
            "下载问题分析明细 Excel", excel_data, f"供应商质量问题分析-{batch_id[:8]}.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            icon=":material/table_view:", width="stretch", key=f"download_excel_{batch_id}",
        )


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
        st.dataframe(frame[["original_name", "document_kind", "page_count", "parse_status", "rule_count", "created_at"]],
                     width="stretch", hide_index=True,
                     column_config={"original_name": "文件名称", "document_kind": "类别", "page_count": "页数",
                                    "parse_status": "解析", "rule_count": "规则数", "created_at": "导入时间"})
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
            "parse_status", "ocr_status", "created_at", "batches"]]
        table.insert(0, "selected", [str(index) == selected_id for index in table.index])
        edited = st.data_editor(
            table, width="stretch", hide_index=True, key="supplier_archive_table",
            disabled=[column for column in table.columns if column != "selected"],
            column_config={"selected": st.column_config.CheckboxColumn("查看", help="勾选后，下方立即显示该报告"),
                           "supplier_name": "供应商名称", "original_name": "文件名称", "document_kind": "类型",
                           "page_count": "页数", "parse_status": "解析", "ocr_status": "OCR",
                           "created_at": "上传时间", "batches": "审核批次"},
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
                     "依据检索": "本地关键词（不使用向量模型）", "审核批次": document["batches"] or "-",
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
    if selected:
        delete_disabled = bool(selected["is_default"])
        if st.button(
            "删除当前模板", icon=":material/delete_forever:", key=f"delete_template_{selected['id']}",
            disabled=delete_disabled,
            help="默认模板不能直接删除，请先把其他模板设为默认模板。" if delete_disabled else "删除后无法恢复，但既有审核批次仍保留模板快照。",
        ):
            _confirm_delete_template(int(selected["id"]), str(selected["name"]))
    basis = ctx.db.query("SELECT id,original_name FROM documents WHERE library_code='basis' AND parse_status='completed' ORDER BY created_at DESC")
    basis_labels = {row["original_name"]: row["id"] for row in basis}
    attached = [] if not selected else [row["document_id"] for row in ctx.db.query("SELECT document_id FROM template_basis WHERE template_id=?", (selected["id"],))]
    st.info("启用行按绑定规则执行。新螺栓模板使用确定性、原页视觉和文字审核，并按文件/WDC分组；禁用行不执行。", icon=":material/account_tree:")
    token = str(selected["id"] if selected else "new")
    rows_key = f"template_task_rows_{token}"
    revision_key = f"template_task_revision_{token}"
    select_state_key = f"template_task_select_all_last_{token}"
    if rows_key not in st.session_state:
        st.session_state[rows_key] = [
            {"选择": False, "启用": bool(row["enabled"]), "必检项目": str(row["text"]),
             '规则ID': row.get('rule_id', ''), '判断标准': row.get('criterion', '')}
            for row in parse_template_tasks(selected["required_items"] if selected else "[]")
        ]
        st.session_state[revision_key] = 0
        st.session_state[select_state_key] = False

    name = st.text_input("模板名称", value=selected["name"] if selected else "", key=f"template_name_{token}")
    description = st.text_area("说明", value=selected["description"] if selected else "", key=f"template_description_{token}")
    select_all = st.checkbox("全选", value=False, key=f"template_select_all_{token}")
    if select_all != st.session_state[select_state_key]:
        st.session_state[rows_key] = [{**row, "选择": select_all} for row in st.session_state[rows_key]]
        st.session_state[select_state_key] = select_all
        st.session_state[revision_key] += 1
    task_frame = pd.DataFrame(st.session_state[rows_key], columns=["选择", "启用", "必检项目", '规则ID', '判断标准'])
    edited_tasks = st.data_editor(
        task_frame, hide_index=True, width="stretch", num_rows="dynamic", disabled=['规则ID'],
        key=f"template_task_editor_{token}_{st.session_state[revision_key]}",
        column_config={
            "选择": st.column_config.CheckboxColumn("选择", help="用于批量删除"),
            "启用": st.column_config.CheckboxColumn("启用", help="启用后执行该项绑定的检查规则"),
            "必检项目": st.column_config.TextColumn("必检项目", required=True, width="large"),
        },
    )
    current_rows = edited_tasks.fillna({"选择": False, "启用": True, "必检项目": "", "规则ID": "", "判断标准": ""}).to_dict("records")
    st.session_state[rows_key] = current_rows
    with st.container(horizontal=True):
        if st.button("增加一项", icon=":material/add:", key=f"template_add_task_{token}"):
            st.session_state[rows_key] = [*current_rows, {"选择": False, "启用": True, "必检项目": ""}]
            st.session_state[revision_key] += 1
            st.rerun()
        selected_count = sum(bool(row.get("选择")) for row in current_rows)
        if st.button("删除已选", icon=":material/delete:", key=f"template_delete_tasks_{token}", disabled=selected_count == 0):
            st.session_state[rows_key] = [row for row in current_rows if not bool(row.get("选择"))]
            st.session_state[revision_key] += 1
            st.session_state[select_state_key] = False
            st.rerun()
        st.caption(f"共 {len(current_rows)} 项，已启用 {sum(bool(row.get('启用')) for row in current_rows)} 项，已选择 {selected_count} 项。")

    instructions = st.text_area(
        "专家审核说明", value=str(selected.get("review_instructions") or "") if selected else "", height=260,
        key=f"template_instructions_{token}",
        help="所有独立任务共用的审核边界、术语和判断要求。这里不会额外产生 LLM 调用。",
    )
    default_basis = st.multiselect(
        "默认审核依据", list(basis_labels),
        default=[label for label, doc_id in basis_labels.items() if doc_id in attached], key=f"template_basis_{token}",
    )
    enabled = st.toggle("启用", value=bool(selected["enabled"]) if selected else True, key=f"template_enabled_{token}")
    is_default = st.toggle("设为默认模板", value=bool(selected["is_default"]) if selected else False, key=f"template_default_{token}")
    if st.button("保存模板", type="primary", icon=":material/save:", key=f"template_save_{token}"):
        original_rules = {str(row.get('rule_id', '')): row for row in parse_template_tasks(selected['required_items'] if selected else '[]')}
        task_items = [
            {**original_rules.get(str(row.get('规则ID') or ''), {}),
             **({'rule_id': row.get('规则ID') or 'CUSTOM-' + uuid.uuid4().hex[:8], 'criterion': str(row.get('判断标准') or '')}
                if selected and selected.get('engine_binding') == BOLT_ENGINE else {}),
             "text": str(row.get("必检项目") or "").strip(), "enabled": bool(row.get("启用"))}
            for row in current_rows if str(row.get("必检项目") or "").strip()
        ]
        if not name.strip():
            st.error("模板名称不能为空。")
        elif not task_items:
            st.error("请至少增加一个必检项目。")
        else:
            if selected:
                template_id = selected["id"]
                ctx.db.execute("UPDATE audit_templates SET name=?,description=?,required_items=?,review_instructions=?,enabled=?,is_default=? WHERE id=?",
                               (name.strip(), description.strip(), json.dumps(task_items, ensure_ascii=False), instructions.strip(), int(enabled), int(is_default), template_id))
            else:
                template_id = ctx.db.execute("""INSERT INTO audit_templates(name,description,required_document_types,required_items,review_instructions,enabled,is_default,created_at)
                    VALUES(?,?,?,?,?,?,?,datetime('now'))""", (name.strip(), description.strip(), "[]", json.dumps(task_items, ensure_ascii=False), instructions.strip(), int(enabled), int(is_default)))
            if is_default:
                ctx.db.execute("UPDATE audit_templates SET is_default=0 WHERE id<>?", (template_id,))
            with ctx.db.connect() as connection:
                connection.execute("DELETE FROM template_basis WHERE template_id=?", (template_id,))
                connection.executemany("INSERT INTO template_basis(template_id,document_id) VALUES(?,?)",
                                       [(template_id, basis_labels[label]) for label in default_basis])
            st.success("模板已保存。")
            st.rerun()


@st.dialog("确认删除规则模板")
def _confirm_delete_template(template_id: int, name: str) -> None:
    ctx = get_context()
    st.warning(f"将永久删除规则模板 **{name}** 及其默认依据绑定。已经创建的审核批次和模板快照不会删除。")
    confirmed = st.checkbox("我确认删除此模板", key=f"confirm_template_delete_{template_id}")
    if st.button(
        "永久删除模板", type="primary", icon=":material/delete_forever:",
        disabled=not confirmed, key=f"confirm_template_delete_button_{template_id}",
    ):
        try:
            _delete_template(ctx.db, template_id)
            st.toast("模板已删除", icon=":material/check_circle:")
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))


def _delete_template(db: object, template_id: int) -> None:
    """Delete through the current API or a hot-reload-safe compatibility path.

    Streamlit keeps the database object in ``cache_resource``. After an app
    update that adds ``ReviewDatabase.delete_template``, the already-cached
    instance can still belong to the older class and therefore lack the new
    method until the whole app is restarted.
    """
    delete = getattr(db, "delete_template", None)
    if callable(delete):
        delete(template_id)
        return

    # ``one`` and ``execute`` are long-standing ReviewDatabase APIs, so this
    # also works for an object cached before delete_template was introduced.
    template = db.one(  # type: ignore[attr-defined]
        "SELECT name,is_default FROM audit_templates WHERE id=?", (template_id,),
    )
    if not template:
        raise ValueError("模板不存在或已被删除")
    if template["is_default"]:
        raise ValueError("默认模板不能直接删除；请先将其他模板设为默认模板")
    db.execute("DELETE FROM audit_templates WHERE id=?", (template_id,))  # type: ignore[attr-defined]


def settings_page() -> None:
    ctx = get_context()
    _section_header("系统设置", "配置 LLM 和 OCR 服务。审核依据使用本地关键词与结构化规则，不需要向量模型。")
    current = ctx.config_store.get()
    presets = [row for row in ctx.config_store.presets() if row["category"] in {"llm", "ocr"}]
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
        llm_model = st.text_input(
            "LLM 模型", current.llm_model,
            placeholder="可留空：自动读取 /models 返回的第一个模型",
            help="建议明确填写模型 ID。留空时程序会先访问 Base URL 的 /models，并自动采用第一个可用模型。",
        )
        llm_temp = st.number_input("温度", 0.0, 2.0, current.llm_temperature, 0.1)
        llm_concurrency = st.number_input("LLM 并发数", 1, 16, current.llm_concurrency, 1,
                                          help="默认 1。LM Studio 单模型建议保持 1；只有服务端明确支持并行推理时再提高。")
        llm_timeout = st.number_input("单条 LLM 读取超时（秒）", 15, 300, current.llm_timeout_seconds, 5,
                                      help="默认 300 秒，为思考模型保留生成最终 JSON 的时间；超时只影响当前任务。")
        st.subheader("MinerU OCR")
        ocr_url = st.text_input("OCR Base URL", current.ocr_base_url)
        ocr_key = st.text_input("OCR API Key", mask_secret(current.ocr_api_key), type="password")
        ocr_backend = st.text_input("Backend", current.ocr_backend)
        ocr_lang = st.text_input("语言", current.ocr_lang)
        submitted = st.form_submit_button("保存配置", type="primary", icon=":material/save:")
    if submitted:
        try:
            updated = ctx.config_store.save({"allow_remote": allow_remote, "llm_base_url": llm_url, "llm_api_key": llm_key,
                "llm_model": llm_model, "llm_temperature": llm_temp,
                "llm_concurrency": llm_concurrency, "llm_timeout_seconds": llm_timeout,
                "ocr_base_url": ocr_url, "ocr_api_key": ocr_key, "ocr_backend": ocr_backend, "ocr_lang": ocr_lang})
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
        if st.button("测试 OCR", icon=":material/document_scanner:"):
            _show_test(lambda: MinerUClient(ctx.config_store.get()).test())
    st.caption("API Key 加密保存在 data/secrets，页面和日志不会显示明文。")
    st.caption("提示：思考型模型的连接测试只确认已经产生推理响应；正式审核会继续等待思考完成并读取最终 JSON。")
    with st.expander("保存当前连接为预设"):
        preset_category = st.segmented_control(
            "服务类型", ["LLM", "OCR"], default="LLM", key="preset_category"
        )
        preset_name = st.text_input("预设名称", placeholder="例如：本机 Ollama / 公司内网 MinerU")
        if st.button("保存预设", icon=":material/bookmark_add:"):
            try:
                ctx.config_store.save_preset({"LLM": "llm", "OCR": "ocr"}[preset_category], preset_name)
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
        ("OCR", lambda: MinerUClient(settings).test()),
    ]
    passed = 0
    with st.status("正在并行测试 LLM 和 OCR…", expanded=True) as status:
        st.caption("两项测试同时开始；已完成的服务会立即显示，不再等待前一项结束。")
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
    return {"llm": "LLM", "ocr": "OCR"}.get(category, category)


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

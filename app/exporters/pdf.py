from __future__ import annotations

import html
import json
from io import BytesIO
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont, TTFError
from reportlab.platypus import (
    CondPageBreak,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.lib.utils import ImageReader

from app.database import ReviewDatabase
from app.auditing.bolt_template import SCOPE_LABELS, SIGNATURE_NOTICE, SINGLE_NOTICE
from app.evidence import render_pdf_evidence_bytes


NAVY = colors.HexColor("#132238")
BLUE = colors.HexColor("#276EF1")
INK = colors.HexColor("#202939")
MUTED = colors.HexColor("#697386")
PALE = colors.HexColor("#F4F7FB")
LINE = colors.HexColor("#DDE3EC")
WHITE = colors.white
SEVERITY_COLORS = {
    "Critical": colors.HexColor("#B42318"),
    "Major": colors.HexColor("#D92D20"),
    "Minor": colors.HexColor("#EAAA08"),
    "Warning": colors.HexColor("#F79009"),
    "Review": colors.HexColor("#667085"),
}
CJK_FONT = "STSong-Light"
LATIN_FONT = "Helvetica"
LATIN_BOLD_FONT = "Helvetica-Bold"


def export_batch_pdf(db: ReviewDatabase, batch_id: str) -> bytes:
    batch = db.one(
        """SELECT b.*,t.name template_name FROM review_batches b
           LEFT JOIN audit_templates t ON t.id=b.template_id WHERE b.id=?""",
        (batch_id,),
    )
    if not batch:
        raise ValueError("审核批次不存在")
    findings = db.query(
        """SELECT * FROM findings WHERE batch_id=? ORDER BY
           CASE severity WHEN 'Critical' THEN 1 WHEN 'Major' THEN 2 WHEN 'Minor' THEN 3
           WHEN 'Warning' THEN 4 ELSE 5 END,id""",
        (batch_id,),
    )
    documents = db.query(
        """SELECT d.*,bd.role,bd.priority FROM batch_documents bd JOIN documents d ON d.id=bd.document_id
           WHERE bd.batch_id=? ORDER BY bd.priority,d.created_at""",
        (batch_id,),
    )
    rule_evaluations = db.query(
        "SELECT task_index,task_name,status,conclusion,error,source_file,metadata FROM rule_evaluations WHERE batch_id=? ORDER BY task_index",
        (batch_id,),
    )
    for row in rule_evaluations:
        metadata = json.loads(row.get('metadata') or '{}')
        row['task_name'] += ' · ' + (row.get('source_file') or '批次') + (' · WDC ' + metadata['wdc'] if metadata.get('wdc') else '')
    batch['_visual'] = db.query('''SELECT v.*,d.original_name FROM visual_evidence v JOIN documents d ON d.id=v.document_id
        WHERE v.batch_id=? ORDER BY v.id''', (batch_id,))
    return _build_report(batch, findings, documents, rule_evaluations)


def _build_report(batch: dict[str, object], findings: list[dict[str, object]],
                  documents: list[dict[str, object]], rule_evaluations: list[dict[str, object]] | None = None) -> bytes:
    _register_fonts()
    styles = _styles()
    output = BytesIO()
    document = SimpleDocTemplate(
        output, pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm,
        topMargin=22 * mm, bottomMargin=18 * mm,
        title="供应商质量审核报告", author="供应商质量文件智能审查工具",
    )
    story: list[object] = []
    supplier_docs = [item for item in documents if item["role"] == "supplier"]
    basis_docs = [item for item in documents if item["role"] != "supplier"]
    counts = {level: sum(item["severity"] == level for item in findings)
              for level in ("Critical", "Major", "Minor", "Warning", "Review")}

    story.extend(_cover(batch, findings, supplier_docs, basis_docs, counts, styles))
    if batch.get('_visual'):
        story.extend([PageBreak(), Paragraph('签章识别记录', styles['page_title']), Paragraph(SIGNATURE_NOTICE, styles['body'])])
        for row in batch['_visual']:
            detail = json.loads(row.get('details') or '{}')
            text = f"{row['original_name']} / 第{row['page']}页 / {row['kind']} / {row['state']} / {row['method']} / 区域{row['bbox']}：{detail.get('description', '')}"
            story.extend([Spacer(1, 3*mm), Paragraph(_safe(text), styles['body'])])
    if rule_evaluations:
        story.append(PageBreak())
        story.extend(_rule_summary(rule_evaluations, styles))
    story.append(PageBreak())
    story.extend(_detail_intro(findings, styles))
    file_lookup = {str(item["original_name"]): item for item in supplier_docs}
    for index, finding in enumerate(findings, start=1):
        if index > 1:
            story.append(PageBreak())
        story.extend(_finding_detail(index, finding, file_lookup, styles))

    if not findings:
        story.append(Spacer(1, 18 * mm))
        story.append(Paragraph("本批次未发现系统判定的问题或待确认项。", styles["empty"]))

    document.build(story, onFirstPage=_first_page, onLaterPages=_later_pages)
    return output.getvalue()


def _rule_summary(rows: list[dict[str, object]], styles: dict[str, ParagraphStyle]) -> list[object]:
    counts = {status: sum(str(row.get("status")) == status for row in rows)
              for status in ("合格", "不合格", "存疑", "不适用", "调用失败")}
    table_rows = [[Paragraph("序号", styles["table_head"]), Paragraph("独立审核任务", styles["table_head"]),
                   Paragraph("结论", styles["table_head"]), Paragraph("说明", styles["table_head"])]]
    for row in rows:
        explanation = str(row.get("conclusion") or row.get("error") or "-")
        table_rows.append([
            Paragraph(str(row.get("task_index") or "-"), styles["table_cell"]),
            Paragraph(_safe(str(row.get("task_name") or "-")), styles["table_cell"]),
            Paragraph(_safe(str(row.get("status") or "-")), styles["table_cell"]),
            Paragraph(_safe(explanation[:600]), styles["table_cell"]),
        ])
    return [
        Paragraph("逐条规则审核汇总", styles["page_title"]), Spacer(1, 2 * mm),
        Paragraph(
            f"模板共执行 {len(rows)} 个相互独立的 LLM 审核任务：合格 {counts['合格']}，不合格 {counts['不合格']}，"
            f"存疑 {counts['存疑']}，不适用 {counts['不适用']}，调用失败 {counts['调用失败']}。",
            styles["body"],
        ), Spacer(1, 6 * mm),
        Table(table_rows, colWidths=[14 * mm, 62 * mm, 22 * mm, 59 * mm], repeatRows=1,
              style=TableStyle([("BACKGROUND", (0, 0), (-1, 0), NAVY),
                                ("BOX", (0, 0), (-1, -1), 0.6, LINE),
                                ("INNERGRID", (0, 0), (-1, -1), 0.35, LINE),
                                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                                ("TOPPADDING", (0, 0), (-1, -1), 5),
                                ("BOTTOMPADDING", (0, 0), (-1, -1), 5)])),
    ]


def _cover(batch: dict[str, object], findings: list[dict[str, object]],
           supplier_docs: list[dict[str, object]], basis_docs: list[dict[str, object]],
           counts: dict[str, int], styles: dict[str, ParagraphStyle]) -> list[object]:
    supplier = str(batch.get("supplier_name") or "未识别")
    summary = _summary_text(counts, len(findings))
    batch_summary = json.loads(str(batch.get('summary') or '{}'))
    summary = SCOPE_LABELS.get(batch.get('audit_scope'), '完整文件包审核') + '。' + summary
    if batch.get('audit_scope') == 'single_document':
        summary = SINGLE_NOTICE + summary
    if batch_summary.get('uncovered_wdcs'):
        summary += ' 未覆盖WDC（未判合格）：' + '、'.join(batch_summary['uncovered_wdcs'])
    items: list[object] = [
        Spacer(1, 12 * mm),
        Paragraph("SUPPLIER QUALITY REVIEW", styles["eyebrow"]),
        Spacer(1, 3 * mm),
        Paragraph("供应商质量审核报告", styles["title"]),
        Spacer(1, 4 * mm),
        Paragraph(str(batch.get("name") or "审核批次"), styles["subtitle"]),
        Spacer(1, 11 * mm),
        Table(
            [[Paragraph("供应商", styles["meta_label"]), Paragraph(_safe(supplier), styles["supplier"])],
             [Paragraph("审核批次 ID", styles["meta_label"]), Paragraph(_safe(str(batch.get("id") or "-")), styles["meta_value"])],
             [Paragraph("审核模板", styles["meta_label"]), Paragraph(_safe(str(batch.get("template_name") or "未指定")), styles["meta_value"])],
             [Paragraph("生成时间", styles["meta_label"]), Paragraph(_safe(_display_time(str(batch.get("updated_at") or batch.get("created_at") or ""))), styles["meta_value"])],
             [Paragraph("审核范围", styles["meta_label"]), Paragraph(f"供应商文件 {len(supplier_docs)} 份 / 审核依据 {len(basis_docs)} 份", styles["meta_value"])]],
            colWidths=[34 * mm, 123 * mm],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), PALE), ("BOX", (0, 0), (-1, -1), 0.7, LINE),
                ("INNERGRID", (0, 0), (-1, -1), 0.35, LINE), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]),
        ),
        Spacer(1, 10 * mm),
        Paragraph("审核摘要", styles["section"]),
        Spacer(1, 3 * mm),
        _severity_cards(counts, styles),
        Spacer(1, 6 * mm),
        Table([[Paragraph(_safe(summary), styles["summary"]) ]], colWidths=[157 * mm],
              style=TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EEF4FF")),
                                ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#B8CCF4")),
                                ("LEFTPADDING", (0, 0), (-1, -1), 11), ("RIGHTPADDING", (0, 0), (-1, -1), 11),
                                ("TOPPADDING", (0, 0), (-1, -1), 9), ("BOTTOMPADDING", (0, 0), (-1, -1), 9)])),
        Spacer(1, 9 * mm),
        Paragraph("文件范围", styles["section"]),
        Spacer(1, 3 * mm),
        _document_table(supplier_docs, basis_docs, styles),
    ]
    return items


def _severity_cards(counts: dict[str, int], styles: dict[str, ParagraphStyle]) -> Table:
    labels = [("Critical", "严重"), ("Major", "主要"), ("Minor", "次要"), ("Warning", "警告"), ("Review", "待复核")]
    cells = []
    for level, label in labels:
        cells.append(Paragraph(
            f'<font color="#{SEVERITY_COLORS[level].hexval()[2:]}"><b>{counts[level]}</b></font><br/>'
            f'<font color="#697386">{label}</font>', styles["card"]
        ))
    return Table([cells], colWidths=[31.4 * mm] * 5, rowHeights=[19 * mm],
                 style=TableStyle([("BOX", (0, 0), (-1, -1), 0.6, LINE),
                                   ("INNERGRID", (0, 0), (-1, -1), 0.5, LINE),
                                   ("BACKGROUND", (0, 0), (-1, -1), WHITE),
                                   ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))


def _document_table(supplier_docs: list[dict[str, object]], basis_docs: list[dict[str, object]],
                    styles: dict[str, ParagraphStyle]) -> Table:
    rows = [[Paragraph("角色", styles["table_head"]), Paragraph("文件名称", styles["table_head"]),
             Paragraph("页数", styles["table_head"]), Paragraph("处理状态", styles["table_head"])]]
    for item in [*supplier_docs, *basis_docs]:
        role = "供应商文件" if item["role"] == "supplier" else "审核依据"
        rows.append([Paragraph(role, styles["table_cell"]), Paragraph(_safe(str(item["original_name"])), styles["table_cell"]),
                     Paragraph(str(item["page_count"]), styles["table_cell"]),
                     Paragraph(_safe(f"{item['parse_status']} / {item['ocr_status']}"), styles["table_cell"] )])
    if len(rows) == 1:
        rows.append([Paragraph("-", styles["table_cell"])] * 4)
    return Table(rows, colWidths=[28 * mm, 78 * mm, 16 * mm, 35 * mm], repeatRows=1,
                 style=TableStyle([("BACKGROUND", (0, 0), (-1, 0), NAVY),
                                   ("BOX", (0, 0), (-1, -1), 0.6, LINE),
                                   ("INNERGRID", (0, 0), (-1, -1), 0.35, LINE),
                                   ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                   ("LEFTPADDING", (0, 0), (-1, -1), 6),
                                   ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                                   ("TOPPADDING", (0, 0), (-1, -1), 5),
                                   ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))


def _detail_intro(findings: list[dict[str, object]], styles: dict[str, ParagraphStyle]) -> list[object]:
    return [Paragraph("问题详情", styles["page_title"]), Spacer(1, 2 * mm),
            Paragraph(f"以下共列出 {len(findings)} 项问题或待复核事项。截图仅在系统能够精确定位原文时显示。", styles["body"]),
            Spacer(1, 7 * mm)]


def _finding_detail(index: int, finding: dict[str, object], file_lookup: dict[str, dict[str, object]],
                    styles: dict[str, ParagraphStyle]) -> list[object]:
    severity = str(finding.get("severity") or "Review")
    severity_color = SEVERITY_COLORS.get(severity, MUTED)
    title = Table([[Paragraph(f"{index:02d}", styles["number"]),
                    Paragraph(_safe(str(finding.get("item") or "审核问题")), styles["finding_title"]),
                    Paragraph(_safe(severity), styles["severity"])]],
                  colWidths=[14 * mm, 118 * mm, 25 * mm], rowHeights=[12 * mm],
                  style=TableStyle([("BACKGROUND", (0, 0), (0, 0), NAVY),
                                    ("BACKGROUND", (-1, 0), (-1, 0), severity_color),
                                    ("BOX", (0, 0), (-1, -1), 0.7, LINE),
                                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                    ("LEFTPADDING", (0, 0), (-1, -1), 7),
                                    ("RIGHTPADDING", (0, 0), (-1, -1), 7)]))
    metadata = Table([
        [Paragraph("类别", styles["meta_label"]), Paragraph(_safe(str(finding.get("category") or "-")), styles["meta_value"]),
         Paragraph("人工状态", styles["meta_label"]), Paragraph(_safe(str(finding.get("status") or "-")), styles["meta_value"])],
        [Paragraph("供应商文件", styles["meta_label"]), Paragraph(_safe(str(finding.get("source_file") or "-")), styles["meta_value"]),
         Paragraph("原页", styles["meta_label"]), Paragraph(str(finding.get("source_page") or 1), styles["meta_value"])],
        [Paragraph("审核依据", styles["meta_label"]), Paragraph(_safe(str(finding.get("standard_file") or "模板内置规则")), styles["meta_value"]),
         Paragraph("条款", styles["meta_label"]), Paragraph(_safe(str(finding.get("standard_clause") or "-")), styles["meta_value"])],
    ], colWidths=[24 * mm, 63 * mm, 24 * mm, 46 * mm],
       style=TableStyle([("BACKGROUND", (0, 0), (-1, -1), PALE), ("BOX", (0, 0), (-1, -1), 0.6, LINE),
                         ("INNERGRID", (0, 0), (-1, -1), 0.3, LINE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                         ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                         ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    blocks: list[object] = [title, Spacer(1, 4 * mm), metadata, Spacer(1, 6 * mm)]
    try:
        finding_metadata = json.loads(str(finding.get("metadata") or "{}"))
    except (ValueError, TypeError):
        finding_metadata = {}
    downgrade_reasons = finding_metadata.get("downgrade_reasons", [])
    if downgrade_reasons:
        blocks.extend([Paragraph("自动降级原因", styles["detail_label"]),
                       Paragraph(_safe("；".join(str(reason) for reason in downgrade_reasons)), styles["detail_value"]),
                       Spacer(1, 3 * mm)])
    for label, key in (("问题说明", "description"), ("原报告证据", "source_text"), ("实际结果", "actual"),
                       ("审核要求", "requirement"), ("判断逻辑", "logic"), ("整改建议", "suggestion")):
        value = str(finding.get(key) or "-")
        blocks.append(KeepTogether([Paragraph(label, styles["detail_label"]),
                                    Paragraph(_safe(value[:4000]), styles["detail_value"]), Spacer(1, 3 * mm)]))

    try:
        evidence = finding_metadata.get("evidence", [])
    except (ValueError, TypeError):
        evidence = []
    if not evidence:
        evidence = [{"file": finding.get("source_file"), "page": finding.get("source_page"),
                     "source_text": finding.get("source_text"), "evidence_type": "source"}]
    for item in evidence[:4]:
        if item.get("evidence_type") == "absence":
            blocks.extend([Paragraph("缺失证明", styles["detail_label"]),
                           Paragraph(_safe(str(item.get("source_text") or "系统已扫描适用页面但未命中该字段")), styles["detail_value"])])
            continue
        source = file_lookup.get(str(item.get("file") or finding.get("source_file") or ""))
        screenshot = _matched_screenshot(source, {**finding, "source_page": item.get("page"), "source_text": item.get("source_text"),
                                                   "bbox": item.get("bbox")})
        if screenshot:
            blocks.extend([CondPageBreak(65 * mm), Spacer(1, 2 * mm), Paragraph("原报告问题位置", styles["detail_label"]),
                           Spacer(1, 2 * mm), _report_image(screenshot),
                           Paragraph(f"{_safe(str(item.get('file') or '原报告'))} · 第 {item.get('page') or 1} 页，红框为精确定位证据。", styles["image_caption"])])
    return blocks


def _matched_screenshot(source: dict[str, object] | None, finding: dict[str, object]) -> bytes | None:
    if not source:
        return None
    path = Path(str(source.get("stored_path") or ""))
    if not path.is_file() or path.suffix.casefold() != ".pdf":
        return None
    try:
        bbox_value = finding.get("bbox")
        bbox = tuple(float(value) for value in bbox_value) if isinstance(bbox_value, list) and len(bbox_value) == 4 else None
        image, matched = render_pdf_evidence_bytes(
            str(path), int(finding.get("source_page") or 1), str(finding.get("source_text") or ""),
            str(finding.get("actual") or ""), str(finding.get("item") or ""), bbox,
        )
        return image if matched else None
    except (OSError, ValueError, RuntimeError):
        return None


def _report_image(payload: bytes) -> Image:
    reader = ImageReader(BytesIO(payload))
    width, height = reader.getSize()
    max_width, max_height = 157 * mm, 82 * mm
    scale = min(max_width / width, max_height / height, 1.0)
    return Image(BytesIO(payload), width=width * scale, height=height * scale, hAlign="CENTER")


def _register_fonts() -> None:
    global CJK_FONT, LATIN_FONT, LATIN_BOLD_FONT

    # Prefer the native Song typeface on both platforms. Avoid PingFang and
    # Hiragino here: their macOS TTC files use PostScript/CFF outlines, which
    # ReportLab's TTFont parser cannot embed and raises TTFError for.
    cjk_candidates = (
        Path("C:/Windows/Fonts/simsun.ttc"),
        Path("C:/Windows/Fonts/simsun.ttf"),
        Path("/System/Library/Fonts/Supplemental/Songti.ttc"),
        Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
        Path("/usr/share/fonts/truetype/arphic/uming.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    )
    if _register_truetype("QaqcCJK", cjk_candidates):
        CJK_FONT = "QaqcCJK"
    else:
        if "STSong-Light" not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
        CJK_FONT = "STSong-Light"

    home_fonts = Path.home() / "Library" / "Fonts"
    latin_candidates = (
        Path("C:/Windows/Fonts/calibri.ttf"),
        home_fonts / "Calibri.ttf",
        Path("/Library/Fonts/Microsoft/Calibri.ttf"),
    )
    latin_bold_candidates = (
        Path("C:/Windows/Fonts/calibrib.ttf"),
        home_fonts / "Calibri Bold.ttf",
        Path("/Library/Fonts/Microsoft/Calibri Bold.ttf"),
    )
    LATIN_FONT = "QaqcLatin" if _register_truetype("QaqcLatin", latin_candidates) else "Helvetica"
    LATIN_BOLD_FONT = (
        "QaqcLatinBold" if _register_truetype("QaqcLatinBold", latin_bold_candidates) else "Helvetica-Bold"
    )


def _register_truetype(alias: str, candidates: tuple[Path, ...]) -> bool:
    if alias in pdfmetrics.getRegisteredFontNames():
        return True
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            pdfmetrics.registerFont(TTFont(alias, str(candidate), subfontIndex=0))
            return True
        except (OSError, ValueError, TTFError):
            continue
    return False


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "eyebrow": ParagraphStyle("eyebrow", parent=base["Normal"], fontName=LATIN_BOLD_FONT, fontSize=8,
                                  leading=10, textColor=BLUE, tracking=1.5),
        "title": ParagraphStyle("title", parent=base["Title"], fontName=CJK_FONT, fontSize=25,
                                leading=31, textColor=NAVY, alignment=TA_LEFT),
        "subtitle": ParagraphStyle("subtitle", parent=base["Normal"], fontName=CJK_FONT, fontSize=11,
                                   leading=15, textColor=MUTED),
        "section": ParagraphStyle("section", parent=base["Heading2"], fontName=CJK_FONT, fontSize=13,
                                  leading=18, textColor=NAVY, spaceAfter=0),
        "page_title": ParagraphStyle("page_title", parent=base["Heading1"], fontName=CJK_FONT, fontSize=20,
                                     leading=26, textColor=NAVY),
        "body": ParagraphStyle("body", parent=base["BodyText"], fontName=CJK_FONT, fontSize=9.2,
                               leading=14, textColor=INK),
        "meta_label": ParagraphStyle("meta_label", parent=base["Normal"], fontName=CJK_FONT, fontSize=8.3,
                                     leading=12, textColor=MUTED),
        "meta_value": ParagraphStyle("meta_value", parent=base["Normal"], fontName=CJK_FONT, fontSize=8.6,
                                     leading=12, textColor=INK),
        "supplier": ParagraphStyle("supplier", parent=base["Normal"], fontName=CJK_FONT, fontSize=10.5,
                                   leading=15, textColor=NAVY),
        "summary": ParagraphStyle("summary", parent=base["Normal"], fontName=CJK_FONT, fontSize=10,
                                  leading=15, textColor=NAVY),
        "card": ParagraphStyle("card", parent=base["Normal"], fontName=CJK_FONT, fontSize=8.5,
                               leading=15, alignment=TA_CENTER),
        "table_head": ParagraphStyle("table_head", parent=base["Normal"], fontName=CJK_FONT, fontSize=8,
                                     leading=11, textColor=WHITE),
        "table_cell": ParagraphStyle("table_cell", parent=base["Normal"], fontName=CJK_FONT, fontSize=7.8,
                                     leading=11, textColor=INK),
        "number": ParagraphStyle("number", parent=base["Normal"], fontName=LATIN_BOLD_FONT, fontSize=10,
                                 leading=12, textColor=WHITE, alignment=TA_CENTER),
        "finding_title": ParagraphStyle("finding_title", parent=base["Normal"], fontName=CJK_FONT, fontSize=12,
                                        leading=15, textColor=NAVY),
        "severity": ParagraphStyle("severity", parent=base["Normal"], fontName=LATIN_BOLD_FONT, fontSize=8,
                                   leading=10, textColor=WHITE, alignment=TA_CENTER),
        "detail_label": ParagraphStyle("detail_label", parent=base["Normal"], fontName=CJK_FONT, fontSize=8.3,
                                       leading=11, textColor=BLUE),
        "detail_value": ParagraphStyle("detail_value", parent=base["BodyText"], fontName=CJK_FONT, fontSize=9.2,
                                       leading=14, textColor=INK, borderColor=LINE, borderWidth=0.5,
                                       borderPadding=7, backColor=colors.HexColor("#FBFCFE")),
        "image_caption": ParagraphStyle("image_caption", parent=base["Normal"], fontName=CJK_FONT, fontSize=7.5,
                                        leading=10, textColor=MUTED, alignment=TA_CENTER, spaceBefore=3),
        "empty": ParagraphStyle("empty", parent=base["Normal"], fontName=CJK_FONT, fontSize=12,
                                leading=18, textColor=MUTED, alignment=TA_CENTER),
    }


def _first_page(canvas: object, document: object) -> None:
    _footer(canvas)


def _later_pages(canvas: object, document: object) -> None:
    canvas.saveState()
    canvas.setStrokeColor(LINE)
    canvas.line(18 * mm, A4[1] - 14 * mm, A4[0] - 18 * mm, A4[1] - 14 * mm)
    canvas.setFont(CJK_FONT, 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, A4[1] - 11 * mm, "供应商质量审核报告 / 问题详情")
    canvas.restoreState()
    _footer(canvas)


def _footer(canvas: object) -> None:
    canvas.saveState()
    canvas.setStrokeColor(LINE)
    canvas.line(18 * mm, 12 * mm, A4[0] - 18 * mm, 12 * mm)
    canvas.setFont(CJK_FONT, 7.2)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 8 * mm, "由供应商质量文件智能审查工具生成 / 结论须结合人工复核")
    canvas.drawRightString(A4[0] - 18 * mm, 8 * mm, f"第 {canvas.getPageNumber()} 页")
    canvas.restoreState()


def _summary_text(counts: dict[str, int], total: int) -> str:
    if counts["Critical"] or counts["Major"]:
        return f"本批次共发现 {total} 项问题或待复核事项，其中严重/主要问题 {counts['Critical'] + counts['Major']} 项。建议完成整改、补证并人工复核后再关闭批次。"
    if total:
        return f"本批次共发现 {total} 项次要、警告或待复核事项，未发现严重/主要问题。建议结合原报告逐项确认。"
    return "系统未发现问题或待复核事项。仍建议按企业质量流程完成必要的人工抽查。"


def _display_time(value: str) -> str:
    return value.replace("T", " ").replace("+00:00", " UTC") if value else "-"


def _safe(value: str) -> str:
    return html.escape(value).replace("\n", "<br/>")

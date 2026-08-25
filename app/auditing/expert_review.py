from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from app.models import Finding, PageText


FASTENER_TEMPLATE_NAME = "CUSTOMER 紧固件质量文件审核"
FASTENER_REQUIRED_CHECKS = [
    "A1 文件完整性",
    "A2 文件可读性/清晰度",
    "A3 签章/签字/日期有效性",
    "A4 文件来源与出具单位",
    "B2 WDC 号识别与格式",
    "B3 合格证日期",
    "B5 规格/尺寸/等级/标准一致性",
    "C1 检测标准存在性与适用性",
    "C2 化学成分",
    "C3 机械性能",
    "C5 检测结论与检测数据逻辑一致性",
    "D1 文件版本/页码完整性",
    "D2 多供应商/多批次风险",
]
FASTENER_REVIEW_INSTRUCTIONS = """你是维修备件紧固件质量文件审核专家。只依据本次文件和所选审核依据判断，不得引用未提供的标准或臆测。
CUSTOMER 文件包应包含 COC，以及每个有后续文件的 WDC 对应的 COI/MTR；COI/MTR应包含尺寸、机械性能和化学成分。
逐项检查文件可读性、签字/盖章、日期、出具单位、WDC、产品规格/尺寸/等级/标准、检测数据与结论一致性。
PO 在此场景仅作识别，不用非 CUSTOMER 格式订单号直接判废；WDC 从文件名或正文识别，去除分隔符后应为 8 或 10 位数字。
标准值或实测值缺失、公差写法无法判断、扫描页无法识别时判为存疑。Sample/Pass 数量不相等、实测值超出文件中明确标准但仍宣称合格、印记不一致时判为不合格。
每个不合格或存疑项必须给出原文件名、页码和可核验的原文/表格证据；没有证据不得下结论。"""


@dataclass(frozen=True)
class ExpertDocument:
    filename: str
    path: Path
    pages: list[PageText]
    ocr_status: str = "not_needed"

    @property
    def text(self) -> str:
        return "\n".join(page.text for page in self.pages)


def deterministic_fastener_audit(documents: list[ExpertDocument]) -> list[Finding]:
    findings: list[Finding] = []
    types = {_document_type(document) for document in documents}
    filenames = "；".join(document.filename for document in documents) or "未提供文件"
    if "COC" not in types:
        findings.append(Finding(
            "文件缺失", "Major", "A1 文件完整性", "CUSTOMER 文件包中未识别到 COC。",
            actual=f"批次文件清单：{filenames}", requirement="必须包含 COC 及各 WDC 对应的 COI/MTR",
            source_file=documents[0].filename if documents else "", source_page=1,
            source_text=f"批次文件清单：{filenames}", logic="已识别文件类型中不包含 COC",
            suggestion="请供应商补充带签字的 COC，并注明与 WDC 的对应关系。", confidence=1.0,
            metadata={"check_id": "A1", "result": "不合格"},
        ))
    if not ({"MTR", "COI"} & types):
        findings.append(Finding(
            "文件缺失", "Major", "A1 文件完整性", "未识别到 COI/MTR 检验文件。",
            actual=f"批次文件清单：{filenames}", requirement="每个有后续文件的 WDC 应有 COI/MTR",
            source_file=documents[0].filename if documents else "", source_page=1,
            source_text=f"批次文件清单：{filenames}", logic="已识别文件类型中不包含 COI 或 MTR",
            suggestion="请供应商补充包含尺寸、机械性能和化学成分的 COI/MTR。", confidence=1.0,
            metadata={"check_id": "A1", "result": "不合格"},
        ))

    for document in documents:
        findings.extend(_readability_findings(document))
        findings.extend(_inspection_table_findings(document))
        if _document_type(document) in {"MTR", "COI"}:
            findings.extend(_mtr_completeness_findings(document))
    return _dedupe(findings)


def build_expert_prompt(
    documents: list[ExpertDocument], instructions: str, existing: Iterable[Finding], basis: str = ""
) -> str:
    document_blocks: list[str] = []
    remaining = 48_000
    for document in documents:
        page_blocks: list[str] = []
        for page in document.pages:
            block = f"[文件={document.filename}][页={page.page}]\n{page.text.strip()}"
            if len(block) > remaining:
                block = block[: max(0, remaining)]
            if block:
                page_blocks.append(block)
                remaining -= len(block)
            if remaining <= 0:
                break
        document_blocks.append("\n".join(page_blocks))
        if remaining <= 0:
            break
    known = [
        {"category": item.category, "item": item.item, "description": item.description,
         "source_file": item.source_file, "page": item.source_page, "evidence": item.source_text or item.actual}
        for item in existing
    ]
    return f"""你是受证据约束的供应商质量文件审核专家。请找出确定性规则尚未发现的不合格项和存疑项，不输出合格项。

审核规则：
{instructions or FASTENER_FALLBACK_INSTRUCTIONS}

所选审核依据（为空表示没有额外标准，不得自行补充标准值）：
{basis[:12_000] or "未提供额外标准"}

确定性规则已发现的问题（不要重复）：
{json.dumps(known, ensure_ascii=False)}

供应商文件逐页证据：
{chr(10).join(document_blocks)}

硬性要求：
1. 只能依据上述内容；不确定一律为“存疑”，不得把缺少标准时的数值直接判为不合格。
2. evidence 必须逐字摘自对应文件页的原文或 [TABLE_ROW]，不得改写；没有可核验证据就不要输出。
3. result 只能是“不合格”或“存疑”。page 必须是正整数，source_file 必须与输入文件名完全一致。
4. 对 Sample/Pass、标准/实测、跨页炉号/规格冲突应逐行核验。
5. 只返回 JSON：{{"findings":[{{"check_id":"C5","result":"不合格","category":"检测结果不合格","item":"Head Height","description":"...","source_file":"...","page":1,"evidence":"[TABLE_ROW] ...","actual":"...","requirement":"...","logic":"...","suggestion":"...","confidence":0.95}}]}}
"""


FASTENER_FALLBACK_INSTRUCTIONS = "检查文件完整性、可读性、追溯字段一致性，以及文件中明确标准与实测结果的一致性。"


def findings_from_llm(payload: dict[str, object], documents: list[ExpertDocument]) -> list[Finding]:
    page_lookup = {(document.filename, page.page): page.text for document in documents for page in document.pages}
    output: list[Finding] = []
    rows = payload.get("findings")
    if not isinstance(rows, list):
        return []
    for row in rows[:80]:
        if not isinstance(row, dict):
            continue
        filename = str(row.get("source_file") or "")
        try:
            page = int(row.get("page") or 0)
        except (TypeError, ValueError):
            continue
        evidence = str(row.get("evidence") or "").strip()
        result = str(row.get("result") or "")
        source = page_lookup.get((filename, page), "")
        if result not in {"不合格", "存疑"} or not evidence or not _evidence_present(evidence, source):
            continue
        confidence = _confidence(row.get("confidence"))
        output.append(Finding(
            str(row.get("category") or ("待人工确认" if result == "存疑" else "数据不一致")),
            "Review" if result == "存疑" else "Major",
            str(row.get("item") or row.get("check_id") or "专家复核"),
            str(row.get("description") or "专家复核发现问题"),
            str(row.get("actual") or evidence), str(row.get("requirement") or "需按审核规则核验"),
            filename, page, evidence, logic=str(row.get("logic") or "原页证据触发专家审核规则"),
            suggestion=str(row.get("suggestion") or "请供应商核实并补充可追溯证据。"), confidence=confidence,
            metadata={"check_id": str(row.get("check_id") or ""), "result": result, "origin": "llm_expert"},
        ))
    return _dedupe(output)


def _document_type(document: ExpertDocument) -> str:
    haystack = f"{document.filename}\n{document.text}".casefold()
    if re.search(r"\bcoc\b|certificate of conformance|certificate of conformity|合格证", haystack):
        return "COC"
    if re.search(r"certificate of inspection|\bcoi\b", haystack):
        return "COI"
    if re.search(r"\bmtr\b|material test|quality certificate|chemical composition", haystack):
        return "MTR"
    return "OTHER"


def _readability_findings(document: ExpertDocument) -> list[Finding]:
    output: list[Finding] = []
    for page in document.pages:
        if len(re.sub(r"\s+", "", page.text)) >= 40:
            continue
        status = "OCR 已调用但该页仍无足够文字" if document.ocr_status == "completed" else "该页无可提取文本，且未完成逐页 OCR"
        output.append(Finding(
            "无法识别", "Review", "A2 文件可读性/清晰度", f"第 {page.page} 页无法可靠识别。",
            actual=status, requirement="关键页和关键数据必须可辨认", source_file=document.filename,
            source_page=page.page, source_text=f"第 {page.page} 页提取文字不足 40 个字符",
            logic="逐页文字密度检查未通过", suggestion="请配置 OCR 后重新审核，或人工查看原页并确认签字、盖章及关键数据。",
            confidence=1.0, metadata={"check_id": "A2", "result": "存疑"},
        ))
    return output


def _inspection_table_findings(document: ExpertDocument) -> list[Finding]:
    output: list[Finding] = []
    for page in document.pages:
        for line in page.text.splitlines():
            if not line.startswith("[TABLE_ROW] "):
                continue
            cells = [cell.strip() for cell in line.removeprefix("[TABLE_ROW] ").split(" || ")]
            if len(cells) < 4:
                continue
            sample = _integer(cells[-2])
            passed = _integer(cells[-1])
            if sample is None or passed is None or sample <= 0 or passed < 0 or passed > sample:
                continue
            item = cells[0]
            if passed < sample:
                output.append(Finding(
                    "检测结果不合格", "Major", item,
                    f"{item} 抽检 {sample} 件，仅 {passed} 件通过，存在 {sample - passed} 件未通过。",
                    actual=f"Sample={sample}, Pass={passed}", requirement="抽检样本应全部通过，且结论应与 Pass 数一致",
                    source_file=document.filename, source_page=page.page, source_text=line,
                    logic=f"Pass({passed}) < Sample({sample})",
                    suggestion=f"请供应商解释 {sample - passed} 件未通过的处置、隔离和复验结果，并更正报告结论。",
                    confidence=1.0, metadata={"check_id": "C5", "result": "不合格"},
                ))
            elif len(cells) == 4 and _looks_like_spec(cells[1]):
                output.append(Finding(
                    "检测数据缺失", "Review", item, f"{item} 有标准和抽检数量，但未识别到实测值。",
                    actual="实测值空白", requirement=cells[1], source_file=document.filename,
                    source_page=page.page, source_text=line, logic="表格行仅含项目、标准、Sample、Pass，缺少 Result",
                    suggestion="请供应商补充该项目的实际检验值，不能仅以 Pass 数代替实测数据。",
                    confidence=0.95, metadata={"check_id": "C5", "result": "存疑"},
                ))
            if len(cells) >= 5:
                bounds = _spec_bounds(cells[-4])
                actual_bounds = _result_bounds(cells[-3])
                if bounds and actual_bounds and not _within(bounds, actual_bounds):
                    output.append(Finding(
                        "数值不符合", "Major", item, f"{item} 实测范围超出报告中列明的标准范围。",
                        actual=cells[-3], requirement=cells[-4], source_file=document.filename,
                        source_page=page.page, source_text=line,
                        logic=f"实测范围 {cells[-3]} 不满足标准 {cells[-4]}",
                        suggestion="请供应商隔离不符合产品并提交复验或纠正后的报告。",
                        confidence=1.0, metadata={"check_id": "C5", "result": "不合格"},
                    ))
    return output


def _mtr_completeness_findings(document: ExpertDocument) -> list[Finding]:
    text = document.text.casefold()
    checks = [
        ("尺寸检验", ("dimensions of spec", "body diameter", "尺寸")),
        ("机械性能", ("mechanical properties", "tensile strength", "机械性能")),
        ("化学成分", ("chemical composition", "化学成分")),
    ]
    output: list[Finding] = []
    for label, markers in checks:
        if any(marker in text for marker in markers):
            continue
        output.append(Finding(
            "检验项目缺失", "Review", label, f"COI/MTR 中未识别到{label}内容。",
            actual="未识别", requirement=f"COI/MTR 应包含{label}", source_file=document.filename,
            source_page=1, source_text=f"全文未检出{label}对应标题或项目",
            logic=f"未找到关键词：{'、'.join(markers)}", suggestion=f"请供应商补充包含{label}数据的报告页。",
            confidence=0.9, metadata={"check_id": "A1", "result": "存疑"},
        ))
    return output


def _integer(value: str) -> int | None:
    return int(value) if re.fullmatch(r"\d+", value.strip()) else None


def _looks_like_spec(value: str) -> bool:
    return bool(re.search(r"\d", value) and re.search(r"min|max|≥|≤|[-~]", value, re.I))


def _spec_bounds(value: str) -> tuple[float | None, float | None] | None:
    value = value.strip().casefold()
    numbers = [float(item) for item in re.findall(r"\d+(?:\.\d+)?", value)]
    if not numbers:
        return None
    if "min" in value or "≥" in value:
        return numbers[-1], None
    if "max" in value or "≤" in value:
        return None, numbers[-1]
    if len(numbers) == 2 and re.search(r"[-~～]", value):
        return min(numbers), max(numbers)
    return None


def _result_bounds(value: str) -> tuple[float, float] | None:
    numbers = [float(item) for item in re.findall(r"\d+(?:\.\d+)?", value)]
    if not numbers:
        return None
    return min(numbers), max(numbers)


def _within(expected: tuple[float | None, float | None], actual: tuple[float, float]) -> bool:
    lower, upper = expected
    return (lower is None or actual[0] >= lower) and (upper is None or actual[1] <= upper)


def _evidence_present(evidence: str, source: str) -> bool:
    normalize = lambda value: re.sub(r"\s+", "", value).casefold()
    needle, haystack = normalize(evidence), normalize(source)
    return len(needle) >= 8 and needle in haystack


def _confidence(value: object) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.7


def _dedupe(items: list[Finding]) -> list[Finding]:
    seen: set[tuple[str, str, int, str]] = set()
    output: list[Finding] = []
    for item in items:
        key = (item.category, item.source_file, item.source_page, re.sub(r"\s+", "", item.source_text))
        if key not in seen:
            seen.add(key)
            output.append(item)
    return output

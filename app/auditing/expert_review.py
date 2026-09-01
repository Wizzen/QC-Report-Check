from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from app.models import Finding, PageText


FASTENER_TEMPLATE_NAME = "紧固件质量文件审核"
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
紧固件质量文件包应包含 COC，以及每个有后续文件的 WDC 对应的 COI/MTR；COI/MTR应包含尺寸、机械性能和化学成分。
逐项检查文件可读性、签字/盖章、日期、出具单位、WDC、产品规格/尺寸/等级/标准、检测数据与结论一致性。
PO 在此场景仅作识别，不因订单号不符合特定客户格式而直接判废；WDC 从文件名或正文识别，去除分隔符后应为 8 或 10 位数字。
标准值或实测值缺失、公差写法无法判断、扫描页无法识别时判为存疑。Sample/Pass 数量不相等、实测值超出文件中明确标准但仍宣称合格、印记不一致时判为不合格。
每个不合格或存疑项必须给出原文件名、页码和可核验的原文/表格证据；没有证据不得下结论。"""


def parse_template_tasks(raw: object) -> list[dict[str, object]]:
    """Read both legacy string lists and editable task rows."""
    try:
        values = json.loads(str(raw or "[]")) if not isinstance(raw, list) else raw
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    output: list[dict[str, object]] = []
    if not isinstance(values, list):
        return output
    for value in values:
        if isinstance(value, dict):
            text = str(value.get("text") or value.get("item") or "").strip()
            enabled = bool(value.get("enabled", True))
        else:
            text = str(value).strip()
            enabled = True
        if text:
            output.append({"text": text, "enabled": enabled})
    return output


def enabled_template_tasks(raw: object) -> list[str]:
    return [str(row["text"]) for row in parse_template_tasks(raw) if row["enabled"]]


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
            "文件缺失", "Major", "A1 文件完整性", "紧固件质量文件包中未识别到 COC。",
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
    # The service chunks long reports before calling this builder. Keep a hard
    # guard here as well for local models with an 8K context window.
    remaining = 16_000
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
{basis[:6_000] or "未提供额外标准"}

确定性规则已发现的问题（不要重复）：
{json.dumps(known, ensure_ascii=False)}

供应商文件逐页证据：
{chr(10).join(document_blocks)}

硬性要求：
1. 只能依据上述内容；不确定一律为“存疑”，不得把缺少标准时的数值直接判为不合格。
2. evidence 必须逐字摘自对应文件页的原文或 [TABLE_ROW]，不得改写；没有可核验证据就不要输出。
3. result 只能是“不合格”或“存疑”。page 必须是正整数，source_file 必须与输入文件名完全一致。
4. 对 Sample/Pass、标准/实测、跨页炉号/规格冲突应逐行核验。
5. 同时识别每份文件的供应商、制造商、材料生产厂或报告出具机构；排除 Customer、Buyer、Purchaser 等采购方。名称必须逐字存在于该文件原文。
6. 只返回 JSON：{{"documents":[{{"source_file":"报告.pdf","supplier_names":["供应商 A","材料厂 B"]}}],"findings":[{{"check_id":"C5","result":"不合格","category":"检测结果不合格","item":"Head Height","description":"...","source_file":"...","page":1,"evidence":"[TABLE_ROW] ...","actual":"...","requirement":"...","logic":"...","suggestion":"...","confidence":0.95}}]}}
"""


def build_rule_task_prompt(
    task: str, documents: list[ExpertDocument], instructions: str, existing: Iterable[Finding], basis: str = ""
) -> str:
    evidence_blocks = []
    remaining = 9_000
    for document in documents:
        for page in document.pages:
            block = f"[文件={document.filename}][页={page.page}]\n{page.text.strip()}"
            block = block[:remaining]
            if block:
                evidence_blocks.append(block)
                remaining -= len(block)
            if remaining <= 0:
                break
        if remaining <= 0:
            break
    known = [{"item": item.item, "result": item.severity, "evidence": item.source_text or item.actual} for item in existing]
    return f"""你是受证据约束的供应商质量文件审核专家。本次只执行一个独立审核任务，不得顺带判断其他任务。

当前审核任务：{task}

通用审核边界：
{instructions or FASTENER_FALLBACK_INSTRUCTIONS}

所选审核依据：
{basis[:3_000] or "未提供额外标准，不得自行补充标准值"}

本地规则已发现内容（仅用于避免矛盾）：
{json.dumps(known, ensure_ascii=False)[:3_000]}

本批次文件清单：{json.dumps([item.filename for item in documents], ensure_ascii=False)}

逐页证据：
{chr(10).join(evidence_blocks)}

判断要求：
1. result 只能是“合格”“不合格”“存疑”“不适用”。
2. 不合格或存疑必须给出逐字存在于对应原页的 evidence；若判断的是文件/字段完全缺失，可使用 evidence_type="absence" 并在 checked_scope 写明检查过的文件和页面。
3. 不得引用输入之外的标准，不得编造页码、数值或证据。
4. 只返回一个 JSON 对象：
{{"task":"{task}","result":"合格|不合格|存疑|不适用","conclusion":"...","source_file":"...","page":1,"evidence":"...","evidence_type":"source|absence","checked_scope":"...","actual":"...","requirement":"...","logic":"...","suggestion":"...","confidence":0.0}}
"""


def rule_evaluation_from_llm(
    payload: dict[str, object], task: str, documents: list[ExpertDocument]
) -> tuple[dict[str, object], Finding | None]:
    # The adaptive path uses compact keys to reduce local-model generation time.
    # Full-key payloads remain supported for deep mode and compatibility.
    result = str(payload.get("result") or payload.get("r") or "").strip()
    if result not in {"合格", "不合格", "存疑", "不适用"}:
        raise ValueError("独立规则结果必须是合格、不合格、存疑或不适用")
    filename = str(payload.get("source_file") or payload.get("f") or "").strip()
    try:
        page = int(payload.get("page") or payload.get("p") or 0)
    except (TypeError, ValueError):
        page = 0
    evidence = str(payload.get("evidence") or payload.get("e") or "").strip()
    evidence_type = str(payload.get("evidence_type") or payload.get("t") or "source").strip()
    checked_scope = str(payload.get("checked_scope") or payload.get("s") or "").strip()
    page_lookup = {(document.filename, item.page): item.text for document in documents for item in document.pages}
    if result in {"不合格", "存疑"}:
        if evidence_type == "absence":
            if not checked_scope:
                evidence_type = "unlocated"
                evidence = ""
            else:
                evidence = checked_scope
        elif not filename or page <= 0 or not evidence or not _evidence_present(evidence, page_lookup.get((filename, page), "")):
            # Never publish fabricated evidence. Keep the model's signal as a
            # manual-review item without a red box instead of spending another
            # LLM call trying to repair a quote.
            result = "存疑"
            evidence_type = "unlocated"
            evidence = ""
        if not filename and documents:
            filename = documents[0].filename
        page = page or 1
    confidence = _confidence(payload.get("confidence") if "confidence" in payload else payload.get("q"))
    conclusion = str(payload.get("conclusion") or payload.get("c") or "")
    if evidence_type == "unlocated":
        conclusion = (conclusion or "模型发现疑点") + "（证据未能自动定位，需人工复核）"
    evaluation = {
        "task_name": task, "status": result, "conclusion": conclusion,
        "source_file": filename, "source_page": page, "evidence": evidence, "evidence_type": evidence_type,
        "actual": str(payload.get("actual") or ""), "requirement": str(payload.get("requirement") or ""),
        "logic": str(payload.get("logic") or ""), "suggestion": str(payload.get("suggestion") or ""),
        "confidence": confidence,
    }
    if result not in {"不合格", "存疑"}:
        return evaluation, None
    finding = Finding(
        "独立规则复核", "Major" if result == "不合格" else "Review", task,
        conclusion or f"独立规则判断为{result}",
        evaluation["actual"], evaluation["requirement"], filename, page or 1, evidence,
        logic=evaluation["logic"], suggestion=evaluation["suggestion"] or "请人工复核并要求供应商补充证据。",
        confidence=confidence, metadata={"origin": "llm_rule_task", "result": result, "evidence_type": evidence_type},
    )
    return evaluation, finding


FASTENER_FALLBACK_INSTRUCTIONS = "检查文件完整性、可读性、追溯字段一致性，以及文件中明确标准与实测结果的一致性。"


_COMPANY_PATTERNS = (
    re.compile(r"[A-Z0-9][A-Z0-9&'().,\-/ ]{2,}?(?:CO\.?\s*,?\s*LTD\.?|COMPANY\s+LIMITED|LIMITED|CORPORATION|CORP\.?|LLC\b|INC(?:\.|\b))", re.I),
    re.compile(r"[\u3400-\u9fffA-Za-z0-9（）()·&\-]{2,}(?:有限责任公司|股份有限公司|有限公司|集团公司|检测中心|研究院)"),
)
_PARTY_PREFIX = re.compile(
    r"^(?:supplier|manufacturer|producer|mill|issued\s+by|inspection\s+by|test(?:ing)?\s+laboratory|company|供应商|制造商|生产商|钢厂|出具单位|检测机构)\s*[:：\-]?\s*",
    re.I,
)
_BUYER_CONTEXT = re.compile(r"\b(?:customer|buyer|purchaser|consignee|ship\s+to|sold\s+to)\b|(?:客户|买方|采购方|收货方)", re.I)


def extract_supplier_names(documents_or_pages: ExpertDocument | list[PageText]) -> list[str]:
    """Extract legal entity names from OCR/text as a deterministic fallback."""
    pages = documents_or_pages.pages if isinstance(documents_or_pages, ExpertDocument) else documents_or_pages
    names: list[str] = []
    seen: set[str] = set()
    for page in pages:
        for raw_line in page.text.splitlines():
            line = raw_line.removeprefix("[TABLE_ROW] ").replace(" || ", " ").strip()
            for pattern in _COMPANY_PATTERNS:
                for match in pattern.finditer(line):
                    name = _PARTY_PREFIX.sub("", match.group(0)).strip(" \t:：,;，；-|_")
                    if "\ufffd" in name:
                        continue
                    before = line[:match.start()]
                    if _BUYER_CONTEXT.search(before) and not _PARTY_PREFIX.search(before):
                        continue
                    key = _normal_name(name)
                    if len(key) < 5 or key in seen:
                        continue
                    seen.add(key)
                    names.append(re.sub(r"\s+", " ", name))
    return names[:12]


def supplier_names_from_llm(payload: dict[str, object], documents: list[ExpertDocument]) -> dict[str, list[str]]:
    """Accept only LLM names that can be found verbatim (ignoring punctuation) in the source file."""
    sources = {document.filename: _normal_name(document.text) for document in documents}
    output: dict[str, list[str]] = {}
    rows = payload.get("documents")
    if not isinstance(rows, list):
        return output
    for row in rows:
        if not isinstance(row, dict):
            continue
        filename = str(row.get("source_file") or "")
        values = row.get("supplier_names")
        if filename not in sources or not isinstance(values, list):
            continue
        accepted: list[str] = []
        for value in values[:12]:
            name = re.sub(r"\s+", " ", str(value)).strip(" \t:：,;，；-|_")
            if "\ufffd" in name:
                continue
            normalized = _normal_name(name)
            if len(normalized) >= 5 and normalized in sources[filename] and normalized not in {_normal_name(item) for item in accepted}:
                accepted.append(name)
        output[filename] = accepted
    return output


def _normal_name(value: str) -> str:
    return re.sub(r"[^0-9a-z\u3400-\u9fff]+", "", value.casefold())


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

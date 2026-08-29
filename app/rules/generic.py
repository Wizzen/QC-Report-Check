from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from difflib import SequenceMatcher
from pathlib import Path

from app.models import ExtractedItem, Finding, PageText

RULE_VERSION = 1
GENERIC_TEMPLATE_NAME = "通用材料质量审核"

GROUPS = {
    "文档": {"DOC-003": "Page X of N 与实际总页数冲突", "DOC-004": "页码不连续", "DOC-005": "疑似重复页面"},
    "基础信息": {"INF-001": "报告编号缺失", "INF-002": "供应商/制造商缺失", "INF-003": "Material/Grade 缺失", "INF-004": "Heat No 缺失", "INF-005": "规格信息缺失", "INF-007": "检测日期缺失", "INF-008": "报告日期缺失", "INF-009": "Standard 缺失", "INF-010": "最终 Result 缺失", "INF-015": "关键数据单位缺失"},
    "一致性": {"CON-001": "Material 不同页面不一致", "CON-002": "Heat No 不一致", "CON-003": "Batch No 不一致", "CON-004": "Report No 不一致", "CON-006": "Thickness 不一致", "CON-007": "尺寸不一致", "CON-010": "Standard 不一致", "CON-013": "材料描述冲突"},
    "日期": {"DATE-001": "Test Date < Manufacturing Date", "DATE-002": "Issue Date < Test Date", "DATE-003": "Test Date < Sampling Date", "DATE-005": "明显未来日期"},
    "基础数值": {"NUM-001": "无法解析应为数值的字段", "NUM-002": "强度 < 0", "NUM-003": "伸长率 < 0", "NUM-004": "硬度 < 0", "NUM-005": "化学元素 < 0", "NUM-006": "百分比 > 100%", "NUM-007": "尺寸为 0", "NUM-008": "尺寸 < 0", "NUM-011": "疑似小数点或数量级错误", "NUM-014": "关键测试字段为空"},
    "Mechanical": {"MEC-001": "Tensile Strength < Yield Strength", "MEC-002": "Tensile Strength = 0", "MEC-003": "Yield Strength = 0", "MEC-007": "Elongation < 0", "MEC-008": "Elongation > 100%", "MEC-012": "Hardness < 0"},
    "Chemistry": {"CHEM-001": "元素值 < 0", "CHEM-002": "单元素 > 100%", "CHEM-004": "Chemical unit 缺失", "CHEM-005": "疑似 % / ppm 单位混淆", "CHEM-006": "同一个 Heat No、同一元素出现矛盾结果", "CHEM-007": "成分表无法关联 Material / Heat No"},
    "Units": {"UNIT-001": "关键结果缺失单位", "UNIT-002": "同字段存在冲突单位", "UNIT-003": "MPa / Pa / GPa 数量级异常", "UNIT-004": "MPa / ksi 疑似错误混用", "UNIT-005": "mm / inch 疑似错误混用", "UNIT-008": "ppm / % 疑似错误混用"},
    "结果一致性": {"RES-001": "Actual < Min，但是 Result = PASS", "RES-002": "Actual > Max，但是 Result = PASS", "RES-003": "存在 NG / FAIL 项，但 Overall Result = PASS", "RES-004": "全部子项目 PASS，但 Overall Result = FAIL", "RES-005": "表格 FAIL，正文写 ACCEPTED", "RES-006": "Actual 为空但 Result = PASS", "RES-008": "Requirement 与 Conclusion 明显冲突"},
}
TITLE = {code: title for values in GROUPS.values() for code, title in values.items()}


@dataclass(frozen=True)
class GenericDocument:
    document_id: str
    filename: str
    path: Path
    pages: list[PageText]
    fields: list[ExtractedItem]
    document_type: str
    type_confidence: float
    supplier_name: str = ""

    @property
    def text(self) -> str:
        return "\n".join(page.text for page in self.pages)


def classify_document(filename: str, pages: list[PageText]) -> tuple[str, float]:
    text = (filename + "\n" + "\n".join(p.text[:5000] for p in pages)).casefold()
    scores = {
        "MTR": sum(x in text for x in ("mtr", "material test", "chemical composition", "mechanical properties")),
        "COI": sum(x in text for x in ("coi", "certificate of inspection")),
        "COC": sum(x in text for x in ("coc", "certificate of conformance", "certificate of conformity", "合格证")),
        "INSPECTION_REPORT": sum(x in text for x in ("inspection report", "test report", "检测报告", "检验报告")),
    }
    kind, score = max(scores.items(), key=lambda pair: pair[1])
    return (kind, min(.98, .55 + score * .12)) if score else ("UNKNOWN", .25)


def seed_generic_rules(db: object) -> int:
    template = db.one("SELECT id FROM audit_templates WHERE name=?", (GENERIC_TEMPLATE_NAME,))
    if not template:
        return 0
    count = 0
    for group, values in GROUPS.items():
        for code, title in values.items():
            severity = "Major" if code.startswith(("RES-", "MEC-", "CON-")) else "Review"
            db.execute("INSERT OR IGNORE INTO audit_rules(code,group_name,title,applies_to,severity,evaluator,current_version) VALUES(?,?,?,?,?,'generic',1)",
                       (code, group, title, json.dumps(["MTR", "COI", "INSPECTION_REPORT"]), severity))
            db.execute("INSERT OR IGNORE INTO audit_rule_versions(rule_code,version,change_reason,created_at) VALUES(?,1,?,datetime('now'))",
                       (code, "v0.4.0 首次引入通用证据规则"))
            db.execute("INSERT OR IGNORE INTO template_rule_versions(template_id,rule_code,rule_version,enabled) VALUES(?,?,1,1)",
                       (template["id"], code))
            count += 1
    return count


def run_generic_rules(documents: list[GenericDocument], enabled: set[str] | None = None) -> list[Finding]:
    enabled = set(TITLE) if enabled is None else enabled
    output: list[Finding] = []
    for doc in documents:
        output += _page_rules(doc, enabled)
        if doc.type_confidence < .7 or doc.document_type == "UNKNOWN":
            output.append(_make("DOC-TYPE", doc, 1, "文档类型待确认", "无法可靠识别文档类型，已跳过字段缺失规则。", doc.filename,
                                "请人工确认文档类型", "类型置信度低于 0.70", "Review", .6))
            continue
        if doc.document_type in {"MTR", "COI", "INSPECTION_REPORT"}:
            output += _missing_rules(doc, enabled)
            output += _value_rules(doc, enabled)
            output += _date_rules(doc, enabled)
            output += _result_rules(doc, enabled)
    output += _consistency_rules(documents, enabled)
    return _dedupe(output)


def _page_rules(doc: GenericDocument, enabled: set[str]) -> list[Finding]:
    output: list[Finding] = []
    declared = []
    for page in doc.pages:
        for match in re.finditer(r"(?i)page\s*(\d+)\s*(?:of|/)\s*(\d+)", page.text):
            declared.append((page.page, int(match[1]), int(match[2]), _line(page.text, match.start())))
    for page, current, total, source in declared:
        if "DOC-003" in enabled and total != len(doc.pages):
            output.append(_make("DOC-003", doc, page, TITLE["DOC-003"], f"声明总页数 {total}，实际 {len(doc.pages)} 页。", source,
                                f"N 应等于 {len(doc.pages)}", f"{total} != {len(doc.pages)}", "Major"))
    sequence = [row[1] for row in declared]
    if "DOC-004" in enabled and sequence and sequence != list(range(sequence[0], sequence[0] + len(sequence))):
        output.append(_make("DOC-004", doc, declared[0][0], TITLE["DOC-004"], f"识别页码顺序：{sequence}", declared[0][3],
                            "页码应连续", "页码序列不连续", "Major", evidence=[_ev(doc, p, s) for p, _, _, s in declared]))
    if "DOC-005" in enabled:
        for i, first in enumerate(doc.pages):
            a = _norm(first.text)
            if len(a) < 80:
                continue
            for second in doc.pages[i + 1:]:
                if SequenceMatcher(None, a, _norm(second.text)).ratio() >= .985:
                    output.append(_make("DOC-005", doc, first.page, TITLE["DOC-005"], f"第 {first.page} 页与第 {second.page} 页文本高度相似。",
                                        first.text[:300], "页面内容应唯一", "归一化文本相似度 >= 98.5%", "Review", .86,
                                        evidence=[_ev(doc, first.page, first.text[:300]), _ev(doc, second.page, second.text[:300])]))
    return output


def _missing_rules(doc: GenericDocument, enabled: set[str]) -> list[Finding]:
    checks = {
        "INF-001": ("报告编号", r"(?i)report\s*(?:no|number)|报告编号"), "INF-002": ("供应商/制造商", r"(?i)supplier|manufacturer|供应商|制造商"),
        "INF-003": ("Material/Grade", r"(?i)material|grade|材质|材料牌号"), "INF-004": ("Heat No", r"(?i)heat\s*(?:no|number)|炉号"),
        "INF-005": ("规格信息", r"(?i)specification|size|dimension|规格|尺寸"), "INF-007": ("检测日期", r"(?i)test\s*date|检测日期|试验日期"),
        "INF-008": ("报告日期", r"(?i)issue\s*date|report\s*date|报告日期|签发日期"), "INF-009": ("Standard", r"(?i)standard|执行标准|标准号"),
        "INF-010": ("最终 Result", r"(?i)overall\s*result|conclusion|accepted|passed|最终结果|结论"),
    }
    output = []
    for code, (label, pattern) in checks.items():
        present = bool(re.search(pattern, doc.text)) or (code == "INF-002" and bool(doc.supplier_name))
        if code in enabled and not present:
            output.append(_make(code, doc, 1, TITLE[code], f"已检查 {len(doc.pages)} 页，未识别到{label}。", f"已检查 {len(doc.pages)} 页",
                                f"{doc.document_type} 应包含{label}", "全文字段和标签均未命中", "Review", .82,
                                evidence=[{**_ev(doc, 1, f"已检查 {len(doc.pages)} 页"), "evidence_type": "absence"}]))
    return output


ELEMENTS = {"C", "Si", "Mn", "P", "S", "Cr", "Ni", "Mo"}


def _value_rules(doc: GenericDocument, enabled: set[str]) -> list[Finding]:
    output = []
    by_key = defaultdict(list)
    for field in doc.fields:
        by_key[field.key].append(field)
        if field.category != "measurement" or not isinstance(field.value, (int, float)):
            continue
        value = float(field.value)
        cases = [
            ("NUM-002", "强度" in field.key and value < 0), ("NUM-003", "延伸" in field.key and value < 0),
            ("NUM-004", "硬度" in field.key and value < 0), ("NUM-005", field.key in ELEMENTS and value < 0),
            ("NUM-006", field.unit == "%" and value > 100), ("NUM-007", _dimension(field.key) and value == 0),
            ("NUM-008", _dimension(field.key) and value < 0), ("MEC-002", field.key == "抗拉强度" and value == 0),
            ("MEC-003", field.key == "屈服强度" and value == 0), ("MEC-007", field.key == "延伸率" and value < 0),
            ("MEC-008", field.key == "延伸率" and value > 100), ("MEC-012", field.key == "硬度" and value < 0),
            ("CHEM-001", field.key in ELEMENTS and value < 0), ("CHEM-002", field.key in ELEMENTS and field.unit == "%" and value > 100),
        ]
        for code, hit in cases:
            if hit and code in enabled:
                output.append(_field(code, doc, field, TITLE[code], "数值边界规则未通过", f"解析值 {value:g} 触发规则"))
        for code in ("INF-015", "UNIT-001"):
            if code in enabled and not field.unit and field.key in {"抗拉强度", "屈服强度", "延伸率", "硬度", *ELEMENTS}:
                output.append(_field(code, doc, field, TITLE[code], "关键测量结果必须带单位", "数值存在但单位为空"))
        if "CHEM-004" in enabled and field.key in ELEMENTS and not field.unit:
            output.append(_field("CHEM-004", doc, field, TITLE["CHEM-004"], "化学成分应声明 % 或 ppm", "元素值存在但单位为空"))
    if "MEC-001" in enabled and by_key["抗拉强度"] and by_key["屈服强度"]:
        tensile, yield_ = by_key["抗拉强度"][0], by_key["屈服强度"][0]
        if float(tensile.value) < float(yield_.value):
            output.append(_field("MEC-001", doc, tensile, TITLE["MEC-001"], f"应不小于屈服强度 {yield_.raw}",
                                 f"{tensile.raw} < {yield_.raw}", evidence=[_fev(doc, tensile), _fev(doc, yield_)]))
    for key, fields in by_key.items():
        units = {f.unit.casefold() for f in fields if f.unit}
        if "UNIT-002" in enabled and len(units) > 1:
            output.append(_field("UNIT-002", doc, fields[0], TITLE["UNIT-002"], "同一字段单位应一致", f"发现单位 {sorted(units)}",
                                 evidence=[_fev(doc, f) for f in fields[:4]]))
        unit_pairs = [
            ("UNIT-003", ({"mpa", "pa", "gpa"} & units) and len({"mpa", "pa", "gpa"} & units) > 1),
            ("UNIT-004", "ksi" in units and bool({"mpa", "pa", "gpa"} & units)),
            ("UNIT-005", bool({"inch", "in", "in."} & units) and bool({"mm", "cm"} & units)),
            ("UNIT-008", "%" in units and "ppm" in units),
        ]
        for code, hit in unit_pairs:
            if code in enabled and hit:
                output.append(_field(code, doc, fields[0], TITLE[code], "同一字段使用单位需先统一换算并核实数量级",
                                     f"检测到疑似混用：{sorted(units)}", evidence=[_fev(doc, f) for f in fields[:4]]))
        numeric_values = [abs(float(f.value)) for f in fields if isinstance(f.value, (int, float)) and float(f.value) != 0]
        if "NUM-011" in enabled and len(numeric_values) >= 2 and max(numeric_values) / min(numeric_values) >= 1000:
            output.append(_field("NUM-011", doc, fields[0], TITLE["NUM-011"], "同字段数值跨越三个以上数量级，疑似小数点或单位错误",
                                 f"最大值/最小值={max(numeric_values)/min(numeric_values):g}", evidence=[_fev(doc, f) for f in fields[:4]]))
    for element in ELEMENTS:
        fields = by_key[element]
        values = {round(float(f.value), 9) for f in fields if isinstance(f.value, (int, float))}
        if "CHEM-006" in enabled and len(values) > 1:
            output.append(_field("CHEM-006", doc, fields[0], TITLE["CHEM-006"], "同一文档同一元素出现不同结果，请按 Heat No 核实",
                                 f"识别到 {sorted(values)}", evidence=[_fev(doc, f) for f in fields[:6]]))
    chemistry = [f for f in doc.fields if f.key in ELEMENTS]
    if "CHEM-007" in enabled and chemistry and not (by_key["heat_number"] or by_key["material_grade"]):
        output.append(_field("CHEM-007", doc, chemistry[0], TITLE["CHEM-007"], "成分表应能关联 Material 或 Heat No",
                             "存在成分数据但未提取到关联标识", evidence=[_fev(doc, f) for f in chemistry[:4]]))
    return output


def _date_rules(doc: GenericDocument, enabled: set[str]) -> list[Finding]:
    aliases = {"manufacturing": r"manufactur(?:ing|ed)?\s*date|生产日期|制造日期", "sampling": r"sampling\s*date|取样日期|采样日期",
               "test": r"test(?:ing)?\s*date|检测日期|试验日期", "issue": r"issue(?:d)?\s*date|report\s*date|签发日期|报告日期"}
    values = defaultdict(list)
    for page in doc.pages:
        for key, alias in aliases.items():
            for match in re.finditer(rf"(?i)(?:{alias})\s*[:：-]?\s*(\d{{4}}[-/.]\d{{1,2}}[-/.]\d{{1,2}})", page.text):
                try: parsed = datetime.strptime(match[1].replace("/", "-").replace(".", "-"), "%Y-%m-%d").date()
                except ValueError: continue
                values[key].append((parsed, page.page, _line(page.text, match.start())))
    output = []
    for code, later, earlier in (("DATE-001", "test", "manufacturing"), ("DATE-002", "issue", "test"), ("DATE-003", "test", "sampling")):
        if code in enabled and values[later] and values[earlier] and values[later][0][0] < values[earlier][0][0]:
            a, b = values[later][0], values[earlier][0]
            output.append(_make(code, doc, a[1], TITLE[code], f"{later}={a[0]}，{earlier}={b[0]}。", a[2], "日期顺序应合理",
                                f"{a[0]} < {b[0]}", "Major", evidence=[_ev(doc, a[1], a[2]), _ev(doc, b[1], b[2])]))
    if "DATE-005" in enabled:
        for rows in values.values():
            for value, page, source in rows:
                if value > date.today():
                    output.append(_make("DATE-005", doc, page, TITLE["DATE-005"], f"识别到未来日期 {value}。", source,
                                        f"日期不得晚于 {date.today()}", f"{value} > {date.today()}", "Review", .95))
    return output


def _result_rules(doc: GenericDocument, enabled: set[str]) -> list[Finding]:
    output, fails, passes = [], [], []
    overall_pass = bool(re.search(r"(?i)(?:overall|conclusion)[^\n]{0,40}(?:pass|accepted)|最终结论[^\n]{0,20}合格", doc.text))
    overall_fail = bool(re.search(r"(?i)(?:overall|conclusion)[^\n]{0,40}(?:fail|reject)|最终结论[^\n]{0,20}不合格", doc.text))
    for page in doc.pages:
        for line in page.text.splitlines():
            if not line.startswith("[TABLE_ROW]"): continue
            if re.search(r"(?i)\b(?:NG|FAIL(?:ED)?)\b|不合格", line): fails.append((page.page, line))
            elif re.search(r"(?i)\bPASS(?:ED)?\b|合格", line): passes.append((page.page, line))
            cells = [x.strip() for x in line.removeprefix("[TABLE_ROW]").split("||")]
            if len(cells) >= 4 and re.search(r"(?i)pass", cells[-1]):
                expected = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", cells[1])]
                actual = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", cells[-2])]
                if not actual and "RES-006" in enabled:
                    output.append(_make("RES-006", doc, page.page, TITLE["RES-006"], line, line, "Actual 不能为空", "Actual 为空但 PASS", "Major"))
                elif expected and actual:
                    if "RES-001" in enabled and re.search(r"(?i)min|>=|≥", cells[1]) and actual[0] < expected[-1]:
                        output.append(_make("RES-001", doc, page.page, TITLE["RES-001"], line, line, cells[1], "Actual < Min 且 PASS", "Major"))
                    if "RES-002" in enabled and re.search(r"(?i)max|<=|≤", cells[1]) and actual[0] > expected[-1]:
                        output.append(_make("RES-002", doc, page.page, TITLE["RES-002"], line, line, cells[1], "Actual > Max 且 PASS", "Major"))
    if fails and overall_pass and "RES-003" in enabled:
        output.append(_make("RES-003", doc, fails[0][0], TITLE["RES-003"], fails[0][1], fails[0][1], "子项 FAIL 时总结果不得 PASS", "结论冲突", "Major", evidence=[_ev(doc, *x) for x in fails[:3]]))
    if passes and not fails and overall_fail and "RES-004" in enabled:
        output.append(_make("RES-004", doc, passes[0][0], TITLE["RES-004"], passes[0][1], passes[0][1], "全部子项 PASS 时总结果不应 FAIL", "结论冲突", "Review"))
    if fails and "RES-005" in enabled and re.search(r"(?i)\baccepted\b", doc.text):
        output.append(_make("RES-005", doc, fails[0][0], TITLE["RES-005"], fails[0][1], fails[0][1], "正文应与表格一致", "FAIL 与 ACCEPTED 冲突", "Major"))
    return output


def _consistency_rules(docs: list[GenericDocument], enabled: set[str]) -> list[Finding]:
    mapping = {"material_grade": "CON-001", "heat_number": "CON-002", "batch_number": "CON-003", "report_number": "CON-004", "厚度": "CON-006", "specification": "CON-007", "standard_number": "CON-010"}
    grouped = defaultdict(list)
    for doc in docs:
        for field in doc.fields:
            if field.key in mapping: grouped[field.key].append((doc, field))
    output = []
    for key, rows in grouped.items():
        code = mapping[key]
        if code in enabled and len({_norm(str(f.value or f.raw)) for _, f in rows}) > 1:
            doc, first = rows[0]
            actual = "；".join(f"{d.filename} 第{f.page}页: {f.raw}" for d, f in rows[:6])
            output.append(_field(code, doc, first, TITLE[code], "同一批次字段应一致", "归一化后存在多个值",
                                 evidence=[_fev(d, f) for d, f in rows[:6]], actual=actual))
    return output


def _field(code: str, doc: GenericDocument, field: ExtractedItem, description: str, requirement: str, logic: str,
           evidence: list[dict] | None = None, actual: str | None = None) -> Finding:
    return _make(code, doc, field.page, TITLE.get(code, code), description, actual or field.raw, requirement, logic,
                 "Major" if code.startswith(("MEC-", "RES-", "CON-")) else "Review", .96,
                 evidence=evidence or [_fev(doc, field)], source_text=field.source_text)


def _make(code: str, doc: GenericDocument, page: int, item: str, description: str, actual: str, requirement: str,
          logic: str, severity: str, confidence: float = 1, evidence: list[dict] | None = None, source_text: str = "") -> Finding:
    metadata = {"origin": "generic_rule", "check_id": code, "result": "存疑" if severity == "Review" else "不合格",
                "evidence": evidence or [_ev(doc, page, source_text or actual)]}
    return Finding("待人工确认" if severity == "Review" else "规则不符合", severity, item, description, actual, requirement,
                   doc.filename, page, source_text or actual, logic=logic, confidence=confidence, metadata=metadata,
                   rule_code=code, rule_version=1, document_type=doc.document_type,
                   extraction_confidence=doc.type_confidence, decision_confidence=confidence)


def _ev(doc: GenericDocument, page: int, source: str) -> dict:
    return {"document_id": doc.document_id, "file": doc.filename, "page": page, "source_text": source, "evidence_type": "source", "matched": False}


def _fev(doc: GenericDocument, field: ExtractedItem) -> dict:
    return _ev(doc, field.page, field.source_text or field.raw)


def _line(text: str, pos: int) -> str:
    start, end = text.rfind("\n", 0, pos) + 1, text.find("\n", pos)
    return text[start:end if end >= 0 else len(text)].strip()[:500]


def _norm(value: str) -> str:
    return re.sub(r"[^0-9a-z\u3400-\u9fff]+", "", value.casefold())


def _dimension(value: str) -> bool:
    return any(x in value.casefold() for x in ("厚度", "尺寸", "长度", "直径", "width", "height", "diameter", "thickness"))


def _dedupe(items: list[Finding]) -> list[Finding]:
    seen, output = set(), []
    for item in items:
        key = (item.rule_code, item.source_file, item.source_page, _norm(item.actual))
        if key not in seen: seen.add(key); output.append(item)
    return output

from __future__ import annotations

import re

from app.models import ExtractedItem, PageText, Requirement


LABELS: dict[str, list[str]] = {
    "material_grade": ["材料牌号", "材质", "Material Grade", "Grade"],
    "heat_number": ["炉号", "Heat No", "Heat Number"],
    "batch_number": ["批次号", "Batch No", "Lot No", "批号"],
    "standard_number": ["标准号", "执行标准", "Standard"],
    "po_number": ["PO号", "采购订单号", "PO No"],
    "drawing_number": ["图号", "Drawing No"],
    "product_model": ["型号", "Model"],
}

TEST_ALIASES: dict[str, list[str]] = {
    "抗拉强度": ["抗拉强度", "Rm", "Tensile Strength"],
    "屈服强度": ["屈服强度", "Rp0.2", "Yield Strength"],
    "延伸率": ["延伸率", "Elongation", "A%"],
    "硬度": ["硬度", "Hardness"],
    "冲击功": ["冲击功", "冲击试验", "冲击", "Impact Energy", "Impact Test", "KV2"],
    "厚度": ["厚度", "Thickness"],
    "C": ["C", "碳"], "Si": ["Si", "硅"], "Mn": ["Mn", "锰"],
    "P": ["P", "磷"], "S": ["S", "硫"], "Cr": ["Cr", "铬"],
    "Ni": ["Ni", "镍"], "Mo": ["Mo", "钼"],
}

UNIT = r"(?:GPa|MPa|N/mm(?:²|2)|Pa|mm|cm|μm|um|%|ppm|°C|℃|J|HV|HBW)?"
NUMBER = r"[-+]?\d+(?:\.\d+)?"


def extract_items(pages: list[PageText]) -> list[ExtractedItem]:
    results: list[ExtractedItem] = []
    seen: set[tuple[str, str, int]] = set()
    for page in pages:
        for key, labels in LABELS.items():
            label_pattern = "|".join(re.escape(label) for label in sorted(labels, key=len, reverse=True))
            pattern = rf"(?i)(?:{label_pattern})\s*(?:No\.?|编号)?\s*[:：=]?\s*([A-Za-z0-9][A-Za-z0-9./_-]{{1,30}})"
            for match in re.finditer(pattern, page.text):
                raw = match.group(1).strip().rstrip(".,;；")
                _append(results, seen, ExtractedItem(key, raw, raw, page=page.page,
                        source_text=_line(page.text, match.start()), category="identity"))
        for item, aliases in TEST_ALIASES.items():
            alias_pattern = "|".join(re.escape(alias) for alias in sorted(aliases, key=len, reverse=True))
            pattern = rf"(?im)(?<![\w])(?:{alias_pattern})(?![\w])\s*(?:[:：=]|实测(?:值)?\s*[:：]?)?\s*({NUMBER})\s*({UNIT})"
            for match in re.finditer(pattern, page.text):
                raw = (match.group(1) + (f" {match.group(2)}" if match.group(2) else "")).strip()
                _append(results, seen, ExtractedItem(item, raw, float(match.group(1)), match.group(2) or "",
                        page.page, _line(page.text, match.start()), "measurement"))
    return results


def extract_requirements(pages: list[PageText], source_file: str = "") -> list[Requirement]:
    requirements: list[Requirement] = []
    seen: set[tuple[str, str, str, int]] = set()
    for page in pages:
        for item, aliases in TEST_ALIASES.items():
            alias_pattern = "|".join(re.escape(alias) for alias in sorted(aliases, key=len, reverse=True))
            tolerance = rf"(?i)(?:{alias_pattern})\s*(?:[:：=为]|应为|要求)?\s*({NUMBER})\s*±\s*({NUMBER})\s*({UNIT})"
            ranges = rf"(?i)(?:{alias_pattern})\s*(?:[:：=为]|应为|要求)?\s*({NUMBER})\s*(?:~|～|至|—|-)\s*({NUMBER})\s*({UNIT})"
            comparison = rf"(?i)(?:{alias_pattern})\s*(?:[:：=为]|要求)?\s*(>=|≥|<=|≤|>|<|=|不得低于|不低于|至少|不得高于|不高于|最大(?:值)?|最小(?:值)?)\s*({NUMBER})(?!(?:[\d.]|\s*(?:~|～|至|—|-|±)))\s*({UNIT})"
            for match in re.finditer(tolerance, page.text):
                center, delta = float(match.group(1)), float(match.group(2))
                req = Requirement(item, "range", center - delta, center + delta, match.group(3) or "",
                                  match.group(0), source_file, page.page, _clause(page.text, match.start()))
                _add_requirement(requirements, seen, req)
            for match in re.finditer(ranges, page.text):
                req = Requirement(item, "range", float(match.group(1)), float(match.group(2)), match.group(3) or "",
                                  match.group(0), source_file, page.page, _clause(page.text, match.start()))
                _add_requirement(requirements, seen, req)
            for match in re.finditer(comparison, page.text):
                operator = _normalize_operator(match.group(1))
                req = Requirement(item, operator, float(match.group(2)), None, match.group(3) or "",
                                  match.group(0), source_file, page.page, _clause(page.text, match.start()))
                _add_requirement(requirements, seen, req)
        for match in re.finditer(r"(?im)(?:材料牌号|材质|Material Grade)\s*(?:[:：=为]|应为|要求)?\s*([A-Z][A-Z0-9.-]{2,20})", page.text):
            req = Requirement("material_grade", "=", match.group(1), raw=match.group(0), source_file=source_file,
                              source_page=page.page, clause=_clause(page.text, match.start()))
            _add_requirement(requirements, seen, req)
        required_pattern = "|".join(re.escape(alias) for aliases in TEST_ALIASES.values() for alias in aliases)
        for match in re.finditer(rf"(?im)(?:必须|应|需)(?:进行|提供|检验|检测)?[^。；\n]{{0,20}}?({required_pattern})(?:试验|检验|检测)?", page.text):
            canonical = _canonical_test(match.group(1))
            req = Requirement(canonical, "exists", None, raw=match.group(0), source_file=source_file,
                              source_page=page.page, clause=_clause(page.text, match.start()), required=True)
            _add_requirement(requirements, seen, req)
    return requirements


def _normalize_operator(value: str) -> str:
    if value in {">=", "≥", "不得低于", "不低于", "至少", "最小", "最小值"}: return ">="
    if value in {"<=", "≤", "不得高于", "不高于", "最大", "最大值"}: return "<="
    return value


def _canonical_test(alias: str) -> str:
    lowered = alias.casefold()
    for item, aliases in TEST_ALIASES.items():
        if any(lowered == candidate.casefold() for candidate in aliases): return item
    return alias


def _append(results: list[ExtractedItem], seen: set[tuple[str, str, int]], item: ExtractedItem) -> None:
    key = (item.key, item.raw.casefold(), item.page)
    if key not in seen:
        seen.add(key); results.append(item)


def _add_requirement(results: list[Requirement], seen: set[tuple[str, str, str, int]], item: Requirement) -> None:
    key = (item.item, item.operator, str(item.value), item.source_page)
    if key not in seen:
        seen.add(key); results.append(item)


def _line(text: str, position: int) -> str:
    start, end = text.rfind("\n", 0, position) + 1, text.find("\n", position)
    return text[start: end if end >= 0 else len(text)].strip()[:500]


def _clause(text: str, position: int) -> str:
    line = _line(text, position)
    match = re.match(r"\s*((?:\d+\.)+\d+|\d+\.?)\s*", line)
    return match.group(1).rstrip(".") if match else ""

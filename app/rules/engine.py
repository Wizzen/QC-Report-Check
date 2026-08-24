from __future__ import annotations

import math
import re
from collections import defaultdict

from app.models import ExtractedItem, Finding, Requirement


UNIT_FACTORS: dict[str, tuple[str, float]] = {
    "pa": ("pressure", 1.0), "mpa": ("pressure", 1_000_000.0), "gpa": ("pressure", 1_000_000_000.0),
    "n/mm²": ("pressure", 1_000_000.0), "n/mm2": ("pressure", 1_000_000.0),
    "mm": ("length", 0.001), "cm": ("length", 0.01), "m": ("length", 1.0),
    "μm": ("length", 0.000001), "um": ("length", 0.000001), "%": ("ratio", 0.01), "ppm": ("ratio", 0.000001),
    "°c": ("temperature", 1.0), "℃": ("temperature", 1.0), "j": ("energy", 1.0),
}


def normalize_identifier(value: str) -> str:
    return re.sub(r"[\s._/-]+", "", value).casefold()


def convert_unit(value: float, source: str, target: str) -> float:
    source_key, target_key = source.strip().casefold(), target.strip().casefold()
    if not source_key or not target_key or source_key == target_key: return value
    if source_key not in UNIT_FACTORS or target_key not in UNIT_FACTORS:
        raise ValueError(f"不支持单位转换：{source} → {target}")
    source_dimension, source_factor = UNIT_FACTORS[source_key]
    target_dimension, target_factor = UNIT_FACTORS[target_key]
    if source_dimension != target_dimension: raise ValueError(f"单位维度不一致：{source} → {target}")
    return value * source_factor / target_factor


def compare_value(actual: float, operator: str, expected: float, upper: float | None = None) -> bool:
    if operator == ">=": return actual >= expected
    if operator == ">": return actual > expected
    if operator == "<=": return actual <= expected
    if operator == "<": return actual < expected
    if operator == "=": return math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12)
    if operator == "range" and upper is not None: return expected <= actual <= upper
    raise ValueError(f"不支持的比较符：{operator}")


class AuditEngine:
    def audit(self, documents: dict[str, list[ExtractedItem]], requirements: list[Requirement]) -> list[Finding]:
        findings: list[Finding] = []
        all_items = [(filename, item) for filename, items in documents.items() for item in items]
        by_key: dict[str, list[tuple[str, ExtractedItem]]] = defaultdict(list)
        for filename, item in all_items: by_key[item.key].append((filename, item))
        findings.extend(self._inconsistencies(by_key))
        for requirement in requirements:
            candidates = by_key.get(requirement.item, [])
            if requirement.operator == "exists":
                if not candidates:
                    findings.append(self._missing(requirement))
                continue
            if not candidates:
                continue
            for filename, actual in candidates:
                finding = self._compare(filename, actual, requirement)
                if finding: findings.append(finding)
        return _dedupe(findings)

    def _compare(self, filename: str, actual: ExtractedItem, requirement: Requirement) -> Finding | None:
        if isinstance(requirement.value, str) or isinstance(actual.value, str):
            if normalize_identifier(str(actual.value)) == normalize_identifier(str(requirement.value)): return None
            severity = "Critical" if requirement.item == "material_grade" else "Major"
            return Finding("材料不符合" if requirement.item == "material_grade" else "数据不一致", severity,
                requirement.item, f"{requirement.item}与审核要求不一致", actual.raw, requirement.raw,
                filename, actual.page, actual.source_text, requirement.source_file, requirement.source_page,
                requirement.clause, f"{actual.raw} ≠ {requirement.value}")
        try:
            value = convert_unit(float(actual.value), actual.unit, requirement.unit)
            passed = compare_value(value, requirement.operator, float(requirement.value), requirement.upper_value)
        except (TypeError, ValueError) as exc:
            return Finding("待人工确认", "Review", requirement.item, f"无法可靠比较：{exc}", actual.raw,
                requirement.raw, filename, actual.page, actual.source_text, requirement.source_file,
                requirement.source_page, requirement.clause, "单位或数据格式需要人工确认", confidence=0.5)
        if passed: return None
        expected = f"{requirement.value}~{requirement.upper_value} {requirement.unit}" if requirement.operator == "range" else f"{requirement.operator} {requirement.value} {requirement.unit}"
        logic = f"{value:g} {requirement.unit} 不满足 {expected}".strip()
        return Finding("数值不符合", "Major", requirement.item, f"{requirement.item}实测值不满足要求",
            actual.raw, requirement.raw, filename, actual.page, actual.source_text, requirement.source_file,
            requirement.source_page, requirement.clause, logic)

    def _missing(self, requirement: Requirement) -> Finding:
        return Finding("检验项目缺失", "Major", requirement.item, f"未找到必检项目：{requirement.item}", "未提供",
            requirement.raw, standard_file=requirement.source_file, standard_page=requirement.source_page,
            standard_clause=requirement.clause, logic="必检项目集合中存在，但供应商文件提取结果中不存在")

    def _inconsistencies(self, by_key: dict[str, list[tuple[str, ExtractedItem]]]) -> list[Finding]:
        output: list[Finding] = []
        categories = {"heat_number": ("炉号不一致", "Major"), "batch_number": ("批次号不一致", "Major"),
                      "material_grade": ("材料不符合", "Critical"), "po_number": ("数据不一致", "Major"),
                      "drawing_number": ("图号不一致", "Major"), "product_model": ("型号不一致", "Major")}
        for key, (category, severity) in categories.items():
            values = by_key.get(key, [])
            distinct = {normalize_identifier(str(item.value)) for _, item in values}
            files = {filename for filename, _ in values}
            if len(distinct) > 1 and len(files) > 1:
                evidence = "; ".join(f"{filename}: {item.raw}" for filename, item in values)
                first_file, first = values[0]
                output.append(Finding(category, severity, key, f"不同供应商文件中的{key}不一致", evidence,
                                      "文件间应保持一致", first_file, first.page, first.source_text,
                                      logic="归一化后标识符不相等"))
        return output


def _dedupe(items: list[Finding]) -> list[Finding]:
    seen: set[tuple[str, str, str, str]] = set(); result = []
    for item in items:
        key = (item.category, item.item, item.source_file, item.actual)
        if key not in seen: seen.add(key); result.append(item)
    return result


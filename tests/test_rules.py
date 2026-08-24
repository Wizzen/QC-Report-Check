from __future__ import annotations

import pytest

from app.models import ExtractedItem, Requirement
from app.rules import AuditEngine, compare_value, convert_unit


def test_numeric_comparisons() -> None:
    assert compare_value(470, ">=", 470)
    assert not compare_value(455, ">=", 470)
    assert compare_value(0.035, "<=", 0.035)


def test_range_and_tolerance() -> None:
    assert compare_value(9.8, "range", 9.8, 10.2)
    assert compare_value(10.2, "range", 9.8, 10.2)
    assert not compare_value(10.21, "range", 9.8, 10.2)


def test_unit_conversion() -> None:
    assert convert_unit(470, "N/mm²", "MPa") == pytest.approx(470)
    assert convert_unit(0.47, "GPa", "MPa") == pytest.approx(470)
    assert convert_unit(10, "mm", "cm") == pytest.approx(1)
    assert convert_unit(0.0035, "%", "ppm") == pytest.approx(35)


def test_material_match_and_mismatch() -> None:
    engine = AuditEngine()
    requirement = Requirement("material_grade", "=", "Q355B", raw="材料牌号 Q355B", source_file="协议.pdf")
    assert engine.audit({"证书.pdf": [ExtractedItem("material_grade", "Q355B", "Q355B")]}, [requirement]) == []
    findings = engine.audit({"证书.pdf": [ExtractedItem("material_grade", "Q235B", "Q235B")]}, [requirement])
    assert findings[0].severity == "Critical"


def test_heat_number_consistency() -> None:
    documents = {
        "材质证明.pdf": [ExtractedItem("heat_number", "H240812", "H240812")],
        "检测报告.pdf": [ExtractedItem("heat_number", "H240821", "H240821")],
    }
    findings = AuditEngine().audit(documents, [])
    assert any(item.category == "炉号不一致" for item in findings)


def test_required_item_missing() -> None:
    required = [Requirement("冲击功", "exists", None, raw="必须进行冲击试验", required=True)]
    findings = AuditEngine().audit({"报告.pdf": [ExtractedItem("抗拉强度", "500 MPa", 500, "MPa")]}, required)
    assert findings[0].category == "检验项目缺失"


def test_numeric_rule_uses_program_logic() -> None:
    requirement = Requirement("抗拉强度", ">=", 470, unit="MPa", raw="Rm ≥ 470 MPa", source_file="协议.pdf", source_page=12, clause="5.3.2")
    actual = ExtractedItem("抗拉强度", "455 MPa", 455, "MPa", 3, "Tensile Strength: 455 MPa", "measurement")
    finding = AuditEngine().audit({"检测报告.pdf": [actual]}, [requirement])[0]
    assert finding.category == "数值不符合"
    assert "455" in finding.logic and "470" in finding.logic


from pathlib import Path

import pytest

from app.auditing.expert_review import (
    ExpertDocument,
    deterministic_fastener_audit,
    extract_supplier_names,
    findings_from_llm,
    rule_evaluation_from_llm,
    supplier_names_from_llm,
)
from app.extractors import extract_filename_items
from app.models import PageText


def test_fastener_table_detects_partial_pass_and_missing_result() -> None:
    page = PageText(1, """
QUALITY CERTIFICATE
CHEMICAL COMPOSITION(%)
Mechanical Properties
Dimensions Of SPEC
[TABLE_ROW] Head Height || 10.18-9.82 || 9.91-10.00 || 20 || 19
[TABLE_ROW] Thread Length || min 38. || 40.00-40.00 || 20 || 17
[TABLE_ROW] HV(2)>=HV(1)-30 || G 0.015max || 6 || 6
""")
    document = ExpertDocument("Q0045 5305-859240(MTR).pdf", Path("sample.pdf"), [page])

    findings = deterministic_fastener_audit([document])

    assert any(item.item == "Head Height" and "Pass(19) < Sample(20)" in item.logic for item in findings)
    assert any(item.item == "Thread Length" and item.severity == "Major" for item in findings)
    assert any(item.item.startswith("HV(2)") and item.severity == "Review" for item in findings)
    assert any(item.item == "A1 文件完整性" and "COC" in item.description for item in findings)


def test_llm_findings_require_exact_page_evidence() -> None:
    line = "[TABLE_ROW] Head Height || 10.18-9.82 || 9.91-10.00 || 20 || 19"
    document = ExpertDocument("MTR.pdf", Path("sample.pdf"), [PageText(1, line)])
    payload = {"findings": [
        {"check_id": "C5", "result": "不合格", "item": "Head Height", "source_file": "MTR.pdf",
         "page": 1, "evidence": line, "description": "通过数少于抽检数"},
        {"check_id": "C5", "result": "不合格", "item": "Fake", "source_file": "MTR.pdf",
         "page": 1, "evidence": "原页并不存在的证据", "description": "模型臆测"},
    ]}

    findings = findings_from_llm(payload, [document])

    assert [item.item for item in findings] == ["Head Height"]


def test_independent_rule_accepts_pass_without_creating_a_finding() -> None:
    document = ExpertDocument("MTR.pdf", Path("sample.pdf"), [PageText(1, "Heat No: H123")])

    evaluation, finding = rule_evaluation_from_llm(
        {"result": "合格", "conclusion": "炉号存在", "source_file": "MTR.pdf", "page": 1,
         "evidence": "Heat No: H123", "confidence": 0.94},
        "检查炉号是否存在", [document],
    )

    assert evaluation["status"] == "合格"
    assert finding is None


def test_independent_rule_rejects_fabricated_problem_evidence() -> None:
    document = ExpertDocument("MTR.pdf", Path("sample.pdf"), [PageText(1, "Heat No: H123")])

    with pytest.raises(ValueError, match="可核验"):
        rule_evaluation_from_llm(
            {"result": "不合格", "source_file": "MTR.pdf", "page": 1, "evidence": "Heat No: FAKE"},
            "检查炉号是否一致", [document],
        )


def test_independent_rule_accepts_missing_field_with_checked_scope() -> None:
    document = ExpertDocument("MTR.pdf", Path("sample.pdf"), [PageText(1, "Inspection Report")])

    evaluation, finding = rule_evaluation_from_llm(
        {"result": "存疑", "conclusion": "未找到报告编号", "evidence_type": "absence",
         "checked_scope": "已检查 MTR.pdf 第1页", "confidence": 0.8},
        "检查报告编号", [document],
    )

    assert evaluation["evidence"] == "已检查 MTR.pdf 第1页"
    assert finding and finding.severity == "Review"


def test_wdc_is_recovered_from_filename() -> None:
    items = extract_filename_items("Q0045 5305-859240(MTR).pdf")

    assert [(item.key, item.value) for item in items] == [("wdc_number", "5305859240")]


def test_supplier_names_are_extracted_and_buyer_is_excluded() -> None:
    document = ExpertDocument("MTR.pdf", Path("sample.pdf"), [PageText(1, """
Customer: CUSTOMER COMPANY
Manufacturer: NINGBO JINDING FASTENING PIECE CO.,LTD
Mill: HENAN JIYUAN IRON&STEEL CO.,LTD
""")])

    assert extract_supplier_names(document) == [
        "NINGBO JINDING FASTENING PIECE CO.,LTD",
        "HENAN JIYUAN IRON&STEEL CO.,LTD",
    ]


def test_llm_supplier_names_must_exist_in_source() -> None:
    document = ExpertDocument("MTR.pdf", Path("sample.pdf"), [
        PageText(1, "Manufacturer: NINGBO JINDING FASTENING PIECE CO.,LTD")
    ])
    payload = {"documents": [{"source_file": "MTR.pdf", "supplier_names": [
        "NINGBO JINDING FASTENING PIECE CO.,LTD", "HALLUCINATED COMPANY LIMITED"
    ]}]}

    assert supplier_names_from_llm(payload, [document]) == {
        "MTR.pdf": ["NINGBO JINDING FASTENING PIECE CO.,LTD"]
    }

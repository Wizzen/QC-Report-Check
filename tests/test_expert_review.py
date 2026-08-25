from pathlib import Path

from app.auditing.expert_review import (
    ExpertDocument,
    deterministic_fastener_audit,
    findings_from_llm,
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


def test_wdc_is_recovered_from_filename() -> None:
    items = extract_filename_items("Q0045 5305-859240(MTR).pdf")

    assert [(item.key, item.value) for item in items] == [("wdc_number", "5305859240")]

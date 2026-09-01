from pathlib import Path

from app.auditing.expert_review import ExpertDocument, rule_evaluation_from_llm
from app.auditing.v2_service import _merge_ocr_text
from app.database import ReviewDatabase
from app.models import PageText


def _document() -> ExpertDocument:
    return ExpertDocument("MTR.pdf", Path("MTR.pdf"), [PageText(1, "Tensile strength 480 MPa")])


def test_confidence_boundary_and_unlocated_low_confidence() -> None:
    payload = {"result": "不合格", "source_file": "MTR.pdf", "page": 1,
               "evidence": "Tensile strength 480 MPa", "confidence": 0.69}
    evaluation, finding = rule_evaluation_from_llm(payload, "抗拉强度", [_document()], 0.70)
    assert evaluation["status"] == "存疑"
    assert finding and finding.severity == "Review"
    assert "confidence_below_threshold" in finding.metadata["downgrade_reasons"]

    payload["confidence"] = 0.70
    evaluation, finding = rule_evaluation_from_llm(payload, "抗拉强度", [_document()], 0.70)
    assert evaluation["status"] == "不合格"
    assert finding and finding.severity == "Major"

    evaluation, finding = rule_evaluation_from_llm(
        {"result": "存疑", "source_file": "MTR.pdf", "page": 1, "evidence": "不存在", "confidence": 0.5},
        "抗拉强度", [_document()], 0.70,
    )
    assert evaluation["status"] == "存疑"
    assert finding is None


def test_deterministic_pass_and_feedback_pattern_only_downgrade_llm() -> None:
    payload = {"result": "不合格", "source_file": "MTR.pdf", "page": 1,
               "evidence": "Tensile strength 480 MPa", "confidence": 0.95}
    evaluation, finding = rule_evaluation_from_llm(
        payload, "抗拉强度", [_document()], deterministic_result="合格",
    )
    assert evaluation["status"] == "存疑"
    assert finding and finding.severity == "Review"
    assert "deterministic_pass_conflict" in finding.metadata["downgrade_reasons"]

    evaluation, finding = rule_evaluation_from_llm(
        payload, "抗拉强度", [_document()], feedback_policy={"downgrade_llm_issue": True},
    )
    assert evaluation["status"] == "存疑"
    assert finding and finding.severity == "Review"


def test_learning_activation_undo_and_anonymous_purge(tmp_path: Path) -> None:
    db = ReviewDatabase(tmp_path / "review.db")
    batch_id = db.create_batch(None)
    finding_ids = []
    for index in range(5):
        finding_id = db.execute(
            """INSERT INTO findings(batch_id,category,severity,item,description,rule_code,document_type,created_at)
               VALUES(?,?,?,?,?,?,?,datetime('now'))""",
            (batch_id, "独立规则", "Major", "RULE", f"issue-{index}", "RULE", "MTR"),
        )
        finding_ids.append(finding_id)
        db.record_finding_feedback(
            finding_id, action="误报驳回", new_status="误报驳回", reason_code="阈值问题",
            service_fingerprint="model-a",
        )
    policy = db.feedback_policy_for(
        template_key="", rule_code="RULE", document_type="MTR", model_fingerprint="model-a",
    )
    assert policy["sample_count"] == 5
    assert policy["downgrade_llm_issue"] is True

    assert db.undo_last_feedback(finding_ids[-1]) is True
    assert db.feedback_policy_for(
        template_key="", rule_code="RULE", document_type="MTR", model_fingerprint="model-a",
    )["sample_count"] == 4

    db.soft_delete_batch(batch_id)
    db.purge_batch(batch_id, force=True, retain_learning=True)
    row = db.one("SELECT batch_fingerprint,finding_fingerprint FROM learning_feedback WHERE active=1 LIMIT 1")
    assert row == {"batch_fingerprint": "", "finding_fingerprint": ""}


def test_purge_can_delete_batch_learning_contribution(tmp_path: Path) -> None:
    db = ReviewDatabase(tmp_path / "review.db")
    batch_id = db.create_batch(None)
    finding_id = db.execute(
        """INSERT INTO findings(batch_id,category,severity,item,description,rule_code,document_type,created_at)
           VALUES(?,?,?,?,?,?,?,datetime('now'))""",
        (batch_id, "独立规则", "Major", "RULE", "issue", "RULE", "MTR"),
    )
    db.record_finding_feedback(finding_id, action="确认问题", new_status="人工确认")
    db.soft_delete_batch(batch_id)
    db.purge_batch(batch_id, force=True, retain_learning=False)
    assert db.one("SELECT COUNT(*) count FROM learning_feedback")["count"] == 0


def test_corrected_severity_is_restored_by_undo(tmp_path: Path) -> None:
    db = ReviewDatabase(tmp_path / "review.db")
    batch_id = db.create_batch(None)
    finding_id = db.execute(
        """INSERT INTO findings(batch_id,category,severity,item,description,created_at)
           VALUES(?,?,?,?,?,datetime('now'))""",
        (batch_id, "独立规则", "Major", "RULE", "issue"),
    )
    db.record_finding_feedback(
        finding_id, action="修正结论", new_status="待人工复核", corrected_status="待人工复核",
        corrected_severity="Review", correction="证据不足",
    )
    assert db.one("SELECT status,severity FROM findings WHERE id=?", (finding_id,)) == {
        "status": "待人工复核", "severity": "Review",
    }
    assert db.undo_last_feedback(finding_id)
    assert db.one("SELECT status,severity FROM findings WHERE id=?", (finding_id,)) == {
        "status": "AI发现", "severity": "Major",
    }


def test_ocr_pages_are_mapped_back_independently() -> None:
    pages = [PageText(1, "good text"), PageText(2, ""), PageText(3, "")]
    merged = _merge_ocr_text(pages, {2: "page two"}, [2, 3])
    assert "page two" in merged[1].text
    assert "[OCR_FAILED_PAGE]" in merged[2].text
    assert "page two" not in merged[2].text

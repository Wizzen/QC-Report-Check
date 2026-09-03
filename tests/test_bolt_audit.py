from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pymupdf
import pytest

from app.auditing.bolt_audit import coverage_for, document_wdcs, evidence_documents, scoped_tasks, seal_requirement, table_checks
from app.auditing.bolt_template import BOLT_ENGINE, BOLT_TEMPLATE_NAME, EXTRACTION_VERSION, bolt_rules
from app.auditing.expert_review import ExpertDocument, parse_template_tasks, rule_evaluation_from_llm
from app.auditing.signatures import LocalVision, aggregate_observations, inspect_document
from app.auditing.v2_service import ReviewService
from app.database import ReviewDatabase
from app.integrations import ConfigStore
from app.models import PageText
from app.parsers.document_parser import _extract_pdf_tables


def service(tmp_path):
    db = ReviewDatabase(tmp_path / 'test.db')
    return ReviewService(db, ConfigStore(db, tmp_path / 'key'), tmp_path/'uploads', tmp_path/'basis', tmp_path/'vectors')


def doc(name, text):
    return ExpertDocument(name, Path(name), [PageText(1, text)])


def test_structured_rules_survive_parsing_and_disabled_rules_do_not_dispatch():
    rules = bolt_rules()
    assert parse_template_tasks(json.dumps(rules)) == rules
    documents = [doc('5305-051854(MTR).pdf', 'Material Test Report')]
    types, wdcs, _ = coverage_for(documents)
    ids = {task.rule['rule_id'] for task in scoped_tasks(rules, documents, types, wdcs)}
    assert not {'B1', 'B4'} & ids
    assert 'A3.1' in ids


def test_template_user_edits_and_default_are_preserved(tmp_path):
    s = service(tmp_path)
    default = s.db.one('SELECT id FROM audit_templates WHERE is_default=1')['id']
    template = s.db.one('SELECT * FROM audit_templates WHERE name=?', (BOLT_TEMPLATE_NAME,))
    assert template['engine_binding'] == BOLT_ENGINE
    assert template['is_default'] == 0
    s.db.execute('UPDATE audit_templates SET review_instructions=?', ('user-owned',))
    service(tmp_path)
    assert {row['review_instructions'] for row in s.db.query('SELECT review_instructions FROM audit_templates')} == {'user-owned'}
    assert s.db.one('SELECT id FROM audit_templates WHERE is_default=1')['id'] == default


def test_scope_snapshot_is_separate_from_review_mode(tmp_path):
    s = service(tmp_path)
    tid = s.ensure_bolt_template()
    bid = s.db.create_batch(tid, 'deep', 1, 'single_document')
    batch = s.db.one('SELECT * FROM review_batches WHERE id=?', (bid,))
    assert batch['review_mode'] == 'deep'
    assert batch['audit_scope'] == 'single_document'
    assert json.loads(batch['template_snapshot'])['engine_binding'] == BOLT_ENGINE
    with pytest.raises(ValueError): s.db.create_batch(tid, audit_scope='arbitrary')


def test_wdc_extracts_candidates_not_whole_filename_and_lists_uncovered():
    report = doc(r'test\missing_stamp\Q0017 5305-091025(MTR).pdf', 'Material Test Report')
    coc = doc('COC.pdf', 'PART NUMBER 5305-091025\nPART NUMBER 5305-051854')
    assert document_wdcs(report) == {'5305091025'}
    types, wdcs, coverage = coverage_for([report, coc])
    assert coverage['uncovered_wdcs'] == ['5305051854']
    tasks = scoped_tasks(bolt_rules(), [report, coc], types, wdcs)
    assert all(task.wdc != '5305051854' for task in tasks if task.rule['scope'] == 'wdc')


def test_certificate_of_compliance_is_coc_without_filename_hint():
    types, _, _ = coverage_for([doc('certificate.pdf', 'Certificate of Compliance')])
    assert types['certificate.pdf'] == 'COC'


@pytest.mark.parametrize('state,confidence,anchored,expected', [
    ('present', .89, True, 'unknown'), ('present', .95, True, 'present'),
    ('absent', .98, False, 'unknown'), ('absent', .98, True, 'absent'),
    ('present', 2, True, 'unknown'), ('invalid', .99, True, 'unknown'),
])
def test_visual_confidence_and_region_guards(state, confidence, anchored, expected):
    client = SimpleNamespace(settings=SimpleNamespace(llm_base_url='http://127.0.0.1:8080/v1', llm_model='local'),
        generate_json=Mock(return_value={'state':state,'confidence':confidence}))
    vision = LocalVision(client)
    vision.available = True
    assert vision.inspect(b'fixture', 'signature', anchored)['state'] == expected


def test_cross_page_header_retained_without_shifting_empty_values():
    document = ExpertDocument('MTR.pdf', Path('MTR.pdf'), [
        PageText(1, '[TABLE_ROW] Item || Standard || Result'),
        PageText(2, '[TABLE_ROW] Length || 9-11 || 12\n[TABLE_ROW] Width || 9-11 || '),
    ])
    findings, checked, uncertain = table_checks(document)
    assert len(findings) == 1 and findings[0].source_page == 2
    assert checked == 1 and len(uncertain) == 1


def test_retry_has_independent_parse_and_preserves_original(tmp_path):
    s = service(tmp_path)
    path = tmp_path/'report.txt'
    path.write_text('Material Test Report', encoding='utf-8')
    batch = s.db.create_batch(s.ensure_bolt_template(), audit_scope='single_document')
    did = s.db.add_document(library='supplier',kind='supplier',original_name='report.txt',stored_path=str(path),sha256='test')
    s.db.attach_document(batch,did,'supplier',4)
    s.db.execute("UPDATE documents SET parse_status='completed',page_text='[]' WHERE id=?", (did,))
    original = s.db.one('SELECT * FROM documents WHERE id=?',(did,))
    retry = s.retry_review(batch)
    clone = s.db.one('SELECT d.* FROM documents d JOIN batch_documents bd ON d.id=bd.document_id WHERE bd.batch_id=?',(retry,))
    assert clone['id'] != did and clone['source_document_id'] == did
    assert clone['stored_path'] != str(path) and Path(clone['stored_path']).read_bytes() == path.read_bytes()
    assert clone['parse_status'] == 'pending'
    assert s.db.one('SELECT * FROM documents WHERE id=?',(did,)) == original
    assert s.db.one('SELECT audit_scope FROM review_batches WHERE id=?',(retry,))['audit_scope'] == 'single_document'


def test_new_engine_blocks_remote_ocr_even_if_global_setting_allows(tmp_path):
    s = service(tmp_path)
    s.db.execute("UPDATE service_config SET allow_remote=1,ocr_base_url='https://example.com/ocr' WHERE id=1")
    batch = s.db.create_batch(s.ensure_bolt_template())
    with patch.object(s, 'process_document') as parser:
        s.process_batch(batch)
    assert s.db.one('SELECT status FROM review_batches WHERE id=?',(batch,))['status'] == 'failed'
    parser.assert_not_called()


def test_manifest_unmatched_wdc_requires_review(tmp_path):
    s = service(tmp_path)
    tid = s.ensure_bolt_template()
    s.db.execute('UPDATE audit_templates SET required_items=? WHERE id=?',(json.dumps([bolt_rules()[0]]),tid))
    batch = s.db.create_batch(tid)
    for filename in ('5305-091025(MTR).pdf','5305-051854(COC).pdf'):
        did = s.db.add_document(library='supplier',kind='supplier',original_name=filename,stored_path='/unused.pdf',sha256=filename)
        s.db.execute("UPDATE documents SET parse_status='completed',page_text=?,extraction_fingerprint=? WHERE id=?",
                     (json.dumps([{'page':1,'text':'synthetic text'}]),EXTRACTION_VERSION,did))
        s.db.attach_document(batch,did,'supplier',4)
    with patch('app.auditing.bolt_audit.LLMClient.generation_concurrency_limit', return_value=1):
        s.process_batch(batch)
    result = s.db.one('SELECT status,conclusion FROM rule_evaluations WHERE batch_id=?',(batch,))
    assert result['status'] == '存疑' and '对应关系' in result['conclusion']


@pytest.mark.parametrize('confirmation,expected', [(True,'Major'),(False,'Review'),('true','Review'),('false','Review'),(RuntimeError('timeout'),'Review')])
def test_second_review_strict_boolean_and_export_state(tmp_path, confirmation, expected):
    s = service(tmp_path)
    tid = s.ensure_bolt_template()
    rule = {'rule_id':'TEST','text':'实测值与要求一致性','enabled':True,'evaluator':'llm','scope':'document'}
    s.db.execute('UPDATE audit_templates SET required_items=? WHERE id=?',(json.dumps([rule]),tid))
    batch = s.db.create_batch(tid, review_mode='deep')
    did = s.db.add_document(library='supplier',kind='supplier',original_name='MTR.pdf',stored_path='/unused.pdf',sha256='test')
    s.db.execute("UPDATE documents SET parse_status='completed',page_text=?,extraction_fingerprint=? WHERE id=?",
        (json.dumps([{'page':1,'text':'Standard: 9-11; Result: 12'}]),EXTRACTION_VERSION,did))
    s.db.attach_document(batch,did,'supplier',4)
    payload = {'r':'不合格','f':'MTR.pdf','p':1,'e':'Standard: 9-11; Result: 12','c':'实测值超限','q':.99}
    response = confirmation if isinstance(confirmation, Exception) else {'confirmed':confirmation}
    with patch('app.auditing.bolt_audit.LLMClient.generation_concurrency_limit', return_value=1), \
         patch('app.auditing.bolt_audit.LLMClient.generate_json', side_effect=[payload,response]) as generate:
        s.process_batch(batch)
    assert generate.call_args_list[0].kwargs['thinking'] is True
    assert generate.call_args_list[1].kwargs['thinking'] is False
    finding = s.db.one('SELECT * FROM findings WHERE batch_id=?',(batch,))
    assert finding['severity'] == expected
    ledger = s.db.one('SELECT * FROM rule_evaluations WHERE batch_id=?',(batch,))
    expected_status = '不合格' if expected == 'Major' else '存疑'
    assert ledger['status'] == json.loads(finding['metadata'])['result'] == expected_status
    assert ledger['evidence'] == finding['source_text']
    if expected == 'Review':
        assert '二次复核未确认' in ledger['conclusion']
        assert json.loads(ledger['metadata'])['downgrade_reasons'].count('second_review_not_confirmed') == 1


def test_low_confidence_unlocated_result_is_not_recreated_as_generic_finding(tmp_path):
    s = service(tmp_path)
    tid = s.ensure_bolt_template()
    rule = {'rule_id':'TEST','text':'出具单位','enabled':True,'evaluator':'llm','scope':'document'}
    s.db.execute('UPDATE audit_templates SET required_items=? WHERE id=?',(json.dumps([rule]),tid))
    batch = s.db.create_batch(tid)
    did = s.db.add_document(library='supplier',kind='supplier',original_name='MTR.pdf',stored_path='/unused.pdf',sha256='test')
    s.db.execute("UPDATE documents SET parse_status='completed',page_text=?,extraction_fingerprint=? WHERE id=?",
        (json.dumps([{'page':1,'text':'Material Test Report'}]),EXTRACTION_VERSION,did))
    s.db.attach_document(batch,did,'supplier',4)
    with patch('app.auditing.bolt_audit.LLMClient.generation_concurrency_limit', return_value=1), \
         patch('app.auditing.bolt_audit.LLMClient.generate_json', return_value={'r':'存疑','e':'not in source','q':.3}):
        s.process_batch(batch)
    assert not s.db.query('SELECT id FROM findings WHERE batch_id=?',(batch,))
    assert s.db.one('SELECT status FROM rule_evaluations WHERE batch_id=?',(batch,))['status'] == '存疑'


def test_table_empty_cells_and_column_mapping():
    page = SimpleNamespace(find_tables=lambda: SimpleNamespace(tables=[SimpleNamespace(extract=lambda: [
        ['Item', 'Standard', 'Result', 'Sample', 'Pass'], ['Length','9-11',None,'15','15']])]))
    text = _extract_pdf_tables(page)
    assert '9-11 ||  || 15 || 15' in text
    findings, checked, uncertain = table_checks(doc('MTR.pdf', text))
    assert not findings and checked == 0 and len(uncertain) == 1
    findings, checked, uncertain = table_checks(doc('MTR.pdf', text), samples=True)
    assert not findings and checked == 1 and not uncertain


def test_real_numeric_and_sample_failures_are_not_suppressed():
    text = '[TABLE_ROW] Item || Standard || Result || Sample || Pass\n[TABLE_ROW] Length || 9-11 || 12 || 15 || 14'
    assert table_checks(doc('MTR.pdf', text))[0][0].severity == 'Major'
    assert table_checks(doc('MTR.pdf', text), samples=True)[0][0].severity == 'Major'


def test_seal_requires_explicit_requirement_not_language_or_logo():
    assert seal_requirement([doc('MTR.pdf', '中文材料报告\nCompany logo\nStamp')], '') is None
    assert seal_requirement([doc('MTR.pdf', '无质量证明专用章无效')], '')
    assert seal_requirement([doc('COC.pdf', 'Certificate of compliance')], 'MTR must be stamped') is None
    assert seal_requirement([doc('MTR.pdf', 'Material test')], 'MTR must be stamped')


def test_budget_is_fair_and_does_not_lose_short_signature_page():
    documents = [doc('MTR.pdf', ('other material\n'*3000)), doc('COC.pdf', 'Signature\nVisible writing\nDate')]
    selected, coverage = evidence_documents(documents, 'signature', budget=8000)
    assert selected[1].pages[0].text == documents[1].pages[0].text
    assert len(coverage) == 2
    assert sum(len(p.text) for d in selected for p in d.pages) <= 8000


def test_probe_requires_pixel_answer_and_does_not_send_answer_in_prompt():
    client = SimpleNamespace(settings=SimpleNamespace(llm_base_url='http://127.0.0.1:8080/v1', llm_model='local'),
                             generate_json=Mock(return_value={'code':'ABC123'}))
    with patch('app.auditing.signatures.secrets.token_hex', return_value='abc123'):
        vision = LocalVision(client)
        assert vision.probe()
        args, kwargs = client.generate_json.call_args
        assert 'ABC123' not in args[0] and kwargs['images'][0].startswith(b'\x89PNG')
    client.generate_json.return_value = {'code':'wrong'}
    assert not LocalVision(client).probe()


def test_visual_failure_and_printed_name_do_not_become_missing_signature():
    unknown = [{'kind':'signature','state':'unknown','method':'local_vision','description':'无法区分打印姓名','confidence':.5}]
    assert aggregate_observations(unknown, 'signature')[0] == 'unknown'
    assert aggregate_observations([{'kind':'digital_signature','state':'unfilled','method':'pdf_widget'}], 'signature')[0] == 'unknown'
    assert aggregate_observations([{'kind':'digital_signature','state':'present','method':'pdf_widget'}], 'signature')[0] == 'present'
    assert aggregate_observations([{'kind':'signature','state':'absent','method':'local_vision'}], 'signature')[0] == 'absent'
    evaluation, finding = rule_evaluation_from_llm({'r':'不合格','f':'COC.pdf','p':1,'e':'Signature','q':.99},
        'COC签字', [doc('COC.pdf', 'Signature')])
    assert evaluation['status'] == '存疑'
    assert '不能证明' in finding.description


def test_conflicting_signature_regions_require_role_review():
    observations = [
        {'kind':'signature','state':'present','method':'local_vision','anchored':True},
        {'kind':'signature','state':'absent','method':'local_vision','anchored':True},
    ]
    state, evidence = aggregate_observations(observations, 'signature')
    assert state == 'unknown' and '角色' in evidence['description']


def test_pdf_is_unchanged_and_signature_region_inspected_despite_dense_text(tmp_path):
    path = tmp_path/'COC.pdf'
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((45,60), 'Certificate of Compliance ' * 4)
        page.insert_text((45,650), 'Representative Signature')
        page.draw_polyline([(50,620),(65,610),(59,630),(85,618)])
        widget = pymupdf.Widget()
        widget.field_type = pymupdf.PDF_WIDGET_TYPE_SIGNATURE
        widget.field_name = 'Signature1'
        widget.rect = pymupdf.Rect(40, 580, 250, 660)
        page.add_widget(widget)
        pdf.save(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    fake = SimpleNamespace(fingerprint='test', inspect=Mock(return_value={'state':'present','confidence':.99,'description':'visible signature'}))
    observations = inspect_document(ExpertDocument('COC.pdf', path, [PageText(1, 'long text '*100)]), fake)
    assert any(row['kind']=='signature' and row.get('anchored') for row in observations)
    assert any(row['kind']=='digital_signature' and row['state']=='unfilled' for row in observations)
    assert aggregate_observations(observations,'signature')[0] == 'present'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


@pytest.mark.parametrize('scope,expected', [('single_document', False), ('full_package', True)])
def test_isolated_validation_batch_manifest_and_old_records(tmp_path, scope, expected):
    s = service(tmp_path)
    tid = s.ensure_bolt_template()
    s.db.execute('UPDATE audit_templates SET required_items=? WHERE id=?',
                 (json.dumps([bolt_rules()[0]]),tid))
    batch = s.db.create_batch(tid, audit_scope=scope)
    did = s.db.add_document(library='supplier',kind='supplier',original_name='5305-091025(MTR).pdf',stored_path='/not-required.pdf',sha256='test')
    s.db.execute("UPDATE documents SET parse_status='completed',page_text=?,extraction_fingerprint=? WHERE id=?",
                 (json.dumps([{'page':1,'text':'Material Test Report'}]),EXTRACTION_VERSION,did))
    s.db.attach_document(batch,did,'supplier',4)
    with patch('app.auditing.bolt_audit.LLMClient.generation_concurrency_limit', return_value=1):
        s.process_batch(batch)
    assert s.db.one('SELECT status FROM review_batches WHERE id=?',(batch,))['status'] == 'completed'
    assert bool(s.db.query('SELECT id FROM findings WHERE batch_id=? AND severity="Major"',(batch,))) is expected
    before = s.db.query('SELECT * FROM findings WHERE batch_id=?',(batch,))
    s.process_batch(batch)
    assert s.db.query('SELECT * FROM findings WHERE batch_id=?',(batch,)) == before

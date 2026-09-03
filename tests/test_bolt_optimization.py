import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pymupdf
import pytest

from app.auditing.bolt_template import BOLT_TEMPLATE_NAME, EXTRACTION_VERSION, bolt_rules
from app.auditing.calibration import feedback_identity
from app.auditing.expert_review import ExpertDocument
from app.auditing.signatures import inspect_document
from app.auditing.v2_service import ReviewService
from app.database import ReviewDatabase
from app.integrations import ConfigStore


@pytest.fixture
def service(tmp_path):
    db = ReviewDatabase(tmp_path/'review.db')
    return ReviewService(db, ConfigStore(db,tmp_path/'key'),tmp_path/'uploads',tmp_path/'basis',tmp_path/'vectors')


def restart(service):
    return ReviewService(ReviewDatabase(service.db.path), service.config_store,
        service.uploads, service.standards, service.vector_root)


def setup_batch(service, rule, text='Standard: 9-11; Result: 12'):
    tid = service.ensure_bolt_template()
    service.db.execute('UPDATE audit_templates SET required_items=? WHERE id=?',(json.dumps([rule]),tid))
    batch = service.db.create_batch(tid)
    did = service.db.add_document(library='supplier',kind='supplier',original_name='MTR.pdf',stored_path='/unused.pdf',sha256='test')
    service.db.execute("UPDATE documents SET parse_status='completed',page_text=?,extraction_fingerprint=? WHERE id=?",
        (json.dumps([{'page':1,'text':text}]),EXTRACTION_VERSION,did))
    service.db.attach_document(batch,did,'supplier',4)
    row = service.db.one('SELECT * FROM review_batches WHERE id=?',(batch,))
    return batch, feedback_identity(tid,json.loads(row['template_snapshot']),rule,service.config_store.get())


def run(service, batch, payload=None):
    payload = payload or {'r':'不合格','f':'MTR.pdf','p':1,'e':'Standard: 9-11; Result: 12','q':.99,'c':'实测值超限'}
    with patch('app.auditing.bolt_audit.LLMClient.generation_concurrency_limit',return_value=1), \
         patch('app.auditing.bolt_audit.LLMClient.generate_json',side_effect=[payload,{'confirmed':True}]):
        service.process_batch(batch)


def seed_feedback(service, identity, rule, count, rejections, document_type='MTR'):
    old = service.db.create_batch(service.ensure_bolt_template())
    ids = []
    for i in range(count):
        fid = service.db.execute('''INSERT INTO findings(batch_id,category,severity,item,description,rule_code,
            document_type,metadata,created_at) VALUES(?,?,?,?,?,?,?,?,datetime('now'))''',
            (old,'test','Major',rule['text'],'synthetic',rule['rule_id'],document_type,
             json.dumps({'feedback_identity':identity})))
        action = '误报驳回' if i < rejections else '确认问题'
        service.db.record_finding_feedback(fid,action=action,new_status=action,reason_code='上下文不足',
            service_fingerprint='wrong-current-model',note='free-form note must not enter prompts')
        ids.append(fid)
    return old, ids


def test_seed_is_once_across_rename_and_delete(service):
    tid = service.ensure_bolt_template()
    historical = service.db.create_batch(tid)
    service.db.execute('UPDATE audit_templates SET name=? WHERE id=?',('Renamed template',tid))
    restarted = restart(service)
    assert restarted.ensure_bolt_template() == tid
    assert not service.db.one('SELECT id FROM audit_templates WHERE name=?',(BOLT_TEMPLATE_NAME,))
    service.db.delete_template(tid)
    restarted = restart(service)
    assert restarted.ensure_bolt_template() is None
    assert not service.db.one("SELECT id FROM audit_templates WHERE engine_binding='bolt-v1'")
    assert json.loads(service.db.one('SELECT template_snapshot FROM review_batches WHERE id=?',(historical,))['template_snapshot'])['name'] == BOLT_TEMPLATE_NAME


def test_upgrade_recognizes_renamed_seed_without_changing_content(service):
    tid = service.ensure_bolt_template()
    service.db.execute("DELETE FROM application_state WHERE key='bolt-template-seeded'")
    service.db.execute('UPDATE audit_templates SET name=?,review_instructions=? WHERE id=?',('User template','User instructions',tid))
    assert restart(service).ensure_bolt_template() == tid
    assert service.db.one('SELECT review_instructions FROM audit_templates WHERE id=?',(tid,))['review_instructions'] == 'User instructions'


@pytest.mark.parametrize('count,rejected,expected',[(4,4,'Major'),(5,3,'Major'),(5,4,'Review'),(5,5,'Review')])
def test_new_engine_applies_five_sample_eighty_percent_gate(service,count,rejected,expected):
    rule = {'rule_id':'CHECK','text':'数据对比','enabled':True,'evaluator':'llm','scope':'document'}
    batch, identity = setup_batch(service,rule)
    seed_feedback(service,identity,rule,count,rejected)
    run(service,batch)
    finding = service.db.one('SELECT * FROM findings WHERE batch_id=?',(batch,))
    assert finding['severity'] == expected
    assert json.loads(finding['metadata'])['feedback_policy']['sample_count'] == count
    saved = service.db.one('SELECT model_fingerprint FROM learning_feedback LIMIT 1')
    assert saved['model_fingerprint'] == identity['model_fingerprint']


def test_rule_model_ocr_and_parser_identity_isolated(service):
    rule = {'rule_id':'CHECK','text':'数据对比','criterion':'9-11'}
    snapshot = {'template_version':1,'review_instructions':'original'}
    settings = service.config_store.get()
    original = feedback_identity(1,snapshot,rule,settings)
    for altered in (replace(settings,llm_model='different'),replace(settings,ocr_backend='different'),
                    replace(settings,ocr_lang='en'),replace(settings,llm_base_url='http://127.0.0.1:8085/v1')):
        assert feedback_identity(1,snapshot,rule,altered) != original
    assert feedback_identity(1,snapshot,{**rule,'criterion':'9-12'},settings) != original
    with patch('app.auditing.calibration.EXTRACTION_VERSION','next-parser'):
        assert feedback_identity(1,snapshot,rule,settings) != original
    assert feedback_identity(1,{**snapshot,'name':'renamed'},rule,settings) == original


@pytest.mark.parametrize('change',['model','document','paused'])
def test_incompatible_or_paused_policy_does_not_downgrade(service,change):
    rule = {'rule_id':'CHECK','text':'数据对比','enabled':True,'evaluator':'llm','scope':'document'}
    batch, identity = setup_batch(service,rule)
    seed_feedback(service,identity,rule,5,5,document_type='COC' if change=='document' else 'MTR')
    if change=='model':
        service.db.execute("UPDATE service_config SET llm_model='different' WHERE id=1")
    if change=='paused':
        policy = service.db.feedback_policy_for(**identity,rule_code='CHECK',document_type='MTR')
        service.db.set_feedback_pattern_enabled(policy['pattern_key'],False)
    run(service,batch)
    assert service.db.one('SELECT severity FROM findings WHERE batch_id=?',(batch,))['severity'] == 'Major'


def test_feedback_never_suppresses_deterministic_failure(service):
    rule = next(r for r in bolt_rules() if r['evaluator']=='table_values')
    batch, identity = setup_batch(service,rule,'[TABLE_ROW] Item || Standard || Result\n[TABLE_ROW] Length || 9-11 || 12')
    seed_feedback(service,identity,rule,5,5)
    run(service,batch)
    finding = service.db.one('SELECT severity,metadata FROM findings WHERE batch_id=?',(batch,))
    assert finding['severity'] == 'Major'
    assert json.loads(finding['metadata'])['origin'] == 'deterministic'


def test_high_confirmation_only_prioritizes_review_and_notes_never_enter_prompt(service):
    rule = {'rule_id':'CHECK','text':'数据对比','enabled':True,'evaluator':'llm','scope':'document'}
    batch, identity = setup_batch(service,rule)
    seed_feedback(service,identity,rule,5,0)
    payload = {'r':'存疑','f':'MTR.pdf','p':1,'e':'Standard: 9-11; Result: 12','q':.8,'c':'需人工核验'}
    with patch('app.auditing.bolt_audit.LLMClient.generation_concurrency_limit',return_value=1), \
         patch('app.auditing.bolt_audit.LLMClient.generate_json',return_value=payload) as generate:
        service.process_batch(batch)
    finding = service.db.one('SELECT severity,metadata FROM findings WHERE batch_id=?',(batch,))
    assert finding['severity'] == 'Review'
    assert json.loads(finding['metadata'])['review_priority'] == 'high'
    assert 'free-form note' not in generate.call_args.args[0]


def test_old_new_engine_findings_without_identity_are_audit_only(service):
    batch, _ = setup_batch(service,{'rule_id':'CHECK','text':'数据对比'})
    fid = service.db.execute('''INSERT INTO findings(batch_id,category,severity,item,description,rule_code,created_at)
        VALUES(?,?,?,?,?,?,datetime('now'))''',(batch,'test','Major','CHECK','historical','CHECK'))
    service.db.record_finding_feedback(fid,action='误报驳回',reason_code='上下文不足',service_fingerprint='current-model')
    assert service.db.one('SELECT COUNT(*) n FROM review_feedback')['n'] == 1
    assert service.db.one('SELECT COUNT(*) n FROM learning_feedback')['n'] == 0


def test_repeated_clicks_count_once_and_remain_once_after_anonymous_purge(service):
    rule = {'rule_id':'CHECK','text':'数据对比','enabled':True,'evaluator':'llm','scope':'document'}
    _, identity = setup_batch(service,rule)
    old, ids = seed_feedback(service,identity,rule,1,1)
    for _ in range(5):
        service.db.record_finding_feedback(ids[0],action='误报驳回',reason_code='上下文不足')
    policy_args = {**identity,'rule_code':'CHECK','document_type':'MTR'}
    assert service.db.feedback_policy_for(**policy_args)['sample_count'] == 1
    assert service.db.learning_summary()['total'] == 1
    service.db.soft_delete_batch(old)
    service.db.purge_batch(old,force=True,retain_learning=True)
    assert service.db.feedback_policy_for(**policy_args)['sample_count'] == 1


def test_latest_feedback_wins_and_undo_restores_previous_sample(service):
    rule = {'rule_id':'CHECK','text':'数据对比'}
    _, identity = setup_batch(service,rule)
    _, ids = seed_feedback(service,identity,rule,1,1)
    service.db.record_finding_feedback(ids[0],action='确认问题')
    args = {**identity,'rule_code':'CHECK','document_type':'MTR'}
    assert service.db.feedback_policy_for(**args)['confirmation_rate'] == 1
    service.db.undo_last_feedback(ids[0])
    assert service.db.feedback_policy_for(**args)['rejection_rate'] == 1


@pytest.mark.parametrize('kinds',[{'signature'},{'stamp'},set()])
def test_pdf_only_renders_requested_visual_kind(kinds):
    page = Mock()
    page.widgets.return_value = []
    page.rect = pymupdf.Rect(0,0,600,800)
    page.get_text.return_value = [(10,200,200,220,'Signature')]
    page.get_pixmap.return_value.tobytes.return_value = b'png'
    vision = SimpleNamespace(fingerprint='fixture',inspect=Mock(return_value={'state':'absent','confidence':.99}))
    with patch('app.auditing.signatures.pymupdf.open') as reader:
        reader.return_value.__enter__.return_value = [page]
        observations = inspect_document(ExpertDocument('MTR.pdf',Path('MTR.pdf'),[]),vision,kinds=kinds)
    assert {row['kind'] for row in observations} == kinds
    assert {call.args[1] for call in vision.inspect.call_args_list} == kinds
    assert page.get_pixmap.call_count == len(kinds)


@pytest.mark.parametrize('required',[False,True])
def test_signature_rule_does_not_preemptively_inspect_stamp(service,required):
    rules = [r for r in bolt_rules() if r['evaluator'] in {'signature','signature_form','seal_policy','seal'}]
    batch, _ = setup_batch(service,rules[0],'Signature\nMTR must be stamped' if required else 'Signature')
    row = service.db.one('SELECT template_snapshot FROM review_batches WHERE id=?',(batch,))
    snapshot = json.loads(row['template_snapshot']);snapshot['required_items']=json.dumps(rules)
    service.db.execute('UPDATE review_batches SET template_snapshot=? WHERE id=?',(json.dumps(snapshot),batch))
    def observe(document,vision,check_cancel,*,kinds):
        return [{'page':1,'bbox':[10,10,20,20],'kind':kind,'state':'present','method':'local_vision','confidence':.99}
                for kind in kinds]
    with patch('app.auditing.bolt_audit.inspect_document',side_effect=observe) as inspect:
        run(service,batch)
    assert [call.kwargs['kinds'] for call in inspect.call_args_list] == ([{'signature'},{'stamp'}] if required else [{'signature'}])
    assert service.db.one('SELECT status FROM review_batches WHERE id=?',(batch,))['status'] == 'completed'

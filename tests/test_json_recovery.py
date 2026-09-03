from unittest.mock import Mock, patch

import pytest

from app.auditing.expert_review import build_rule_task_prompt
from app.database import ReviewDatabase
from app.integrations import ConfigStore, LLMClient
from app.llm.ollama_client import parse_json_object


@pytest.fixture
def client(tmp_path):
    db = ReviewDatabase(tmp_path/'test.db')
    settings = ConfigStore(db,tmp_path/'key').save({'llm_model':'Qwen3.8-27B-4bit'})
    client = LLMClient(settings)
    client._lm_studio_native = False
    return client


def response(content, finish='stop'):
    result = Mock()
    result.raise_for_status.return_value = None
    result.json.return_value = {'choices':[{'message':{'content':content},'finish_reason':finish}],
                                'usage':{'completion_tokens':350}}
    return result


@pytest.mark.parametrize('content,finish', [
    ('{"r":"存疑","c":"unclosed','stop'),
    ('{"r":"合格"}','length'),
    ('', 'length'),
    ('{"r":"存疑","e":{"nested":"evidence"}', 'stop'),
])
def test_malformed_or_truncated_json_recovers_once(client,content,finish):
    with patch('app.integrations.clients.requests.post',side_effect=[response(content,finish),response('{"r":"存疑"}')]) as post:
        result = client.generate_json('original evidence',retries=0,max_tokens=512,thinking=True,recover_json=True)
    assert result == {'r':'存疑'} and post.call_count == 2
    retry = post.call_args.kwargs['json']
    assert retry['max_tokens'] == 1024
    assert retry['chat_template_kwargs']['enable_thinking'] is False
    assert 'original evidence' in retry['messages'][0]['content']
    assert '不要续写片段' in retry['messages'][0]['content']


def test_bad_json_is_not_repaired_or_retried_forever(client):
    with patch('app.integrations.clients.requests.post',return_value=response('{"r":"合格')) as post:
        with pytest.raises(RuntimeError,match='JSON'):
            client.generate_json('test',retries=0,max_tokens=512,recover_json=True)
    assert post.call_count == 2


def test_recovery_does_not_retry_http_errors(client):
    import requests
    failure = Mock(status_code=422,text='bad request',reason='bad request')
    failure.raise_for_status.side_effect = requests.HTTPError(response=failure)
    with patch('app.integrations.clients.requests.post',return_value=failure) as post:
        with pytest.raises(RuntimeError,match='422'):
            client.generate_json('test',retries=0,recover_json=True)
    assert post.call_count == 1


def test_cancel_before_recovery_stops_second_request(client):
    cancel = Mock(side_effect=[None,RuntimeError('cancelled')])
    with patch('app.integrations.clients.requests.post',return_value=response('{"r":"bad')) as post:
        with pytest.raises(RuntimeError,match='cancelled'):
            client.generate_json('test',retries=0,recover_json=True,check_cancel=cancel)
    assert post.call_count == 1


def test_native_json_recovery_uses_native_limit(client):
    client._lm_studio_native = True
    failed, ok = Mock(),Mock()
    failed.json.return_value = {'output':[{'type':'message','content':'{"r":"bad'}]}
    ok.json.return_value = {'output':[{'type':'message','content':'{"r":"存疑"}'}]}
    with patch('app.integrations.clients.requests.post',side_effect=[failed,ok]) as post:
        assert client.generate_json('test',retries=0,max_tokens=512,thinking=True,recover_json=True) == {'r':'存疑'}
    assert post.call_args.kwargs['json']['max_output_tokens'] == 1024
    assert post.call_args.kwargs['json']['reasoning'] == 'off'


def test_parser_handles_braces_in_quotes_but_not_truncated_outer_object():
    assert parse_json_object('Result: {"e":"literal } brace"} trailing explanation') == {'e':'literal } brace'}
    for bad in ('{"e":{"a":1}', '[{"a":1}', '{"e":"unfinished }'):
        with pytest.raises(ValueError):
            parse_json_object(bad)
    with pytest.raises(ValueError):
        parse_json_object('{"r":"合格"}{"r":"不合格"}')


def test_compact_schema_is_not_mixed_with_long_schema():
    compact = build_rule_task_prompt('数据检查', [], '', [], compact=True)
    full = build_rule_task_prompt('数据检查', [], '', [])
    assert '"r":' in compact and '"result":' not in compact
    assert '"checked_scope":' not in compact and '不超过80字' in compact
    assert '"result":' in full and '"suggestion":' in full

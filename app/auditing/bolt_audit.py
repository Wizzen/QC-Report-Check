"""Explicit rule dispatch for bolt-v1; independent of template display names."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from app.auditing.bolt_template import SIGNATURE_NOTICE, SINGLE_NOTICE
from app.auditing.calibration import feedback_identity
from app.auditing.expert_review import ExpertDocument, parse_template_tasks, _document_type, rule_evaluation_from_llm, build_rule_task_prompt
from app.auditing.signatures import LocalVision, inspect_document, aggregate_observations
from app.database.v2 import utcnow
from app.integrations import LLMClient
from app.integrations.settings import ensure_url_allowed
from app.models import Finding, PageText


def document_wdcs(document: ExpertDocument) -> set[str]:
    # A path, prefix or file extension is not a WDC. Keep only isolated 4+4/6 or 8/10 digit tokens.
    result = set()
    filename = document.filename.replace('\\', '/').split('/')[-1]
    for match in re.finditer(r"(?<![A-Za-z0-9])(\d{4}[-_ ]\d{4}(?:\d{2})?|\d{10}|\d{8})(?![A-Za-z0-9])", filename):
        result.add(re.sub(r"\D", "", match.group()))
    for page in document.pages:
        for match in re.finditer(r"(?:WDC|PART\s*(?:NO\.?|NUMBER|#)|部件号|零件号)\s*[:：]?\s*(\d[\d _-]{6,16}\d)", page.text, re.I):
            value = re.sub(r"\D", "", match.group(1))
            if len(value) in {8, 10}:
                result.add(value)
    return result


def coverage_for(documents):
    types = {d.filename: _document_type(d) for d in documents}
    wdcs = {d.filename: document_wdcs(d) for d in documents}
    coc = set().union(*(wdcs[d.filename] for d in documents if types[d.filename] == "COC"))
    reports = set().union(*(wdcs[d.filename] for d in documents if types[d.filename] in {"COI", "MTR"}))
    return types, wdcs, {"covered_wdcs": sorted(coc & reports), "uncovered_wdcs": sorted(coc - reports),
                         "unmatched_report_wdcs": sorted(reports - coc)}


@dataclass
class ScopedTask:
    rule: dict
    documents: list[ExpertDocument]
    wdc: str = ""


def scoped_tasks(rules, documents, types, wdcs):
    result = []
    for rule in rules:
        if not rule.get("enabled", True):
            continue
        applicable = [d for d in documents if not rule.get("document_types") or types[d.filename] in rule["document_types"]]
        if rule.get("scope") == "batch":
            result.append(ScopedTask(rule, documents))
        elif rule.get("scope") == "wdc":
            identifiers = sorted(set().union(*(wdcs[d.filename] for d in applicable)))
            for wdc in identifiers:
                group = [d for d in applicable if wdc in wdcs[d.filename]]
                # A COC-only WDC is deliberately outside the report scope, not a passed group.
                if any(types[d.filename] in {"MTR", "COI"} for d in group):
                    result.append(ScopedTask(rule, group, wdc))
            if not any(task.rule is rule for task in result):
                result.append(ScopedTask(rule, []))
        else:
            result.extend(ScopedTask(rule, [d], ",".join(sorted(wdcs[d.filename]))) for d in applicable)
            if not applicable:
                result.append(ScopedTask(rule, []))
    return result


def evidence_documents(documents, task, budget=8000):
    """Fair per-document/page budgets, with intact neighbouring lines and table headers."""
    pages_count = sum(len(d.pages) for d in documents) or 1
    per_page = max(1, budget // pages_count)
    tokens = set(re.findall(r"[a-z]+|[\u3400-\u9fff]{2}", task.lower()))
    selected, coverage = [], []
    for document in documents:
        pages = []
        for page in document.pages:
            lines = page.text.splitlines()
            if len(page.text) <= per_page:
                text = page.text
            else:
                # Preserve complete context windows; never stitch half a table row into a false value.
                ranked = sorted(range(len(lines)), key=lambda i: (
                    sum(token in lines[i].lower() for token in tokens) * 8 +
                    (3 if '[TABLE_ROW]' in lines[i] else 0) + (2 if i < 4 or i >= len(lines)-4 else 0)), reverse=True)
                chosen, used = set(), 0
                for index in ranked:
                    neighbours = set(range(max(0, index-2), min(len(lines), index+3))) - chosen
                    size = sum(len(lines[i])+1 for i in neighbours)
                    if used + size <= per_page:
                        chosen.update(neighbours)
                        used += size
                text = '\n'.join(lines[i] for i in sorted(chosen))
            pages.append(PageText(page.page, text))
            coverage.append({"file": document.filename, "page": page.page, "complete": text == page.text})
        selected.append(ExpertDocument(document.filename, document.path, pages, document.ocr_status))
    return selected, coverage


def seal_requirement(documents, basis):
    """Require an explicit obligation, never a company logo, language or generic 'stamp' label."""
    positive = re.compile(r"无.{0,16}章.{0,8}无效|须.{0,12}(?:盖章|印章)|应.{0,12}加盖|必须.{0,12}(?:印章|盖章)|"
                          r"valid\s+only\s+after\s+being\s+stamped|must\s+(?:be\s+)?(?:stamped|bear.{0,20}seal)|seal\s+required", re.I)
    for document in documents:
        for page in document.pages:
            for line in page.text.splitlines():
                if positive.search(line):
                    return {"file": document.filename, "page": page.page, "quote": line}
    # Basis already scoped by the selected rule, but explicit applicability is still required.
    for line in basis.splitlines():
        if positive.search(line) and any(_document_type(d) in line.upper() for d in documents):
            return {"file": "所选采购依据", "page": 0, "quote": line}
    return None


def numeric_range(value):
    value = value.strip().replace('−', '-').replace('～', '~')
    number = r"[+-]?\d+(?:\.\d+)?"
    exact = re.fullmatch(rf"({number})", value)
    if exact:
        return float(exact[1]), float(exact[1])
    spread = re.fullmatch(rf"({number})\s*(?:±|\+/-)\s*(\d+(?:\.\d+)?)", value)
    if spread:
        return float(spread[1])-float(spread[2]), float(spread[1])+float(spread[2])
    interval = re.fullmatch(rf"({number})\s*[-~]\s*({number})", value)
    if interval:
        return tuple(sorted((float(interval[1]), float(interval[2]))))
    lower = re.fullmatch(rf"(?:min\.?|>=|≥)\s*({number})|({number})\s*min\.?", value, re.I)
    if lower:
        return float(lower[1] or lower[2]), float('inf')
    upper = re.fullmatch(rf"(?:max\.?|<=|≤)\s*({number})|({number})\s*max\.?", value, re.I)
    if upper:
        return -float('inf'), float(upper[1] or upper[2])
    return None


def table_checks(document, samples=False):
    columns, findings, checked, uncertain = {}, [], 0, []
    aliases = {"sample": r"sample|抽检|样本", "pass": r"^pass$|合格数",
               "spec": r"standard|spec|标准|要求", "actual": r"result|actual|实测|检验值"}
    for page in document.pages:
        for line in page.text.splitlines():
            if not line.startswith('[TABLE_ROW]'):
                continue
            cells = [value.strip() for value in line.removeprefix('[TABLE_ROW]').split('||')]
            header = {key: i for i, cell in enumerate(cells) for key, pattern in aliases.items() if re.search(pattern, cell, re.I)}
            if len(header) >= 2:
                columns = header
                continue
            if (columns and cells and cells[0]
                    and all(index >= len(cells) or not cells[index] for index in columns.values())):
                # A titled subsection or narrative row ends the previous
                # table's column mapping. Do not reuse Standard/Result offsets
                # in the following chemical composition or notes table.
                columns = {}
                continue
            keys = ("sample", "pass") if samples else ("spec", "actual")
            if not all(key in columns and columns[key] < len(cells) for key in keys):
                continue
            first, second = [cells[columns[key]] for key in keys]
            if samples:
                if not first and not second:
                    continue
                # '/', dashes and textual results do not belong to the
                # deterministic Sample/Pass numeric rule.
                if not first.isdigit() and not second.isdigit():
                    continue
                if not first.isdigit() or not second.isdigit():
                    uncertain.append((page.page, line)); continue
                failed = int(first) != int(second) or int(first) <= 0
            else:
                # C5.2 deliberately handles only explicit simple numeric
                # specifications. Headers, narrative rows, '/', chemical
                # columns and compound formulae are outside this evaluator;
                # they must not inflate the unresolved-row count.
                limits = numeric_range(first)
                if not limits:
                    continue
                actual = numeric_range(second)
                next_semantic_column = min(
                    (index for key, index in columns.items() if key != 'actual' and index > columns['actual']),
                    default=len(cells),
                )
                if (not actual and columns['actual'] + 1 < len(cells)
                        and columns['actual'] + 1 < next_semantic_column):
                    # Merged PDF cells occasionally shift Result one position
                    # to the right while preserving the visible table layout.
                    shifted = cells[columns['actual'] + 1]
                    actual = numeric_range(shifted)
                    if actual:
                        second = shifted
                if not actual:
                    uncertain.append((page.page, line)); continue
                failed = actual[0] < limits[0] or actual[1] > limits[1]
            checked += 1
            if failed:
                findings.append(Finding("检测结果不合格", "Major", cells[0] or "检验行",
                    "Sample/Pass数量不一致" if samples else "实测值超出明确标准范围", actual=second,
                    requirement=first, source_file=document.filename, source_page=page.page, source_text=line,
                    logic=f"按同一表头对应列比较：{first} / {second}", metadata={"origin": "deterministic"}))
    return findings, checked, uncertain


def product_marking(document):
    """Return a document-level product marking field, never a seal/signature."""
    pattern = re.compile(r"(?im)^\s*(?:product\s*)?(?:marking|产品印记|印记)\s*[:：]\s*([^\n|]+)")
    for page in document.pages:
        match = pattern.search(page.text)
        if not match:
            continue
        # PDF text layers often place the next label on the same logical line
        # separated by a wide run of spaces (e.g. "Marking: JDF 8.8  Order No").
        value = re.split(r"\s{2,}", match.group(1).strip(), maxsplit=1)[0].strip().strip('-_/ ')
        if value:
            return value, page.page, match.group(0).strip()
    return '', 1, ''


def run_bolt_audit(service, batch_id):
    db = service.db
    batch = db.one('SELECT * FROM review_batches WHERE id=?', (batch_id,))
    snapshot = json.loads(batch['template_snapshot'])
    rules = parse_template_tasks(snapshot.get('required_items', '[]'))
    rows = db.query("SELECT d.* FROM documents d JOIN batch_documents bd ON d.id=bd.document_id WHERE bd.batch_id=? AND bd.role='supplier'", (batch_id,))
    from app.auditing.v2_service import _expert_document
    documents = [_expert_document(row) for row in rows]
    by_name = {row['original_name']: row for row in rows}
    types, wdcs, coverage = coverage_for(documents)
    settings = service.config_store.get()
    ensure_url_allowed(settings.llm_base_url, False)
    client = LLMClient(settings)
    client.generation_concurrency_limit()
    vision, visual = LocalVision(client), {}
    tasks = scoped_tasks(rules, documents, types, wdcs)
    findings = []
    db.execute('DELETE FROM rule_evaluations WHERE batch_id=?', (batch_id,))
    for index, task in enumerate(tasks, 1):
        service._check_cancel(batch_id)
        rule, docs = task.rule, task.documents
        engine = rule.get('evaluator', 'llm')
        # Existing batch snapshots may predate the deterministic C5.3 engine.
        # Bind by stable rule id so retries receive the accuracy fix without
        # rewriting their historical evidence.
        if str(rule.get('rule_id')) == 'C5.3':
            engine = 'marking'
        name = rule['text']
        basis = service._basis_context(batch_id, [], name)
        status, conclusion, items, extra = '合格', '', [], {}
        extra['feedback_identity'] = feedback_identity(batch.get('template_id'), snapshot, rule, settings)
        suppress_finding = False
        source = docs[0] if docs else None
        if not docs:
            status, conclusion = '不适用', '未提供适用文件或没有纳入范围的WDC；不重复报告下游缺失'
        elif engine == 'manifest':
            missing = []
            if batch['audit_scope'] == 'full_package':
                if 'COC' not in types.values(): missing.append('COC')
                if not {'COI', 'MTR'} & set(types.values()): missing.append('COI/MTR')
            if missing:
                status = '存疑' if 'OTHER' in types.values() else '不合格'
                conclusion = '文件类型不明确，无法确认必需文件' if status == '存疑' else '缺少必需文件：' + '、'.join(missing)
            else:
                conclusion = SINGLE_NOTICE if batch['audit_scope'] == 'single_document' else '必需文件类型存在；WDC覆盖范围另列'
                if batch['audit_scope'] == 'full_package' and coverage['unmatched_report_wdcs']:
                    status, conclusion = '存疑', '检测报告WDC未与COC建立对应关系，请核对归属：' + ','.join(coverage['unmatched_report_wdcs'])
                elif batch['audit_scope'] == 'full_package' and any(not wdcs[d.filename] for d in docs if types[d.filename] in {'COC', 'MTR', 'COI'}):
                    status, conclusion = '存疑', '必需文件类型存在，但部分文件WDC无法可靠提取，不能确认文件包对应关系'
            extra['coverage'] = coverage
        elif engine in {'signature', 'signature_form', 'seal', 'seal_policy'}:
            requirement = seal_requirement(docs, basis)
            extra['seal_requirement'] = requirement
            if engine == 'seal_policy':
                status = '合格' if requirement else '不适用'
                conclusion = requirement['quote'] if requirement else '未发现明确印章要求，不按语言推断'
            elif engine == 'seal' and not requirement:
                status, conclusion = '不适用', '无明确印章要求'
            elif engine == 'signature_form':
                status, conclusion = '不适用', SIGNATURE_NOTICE
            else:
                kind = 'stamp' if engine == 'seal' else 'signature'
                visual_key = (source.filename, kind)
                if visual_key not in visual:
                    visual[visual_key] = inspect_document(source, vision, lambda: service._check_cancel(batch_id), kinds={kind})
                    for observation in visual[visual_key]:
                        db.execute("""INSERT INTO visual_evidence(batch_id,document_id,page,bbox,kind,state,method,confidence,
                            details,model_fingerprint,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                            (batch_id, by_name[source.filename]['id'], observation['page'], json.dumps(observation['bbox']),
                             observation['kind'], observation['state'], observation['method'], observation['confidence'],
                             json.dumps(observation, ensure_ascii=False), observation.get('model_fingerprint', ''), utcnow()))
                state, evidence = aggregate_observations(visual[visual_key], kind)
                extra['visual'] = evidence
                if engine == 'signature_form':
                    status, conclusion = '不适用', SIGNATURE_NOTICE
                else:
                    status = {'present': '合格', 'absent': '不合格', 'unknown': '存疑'}[state]
                    conclusion = ('已识别签名；未验证身份或证书有效性' if kind == 'signature' else '已识别印章') if state == 'present' else (
                        ('适用签名区域明确空白' if kind == 'signature' else '文件明确要求印章，但全部页面未见印章') if state == 'absent' else '视觉证据不足，需人工检查原页；不判定签章缺失')
        elif engine == 'wdc':
            status = '合格' if wdcs[source.filename] else '存疑'
            conclusion = '已识别WDC：' + ','.join(sorted(wdcs[source.filename])) if status == '合格' else 'WDC未能可靠提取，请确认部件归属'
        elif engine == 'marking':
            value, marking_page, marking_evidence = product_marking(source)
            if value:
                has_explicit_basis = bool(re.search(r'(?i)\bmarking\b|产品印记|印记要求', basis or ''))
                if has_explicit_basis:
                    status, conclusion = '存疑', f'已识别产品Marking：{value}；采购依据包含印记要求，需确认目标值是否一致'
                else:
                    status, conclusion = '合格', f'已识别产品Marking：{value}；未发现采购依据指定目标印记'
                extra['marking'] = {'value': value, 'page': marking_page, 'evidence': marking_evidence}
            else:
                status, conclusion = '存疑', '未定位明确的产品Marking/印记字段；公司印章和签名不作为产品印记证据'
        elif engine == 'pages':
            conclusion = f'实际解析 {len(source.pages)} 页；无印刷页码不等于缺页'
            declared = re.findall(r'page\s+\d+\s+of\s+(\d+)', source.text, re.I)
            if declared and any(int(total) != len(source.pages) for total in declared):
                status, conclusion = '存疑', '文件声明的页数与实际PDF页数不一致，请检查分卷或缺页'
        elif engine == 'readability':
            bad = [p.page for p in source.pages if '[OCR_FAILED_PAGE]' in p.text or len(re.sub(r'\s+', '', p.text)) < 40 or p.text.count('�') > 3]
            status = '存疑' if bad else '合格'
            conclusion = f'第 {bad} 页识别质量不足；不据此断言原页模糊' if bad else '文字层可提取；签章图像由专用规则检查'
        elif engine in {'table_samples', 'table_values'}:
            items, count, unknown = table_checks(source, engine == 'table_samples')
            status = '不合格' if items else ('存疑' if unknown or not count else '合格')
            conclusion = f'已比较 {count} 行，异常 {len(items)} 行，未可靠比较 {len(unknown)} 行'
            if not count and not unknown: conclusion = '未识别完整对应表头，不能可靠比较；请人工检查原表'
        else:
            selected, inspected = evidence_documents(docs, name + ' ' + str(rule.get('criterion', '')))
            extra['checked_pages'] = inspected
            prompt = build_rule_task_prompt(name, selected,
                snapshot.get('review_instructions', '') + '\n当前规则标准：' + str(rule.get('criterion', '')) +
                '\n当前WDC：' + task.wdc + '\n缺失范围由程序核验，不得虚构已检查范围。', [], basis,
                compact=batch.get('review_mode') != 'deep')
            try:
                deep = batch.get('review_mode') == 'deep'
                payload = client.generate_json(prompt, thinking=deep, retries=0, max_tokens=4096 if deep else 512,
                                               timeout_seconds=settings.llm_timeout_seconds if deep else min(90, settings.llm_timeout_seconds),
                                               recover_json=True, check_cancel=lambda: service._check_cancel(batch_id))
                # Scope calibration to the evidence document. Ambiguous cross-file
                # findings must not inherit another document type's feedback.
                evidence_file = str(payload.get('source_file') or payload.get('f') or '')
                eligible_files = {document.filename for document in docs}
                policy_type = types.get(evidence_file, '') if evidence_file in eligible_files else (
                    types.get(source.filename, '') if len(docs) == 1 else '')
                policy = db.feedback_policy_for(**extra['feedback_identity'],
                    rule_code=str(rule.get('rule_id', name)), document_type=policy_type) if policy_type else {}
                extra['feedback_policy'] = policy
                evaluation, finding = rule_evaluation_from_llm(payload, name, docs, service.confidence_threshold, policy)
                status, conclusion = evaluation['status'], str(evaluation['conclusion'])
                extra['downgrade_reasons'] = list(evaluation.get('downgrade_reasons', []))
                suppress_finding = 'unlocated_low_confidence' in extra['downgrade_reasons']
                if finding:
                    if evaluation['evidence_type'] == 'absence':
                        finding.severity, status = 'Review', '存疑'
                        finding.metadata['checked_pages'] = inspected
                        finding.metadata.setdefault('downgrade_reasons', []).append('absence_not_program_verified')
                        conclusion = '未在已提供证据中定位该信息，需核对完整原页；不作为明确缺失'
                        finding.description = conclusion
                    elif finding.severity == 'Major':
                        context = '\n'.join(p.text for d in docs if d.filename == finding.source_file for p in d.pages if p.page == finding.source_page)
                        try:
                            check = client.generate_json('只检查结论是否被证据直接证明，而不是仅存在一条要求。只返回 {"confirmed":true或false}。\n'
                                + f'规则：{name}\n结论：{finding.description}\n原页：{context[:10000]}',
                                thinking=False, retries=0, max_tokens=80, timeout_seconds=45)
                        except Exception as exc:
                            service._check_cancel(batch_id)
                            check = {}
                            finding.metadata['second_review_error'] = str(exc)[:240]
                        if check.get('confirmed') is not True:
                            finding.severity, status = 'Review', '存疑'
                            reasons = finding.metadata.setdefault('downgrade_reasons', [])
                            if 'second_review_not_confirmed' not in reasons:
                                reasons.append('second_review_not_confirmed')
                            conclusion = finding.description + '（二次复核未确认，需人工核验）'
                            finding.description = conclusion
                        finding.metadata['result'] = status
                    finding.metadata['result'] = status
                    extra.update(finding.metadata)
                    items = [finding]
            except Exception as exc:
                service._check_cancel(batch_id)
                status, conclusion = '调用失败', f'规则未完成：{str(exc)[:240]}'
        service._check_cancel(batch_id)
        if not items and not suppress_finding and status in {'不合格', '存疑', '调用失败'}:
            observation = extra.get('visual', {})
            page = int(observation.get('page') or 1)
            evidence_type = 'visual' if observation else ('absence' if engine == 'manifest' else 'review_scope')
            items = [Finding('规则检查', 'Major' if status == '不合格' else 'Review', name, conclusion,
                source_file=source.filename if source else '', source_page=page,
                source_text=str(observation.get('description') or conclusion), requirement=str(rule.get('criterion', '')),
                metadata={'origin': 'visual' if observation else 'deterministic', 'result': status, 'evidence_type': evidence_type,
                          'evidence': [{'file': source.filename if source else '', 'page': page,
                                        'bbox': observation.get('bbox', []), 'source_text': str(observation.get('description') or conclusion),
                                        'evidence_type': evidence_type}]})]
        scope_meta = {'rule_id': rule.get('rule_id', name), 'wdc': task.wdc, 'files': [d.filename for d in docs], **extra}
        for item in items:
            item.rule_code = str(rule.get('rule_id', name))
            item.document_type = types.get(item.source_file, '')
            item.metadata.update(scope_meta)
            findings.append(item)
        db.execute("""INSERT INTO rule_evaluations(batch_id,task_index,task_name,status,conclusion,source_file,source_page,
                    evidence,metadata,completed_at) VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (batch_id, index, name, status, conclusion, source.filename if source else '',
             int(extra.get('visual', {}).get('page') or 1), str(extra.get('visual', {}).get('description') or ''),
             json.dumps(scope_meta, ensure_ascii=False), utcnow()))
        if items:
            item = items[0]
            db.execute('''UPDATE rule_evaluations SET source_file=?,source_page=?,evidence=?,evidence_type=?,
                actual=?,requirement=?,logic=?,suggestion=?,confidence=? WHERE batch_id=? AND task_index=?''',
                (item.source_file, item.source_page, item.source_text, item.metadata.get('evidence_type', 'source'),
                 item.actual, item.requirement, item.logic, item.suggestion, item.confidence, batch_id, index))
        service._activity(batch_id, progress=60+int(34*index/max(1,len(tasks))), stage='按文件/WDC审核',
                          activity=f'{index}/{len(tasks)} {name} · {source.filename if source else "无适用文件"}')
    # Merge only identical evidence within a scope; never merge different numerical rows.
    merged = {}
    for item in findings:
        key = (item.source_file, item.source_page, item.source_text, item.metadata.get('wdc', ''), item.description)
        if key in merged:
            merged[key].metadata.setdefault('related_rules', [merged[key].rule_code]).append(item.rule_code)
        else:
            merged[key] = item
    return list(merged.values()), {**coverage, 'audit_scope': batch['audit_scope'],
        'scope_notice': SINGLE_NOTICE if batch['audit_scope'] == 'single_document' else '完整文件包审核；仅对覆盖的WDC给出结论',
        'signature_notice': SIGNATURE_NOTICE, 'vision_status': vision.reason,
        'model_identity': vision.model_identity, 'model_fingerprint': vision.fingerprint}

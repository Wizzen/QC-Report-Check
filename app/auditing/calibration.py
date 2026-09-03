"""Inference identities for local feedback. No notes or business values in prompts."""
from __future__ import annotations

import hashlib
import json

from app.auditing.bolt_template import BOLT_ENGINE, EXTRACTION_VERSION


def feedback_identity(template_id, snapshot, rule, settings):
    """Changes to models, extraction, policy or rule meaning invalidate calibration."""
    version = {
        'engine': BOLT_ENGINE, 'calibration': 'v1', 'parser': EXTRACTION_VERSION,
        'llm_endpoint': settings.llm_base_url.rstrip('/'), 'model': settings.llm_model,
        'ocr_endpoint': settings.ocr_base_url.rstrip('/'),
        'ocr_backend': settings.ocr_backend, 'ocr_lang': settings.ocr_lang,
        'template_version': snapshot.get('template_version', 1),
        'instructions': snapshot.get('review_instructions', ''),
        'rule': {key: value for key, value in rule.items() if key != 'enabled'},
    }
    digest = hashlib.sha256(json.dumps(version, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return {'template_key': f'{BOLT_ENGINE}:template:{template_id}', 'model_fingerprint': digest}

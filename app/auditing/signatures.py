"""Local image capability check and non-destructive PDF signature observations.

These are presence observations, not cryptographic or legal validation.
"""
from __future__ import annotations

import hashlib
import math
import re
import secrets
from io import BytesIO

import pymupdf
from PIL import Image, ImageDraw, ImageFont

from app.integrations.settings import ensure_url_allowed

SIGN_ANCHOR = re.compile(r"signature|signed by|authorized by|签字|签名|签署", re.I)

_CONFIDENCE_LABELS = {
    "high": .95, "高": .95, "medium": .70, "中": .70,
    "low": .30, "低": .30,
}


def normalize_confidence(value) -> tuple[float, bool]:
    """Normalize common vision-model confidence formats without failing a page."""
    if value is None or isinstance(value, bool):
        return 0.0, False
    if isinstance(value, str):
        text = value.strip().casefold()
        if text in _CONFIDENCE_LABELS:
            return _CONFIDENCE_LABELS[text], True
        try:
            number = float(text[:-1]) / 100 if text.endswith("%") else float(text)
        except ValueError:
            return 0.0, False
    elif isinstance(value, (int, float)):
        number = float(value)
    else:
        return 0.0, False
    return (number, True) if math.isfinite(number) and 0 <= number <= 1 else (0.0, False)


class LocalVision:
    def __init__(self, client):
        self.client = client
        self.available = None
        self.reason = "尚未验证图片能力"
        identity = client.model_identity() if callable(getattr(client, "model_identity", None)) else client.settings.llm_model
        self.model_identity = str(identity or "未识别")
        self.fingerprint = hashlib.sha256(
            f"{client.settings.llm_base_url}|{self.model_identity}|signature-v2".encode()
        ).hexdigest()[:24]

    def _refresh_model_identity(self) -> None:
        identity = self.client.model_identity() if callable(getattr(self.client, "model_identity", None)) else self.client.settings.llm_model
        self.model_identity = str(identity or "未识别")
        self.fingerprint = hashlib.sha256(
            f"{self.client.settings.llm_base_url}|{self.model_identity}|signature-v2".encode()
        ).hexdigest()[:24]

    def probe(self) -> bool:
        if self.available is not None:
            return self.available
        try:
            ensure_url_allowed(self.client.settings.llm_base_url, False)
            # The answer occurs only in pixels, never in the prompt or filename.
            token = secrets.token_hex(3).upper()
            canvas = Image.new("RGB", (440, 150), "white")
            ImageDraw.Draw(canvas).text((24, 42), token, fill="black", font=ImageFont.load_default(size=56))
            data = BytesIO()
            canvas.save(data, format="PNG")
            result = self.client.generate_json(
                'Read the six characters in this image. Return only {"code":"..."}.',
                images=[data.getvalue()], thinking=False, retries=0, max_tokens=60, timeout_seconds=35,
            )
            self._refresh_model_identity()
            self.available = str(result.get("code", "")).strip() == token
            self.reason = "图片读取验证通过" if self.available else "图片答案不正确，视觉识别未启用"
        except Exception as exc:
            self.available = False
            self.reason = f"视觉识别未启用：{str(exc)[:200]}"
        return bool(self.available)

    def inspect(self, image: bytes, kind: str, anchored: bool) -> dict:
        if not self.probe():
            return {"state": "unknown", "confidence": 0.0, "description": self.reason}
        prompt = (
            "检查这张原页区域图片中的" + ("签署痕迹" if kind == "signature" else "印章") + "。"
            "普通印刷姓名、空签名框、签名栏标题、公司logo、印记检验文字都不证明存在签章。"
            "手写或电子签名图像算签署痕迹，不能验证身份/法律有效性。模糊或无法区分必须unknown。"
            "明确的电子签署声明只有与签署位置和签署角色对应时才可作为签署证据，普通打印姓名不可以。"
            '只返回JSON：{"state":"present|absent|unknown","confidence":0.0,"description":"简短视觉依据"}。'
        )
        try:
            raw = self.client.generate_json(prompt, images=[image], retries=0, thinking=False,
                                            timeout_seconds=60, max_tokens=180)
            state = raw.get("state", "unknown")
            raw_confidence = raw.get("confidence", 0)
            confidence, confidence_valid = normalize_confidence(raw_confidence)
            if state not in {"present", "absent", "unknown"} or not confidence_valid:
                state, confidence = "unknown", 0.0
            if confidence < .90 or (state == "absent" and kind == "signature" and not anchored):
                state = "unknown"
            description = str(raw.get("description", ""))[:400]
            if not confidence_valid:
                description = (description + "；模型置信度格式无效，已转人工复核").strip("；")
            return {"state": state, "confidence": confidence, "description": description,
                    "confidence_normalized": isinstance(raw_confidence, str) and confidence_valid}
        except Exception as exc:
            return {"state": "unknown", "confidence": 0.0, "description": f"视觉识别失败：{str(exc)[:200]}"}


def inspect_document(document, vision: LocalVision, check_cancel=lambda: None, *, kinds=None) -> list[dict]:
    kinds = {'signature', 'stamp'} if kinds is None else set(kinds)
    if not kinds <= {'signature', 'stamp'}:
        raise ValueError('未知签章检查类型')
    if not kinds:
        return []
    observations = []
    if document.path.suffix.lower() != ".pdf":
        if document.path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            with Image.open(document.path) as picture:
                data = BytesIO()
                picture.convert("RGB").save(data, format="PNG")
            for kind in ("signature", "stamp"):
                if kind not in kinds:
                    continue
                check_cancel()
                observations.append({"page": 1, "bbox": [], "kind": kind, "method": "local_vision",
                    **vision.inspect(data.getvalue(), kind, False), "model_fingerprint": vision.fingerprint})
        else:
            for kind in sorted(kinds):
                observations.append({"page": 1, "bbox": [], "kind": kind, "method": "unsupported",
                                     "state": "unknown", "confidence": 0, "description": "此格式暂不支持签章图像检查"})
        return observations
    with pymupdf.open(document.path) as pdf:
        for index, page in enumerate(pdf, 1):
            check_cancel()
            for widget in (page.widgets() or []) if 'signature' in kinds else []:
                if widget.field_type == pymupdf.PDF_WIDGET_TYPE_SIGNATURE:
                    observations.append({"page": index, "bbox": list(widget.rect), "kind": "digital_signature",
                        "state": "present" if widget.is_signed else "unfilled", "method": "pdf_widget",
                        "confidence": 1.0, "description": "已填数字签名字段；未验证证书" if widget.is_signed else "未填数字签名字段"})
            anchors = [pymupdf.Rect(block[:4]) for block in page.get_text("blocks")
                       if len(block) > 4 and SIGN_ANCHOR.search(str(block[4]))]
            regions = []
            for rect in anchors:
                region = pymupdf.Rect(max(0, rect.x0 - 18), max(0, rect.y0 - 75),
                                     min(page.rect.width, rect.x1 + 45), min(page.rect.height, rect.y1 + 60))
                if not any((region & other).get_area() > .8 * region.get_area() for other in regions):
                    regions.append(region)
            if not regions:
                regions = [page.rect]
            for region in regions if 'signature' in kinds else []:
                check_cancel()
                image = page.get_pixmap(matrix=pymupdf.Matrix(1.6, 1.6), clip=region).tobytes("png")
                observations.append({"page": index, "bbox": list(region), "kind": "signature",
                    "method": "local_vision", "anchored": bool(anchors),
                    **vision.inspect(image, "signature", bool(anchors)), "model_fingerprint": vision.fingerprint,
                    "model_identity": getattr(vision, "model_identity", "")})
            if 'stamp' in kinds:
                check_cancel()
                image = page.get_pixmap(matrix=pymupdf.Matrix(1.25, 1.25)).tobytes("png")
                observations.append({"page": index, "bbox": list(page.rect), "kind": "stamp", "method": "local_vision",
                    **vision.inspect(image, "stamp", False), "model_fingerprint": vision.fingerprint,
                    "model_identity": getattr(vision, "model_identity", "")})
    return observations


def aggregate_observations(observations: list[dict], kind: str) -> tuple[str, dict]:
    kinds = {kind, "digital_signature"} if kind == "signature" else {kind}
    rows = [row for row in observations if row.get("kind") in kinds]
    present = next((row for row in rows if row["state"] == "present"), None)
    if present:
        if kind == 'signature' and any(row['state'] == 'absent' and row.get('anchored') for row in rows):
            return 'unknown', {**present, 'state': 'unknown',
                'description': '不同签署区域结果冲突：部分区域有签名、部分明确空白，需人工核对必需签署角色'}
        return "present", present
    checked = [row for row in rows if row.get("method") == "local_vision"]
    if checked and all(row["state"] == "absent" for row in checked):
        return "absent", checked[0]
    return "unknown", next((row for row in rows if row["state"] == "unknown"), rows[0] if rows else {})

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PageText:
    page: int
    text: str


@dataclass(frozen=True)
class ExtractedItem:
    key: str
    raw: str
    value: float | str | None
    unit: str = ""
    page: int = 1
    source_text: str = ""
    category: str = "field"


@dataclass(frozen=True)
class Requirement:
    item: str
    operator: str
    value: float | str | None
    upper_value: float | None = None
    unit: str = ""
    raw: str = ""
    source_file: str = ""
    source_page: int = 1
    clause: str = ""
    required: bool = False


@dataclass
class Finding:
    category: str
    severity: str
    item: str
    description: str
    actual: str = ""
    requirement: str = ""
    source_file: str = ""
    source_page: int = 1
    source_text: str = ""
    standard_file: str = ""
    standard_page: int = 1
    standard_clause: str = ""
    logic: str = ""
    suggestion: str = "请供应商核实并提交符合要求的证据。"
    confidence: float = 1.0
    status: str = "AI发现"
    metadata: dict[str, Any] = field(default_factory=dict)
    rule_code: str = ""
    rule_version: int = 1
    document_type: str = ""
    extraction_confidence: float = 1.0
    decision_confidence: float = 1.0

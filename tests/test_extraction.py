from app.extractors import extract_items, extract_requirements
from app.models import PageText


def test_extract_traceable_supplier_values() -> None:
    pages = [PageText(3, "Material Grade: Q355B\nHeat No: H240812\nBatch No: 00125\nTensile Strength: 455 MPa\nP: 0.041 %")]
    items = extract_items(pages)
    values = {(item.key, item.raw) for item in items}
    assert ("material_grade", "Q355B") in values
    assert ("heat_number", "H240812") in values
    assert ("抗拉强度", "455 MPa") in values
    assert all(item.page == 3 for item in items)


def test_extract_comparison_range_tolerance_and_required() -> None:
    text = "5.3.2 抗拉强度 ≥ 470 MPa\n5.3.3 Rm = 470~630 MPa\n厚度 10 ± 0.2 mm\n必须进行冲击试验"
    requirements = extract_requirements([PageText(12, text)], "采购技术协议.pdf")
    assert any(item.operator == ">=" and item.value == 470 for item in requirements)
    assert any(item.operator == "range" and item.value == 470 and item.upper_value == 630 for item in requirements)
    assert not any(item.item == "抗拉强度" and item.operator == "=" and item.value == 470 for item in requirements)
    assert any(item.item == "厚度" and item.value == 9.8 and item.upper_value == 10.2 for item in requirements)
    assert any(item.item == "冲击功" and item.required for item in requirements)

from __future__ import annotations

import pymupdf


def render_pdf_evidence_bytes(
    path_text: str, page_number: int, source_text: str, actual: str, item: str
) -> tuple[bytes, bool]:
    """Render a focused PDF screenshot and outline exact source evidence when found."""
    with pymupdf.open(path_text) as document:
        page_index = max(0, min(page_number - 1, document.page_count - 1))
        page = document.load_page(page_index)
        matches: list[pymupdf.Rect] = []
        for candidate in evidence_candidates(source_text, actual, item):
            matches = page.search_for(candidate)
            if matches:
                break
        clip = page.rect
        if matches:
            focus = matches[0]
            for match in matches[1:6]:
                if abs(match.y0 - focus.y0) <= 80:
                    focus |= match
            page.draw_rect(focus + (-3, -3, 3, 3), color=(0.85, 0.12, 0.08), width=2.2, overlay=True)
            clip = pymupdf.Rect(
                page.rect.x0,
                max(page.rect.y0, focus.y0 - 105),
                page.rect.x1,
                min(page.rect.y1, focus.y1 + 105),
            )
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(1.6, 1.6), clip=clip, alpha=False)
        return pixmap.tobytes("png"), bool(matches)


def evidence_candidates(source_text: str, actual: str, item: str) -> list[str]:
    values: list[str] = []
    source = source_text.removeprefix("[TABLE_ROW] ")
    for value in [source, *source.split(" || "), actual, item]:
        compact = " ".join(str(value).split()).strip()
        if len(compact) >= 3 and compact not in values:
            values.append(compact)
    values.sort(key=lambda value: (len(value) > 80, -len(value)))
    return values

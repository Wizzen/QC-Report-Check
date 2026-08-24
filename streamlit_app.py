from __future__ import annotations

import streamlit as st

from app.logging_config import configure_logging
from app.ui.pages import (
    basis_library_page,
    review_records_page,
    settings_page,
    start_review_page,
    supplier_library_page,
    templates_page,
)
from app.ui.styles import apply_global_styles


st.set_page_config(page_title="供应商质量智能审查", page_icon=":material/fact_check:", layout="wide")
configure_logging()
apply_global_styles()

navigation = st.navigation(
    [
        st.Page(start_review_page, title="开始审核", icon=":material/upload_file:", default=True),
        st.Page(review_records_page, title="审核记录", icon=":material/fact_check:"),
        st.Page(basis_library_page, title="审核依据库", icon=":material/library_books:"),
        st.Page(supplier_library_page, title="供应商档案", icon=":material/inventory_2:"),
        st.Page(templates_page, title="规则模板", icon=":material/rule:"),
        st.Page(settings_page, title="系统设置", icon=":material/settings:"),
    ],
    position="top",
)
navigation.run()


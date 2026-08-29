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

start_page = st.Page(start_review_page, title="开始审核", icon=":material/upload_file:", default=True)
review_page = st.Page(review_records_page, title="审核中心", icon=":material/dashboard:")
st.session_state["_review_page"] = review_page

navigation = st.navigation(
    [
        start_page,
        review_page,
        st.Page(basis_library_page, title="审核依据库", icon=":material/library_books:"),
        st.Page(templates_page, title="规则模板", icon=":material/rule:"),
        st.Page(settings_page, title="系统设置", icon=":material/settings:"),
    ],
    position="top",
)
navigation.run()

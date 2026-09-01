from __future__ import annotations

import streamlit as st


GLOBAL_CSS = """
<style>
/* Deliberately small CSS layer: the theme owns colors; these rules create the requested upload-focused composition. */
.stMainBlockContainer {max-width: 1180px; padding-top: 3.5rem; padding-bottom: 4rem;}
[data-testid="stHeader"] {background: rgba(255,255,255,.94); border-bottom: 1px solid #eceef1; backdrop-filter: blur(14px);}
[data-testid="stNavigation"] {max-width: 1180px; margin: 0 auto;}
.qaqc-hero {text-align:center; max-width:840px; margin:.35rem auto .8rem;}
.qaqc-hero h1 {font-size:clamp(2.2rem,4vw,3.25rem); line-height:1.05; letter-spacing:-.055em; margin:.45rem 0 .5rem; color:#090d18;}
.qaqc-hero p {font-size:.96rem; line-height:1.5; color:#667085; margin:0 auto; max-width:720px;}
.qaqc-section-title {font-size:1.8rem; letter-spacing:-.025em; margin:0 0 .25rem;}
.qaqc-section-sub {color:#667085; margin:0 0 1.6rem;}
.qaqc-privacy {display:flex; justify-content:center; align-items:center; gap:.5rem; color:#64748b; font-size:.82rem; margin-top:.55rem;}
.qaqc-privacy:before {content:'✓'; display:grid; place-items:center; width:1.1rem; height:1.1rem; border-radius:50%; background:#ecfdf3; color:#16803b; font-weight:800;}
.st-key-upload_supplier [data-testid="stFileUploaderDropzone"],
.st-key-upload_supplemental [data-testid="stFileUploaderDropzone"] {min-height:150px; display:flex; align-items:center; justify-content:center;
  border:1.5px dashed #cfd4dc; background:linear-gradient(145deg,#fff,#fafbfc); border-radius:1rem;}
.st-key-upload_supplier [data-testid="stFileUploaderDropzone"]:hover,
.st-key-upload_supplemental [data-testid="stFileUploaderDropzone"]:hover {border-color:#6b7280; background:#f8fafc;}
.st-key-start_review button {min-height:3.15rem; font-size:1rem; font-weight:700; box-shadow:0 8px 22px rgba(17,24,39,.13);}
.qaqc-file-label {font-size:.86rem; font-weight:700; color:#344054; margin:.3rem 0 .5rem;}
.qaqc-badge {display:inline-flex; align-items:center; border:1px solid #e5e7eb; border-radius:999px; padding:.25rem .6rem; font-size:.76rem; color:#475569; background:#fff;}
.qaqc-badge.remote {color:#9a3412; background:#fff7ed; border-color:#fed7aa;}
.qaqc-evidence {border:1px solid #e5e7eb; border-radius:.9rem; padding:1rem; background:#fafafa; min-height:180px;}
.qaqc-evidence + .qaqc-evidence {margin-top:.75rem;}
.qaqc-evidence .meta {font-size:.78rem; color:#667085; margin-bottom:.75rem;}
.qaqc-evidence pre {white-space:pre-wrap; word-break:break-word; font-size:.84rem; margin:0; font-family:ui-monospace,SFMono-Regular,Consolas,monospace;}
.st-key-dashboard_all button,.st-key-dashboard_completed button,.st-key-dashboard_processing button,
.st-key-dashboard_issues button,.st-key-dashboard_major button,.st-key-dashboard_review button {
  min-height:5rem; border-radius:.9rem; font-size:1rem; font-weight:700;
}
.st-key-dashboard_all button p,.st-key-dashboard_completed button p,.st-key-dashboard_processing button p,
.st-key-dashboard_issues button p,.st-key-dashboard_major button p,.st-key-dashboard_review button p {line-height:1.2;}
[data-testid="stButton"] button {min-height:2.55rem;}
[data-testid="stDownloadButton"] button {min-height:3rem; font-weight:700;}
[data-testid="stDataFrame"] {border-radius:.75rem; overflow:hidden;}
@media (max-width: 700px) {
  .stMainBlockContainer {padding:3rem 1rem 4rem;}
  .qaqc-hero {margin:.7rem auto 1.4rem;}
  .qaqc-hero h1 {font-size:2rem; line-height:1.12; letter-spacing:-.045em;}
  .qaqc-hero p {font-size:.96rem;}
  .st-key-upload_supplier [data-testid="stFileUploaderDropzone"],
  .st-key-upload_supplemental [data-testid="stFileUploaderDropzone"] {min-height:190px;}
  .qaqc-evidence {min-height:0;}
}
</style>
"""


def apply_global_styles() -> None:
    st.html(GLOBAL_CSS)

"""Shared styling helpers for the FEVS dashboard."""
from __future__ import annotations

import streamlit as st


_GLOBAL_CSS = """
<style>
:root {
    --fevs-font-base: 1.05rem;
    --fevs-font-accent: #0f172a;
    --fevs-font-secondary: #334155;
}

html, body, [class*="block-container"], [data-testid="stSidebar"] * {
    font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    color: var(--fevs-font-accent);
}

[class*="block-container"] p,
[class*="block-container"] span,
[class*="block-container"] li,
[class*="block-container"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] li,
[data-testid="stSidebar"] label {
    font-size: var(--fevs-font-base);
    line-height: 1.6;
}

h1, h2, h3, h4, h5, h6 {
    color: var(--fevs-font-accent);
    font-weight: 700 !important;
}

h1 {
    font-size: 2.6rem !important;
}

h2 {
    font-size: 2.1rem !important;
}

h3 {
    font-size: 1.65rem !important;
}

h4 {
    font-size: 1.35rem !important;
}

[data-testid="stMarkdownContainer"] > p {
    font-size: var(--fevs-font-base);
}

[data-testid="stMetricValue"] {
    font-size: 2rem;
}

[data-testid="stMetricDelta"] {
    font-size: 1.1rem;
}

[data-testid="stCaptionContainer"] p,
small, .stCaption {
    font-size: 1rem !important;
    color: var(--fevs-font-secondary);
}

[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: var(--fevs-font-accent);
}
</style>
"""


def apply_global_styles() -> None:
    """Inject global CSS rules to harmonize typography across pages."""
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)

"""SEPA Guide page — renders the full Minervini method reference.

Streamlit auto-discovers files in this ``pages/`` dir and adds them to the sidebar
nav. Content is read live from ``minervini_sepa_system.md`` so there is a single
source of truth (edit the markdown, not this page).
"""
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="SEPA Guide", page_icon="📖", layout="wide")
st.title("📖 Minervini SEPA — Method Guide")
st.caption("The 4-step funnel the cockpit implements. Educational reference only — "
           "not financial advice.")

# minervini_sepa_system.md lives at src/stock_screener/ (two levels up from pages/)
md_path = Path(__file__).resolve().parents[2] / "minervini_sepa_system.md"
try:
    text = md_path.read_text(encoding="utf-8")
    # The page already shows an H1 title above, so drop the doc's leading "# ..." line.
    lines = text.splitlines()
    if lines and lines[0].startswith("# "):
        text = "\n".join(lines[1:])
    st.markdown(text)
except FileNotFoundError:
    st.error(f"Guide source not found at {md_path}. Expected "
             "src/stock_screener/minervini_sepa_system.md.")

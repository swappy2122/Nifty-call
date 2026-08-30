"""
═══════════════════════════════════════════════════════════════════════════════
 NIFTY & BANKNIFTY INSTITUTIONAL ALGO RADAR  v5.0
 Streamlit Entrypoint for Cloud Deployment
═══════════════════════════════════════════════════════════════════════════════
 Renders the luxury Bloomberg/Apple Dark Terminal (Tailwind + TradingView Charts)
 seamlessly inside Streamlit Cloud & local Streamlit runners.

 Run: streamlit run app.py
═══════════════════════════════════════════════════════════════════════════════
"""

import streamlit as st
import streamlit.components.v1 as components
from main import HTML_TEMPLATE

st.set_page_config(
    page_title="Algo Radar v5 | NIFTY & BANKNIFTY Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Hide Streamlit header padding for clean full-screen Bloomberg experience
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    padding: 0 !important;
    background-color: #07090e !important;
}
[data-testid="stHeader"] {
    display: none !important;
}
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}
iframe {
    border: none !important;
}
</style>
""", unsafe_allow_html=True)

# Embed luxury Apple/Bloomberg Dark Terminal
components.html(HTML_TEMPLATE, height=1400, scrolling=True)

"""Reusable Streamlit theme helper for Applied Geostatistics Lab apps."""
import streamlit as st

THEMES = {
    "Geostatistics Light": {"bg":"#FFFFFF","text":"#172033","heading":"#123B6D","panel":"#F1F5F9","primary":"#2563EB","green":"#16A34A","orange":"#F59E0B","red":"#DC2626","border":"#D6E0EA"},
    "Geostatistics Dark": {"bg":"#0F172A","text":"#E5E7EB","heading":"#67E8F9","panel":"#1E293B","primary":"#38BDF8","green":"#4ADE80","orange":"#FBBF24","red":"#F87171","border":"#334155"},
    "Reservoir Green": {"bg":"#F7FBF7","text":"#17251B","heading":"#166534","panel":"#EAF4EC","primary":"#15803D","green":"#16A34A","orange":"#D97706","red":"#DC2626","border":"#CFE3D3"},
    "Sandstone": {"bg":"#FFF9EF","text":"#3F3426","heading":"#8A4F16","panel":"#F3E5CE","primary":"#B56A24","green":"#4D7C0F","orange":"#EA8A19","red":"#C24132","border":"#E2CBA8"},
    "High Contrast": {"bg":"#FFFFFF","text":"#000000","heading":"#000000","panel":"#EEEEEE","primary":"#0047FF","green":"#008A00","orange":"#B45309","red":"#C00000","border":"#000000"},
}

STATUS_COLORS = {
    "calculation": "primary", # Blue: calculations / general controls
    "correct": "green",      # Green: correct / acceptable
    "caution": "orange",     # Orange: caution / interpretation
    "error": "red",          # Red: errors / outliers / warnings
    "secondary": "panel",    # Light gray/panel: data & secondary controls
}

def theme_selector(label="Application theme"):
    """Place this near the top of the app. Returns the selected theme name."""
    return st.sidebar.selectbox(label, list(THEMES), index=0, key="geo_theme")

def apply_theme(theme_name="Geostatistics Light"):
    """Apply the selected visual theme to the current Streamlit page."""
    t = THEMES.get(theme_name, THEMES["Geostatistics Light"])
    st.markdown(f"""
    <style>
    :root {{
      --geo-bg: {t['bg']}; --geo-text: {t['text']}; --geo-heading: {t['heading']};
      --geo-panel: {t['panel']}; --geo-blue: {t['primary']}; --geo-green: {t['green']};
      --geo-orange: {t['orange']}; --geo-red: {t['red']}; --geo-border: {t['border']};
    }}
    .stApp {{ background-color: var(--geo-bg); color: var(--geo-text); }}
    h1, h2, h3, h4, h5, h6 {{ color: var(--geo-heading) !important; }}
    [data-testid="stSidebar"] {{ background-color: var(--geo-panel); }}
    [data-testid="stMetric"], [data-testid="stDataFrame"] {{
      background: var(--geo-panel); border: 1px solid var(--geo-border);
      border-radius: 10px; padding: 0.35rem;
    }}
    .stButton > button, .stDownloadButton > button {{
      border-color: var(--geo-blue); color: var(--geo-blue); border-radius: 8px;
    }}
    .stButton > button:hover, .stDownloadButton > button:hover {{
      border-color: var(--geo-blue); color: #FFFFFF; background: var(--geo-blue);
    }}
    .geo-panel {{ background: var(--geo-panel); border:1px solid var(--geo-border); padding:1rem; border-radius:10px; }}
    .geo-calc {{ color: var(--geo-blue); }} .geo-correct {{ color: var(--geo-green); }}
    .geo-caution {{ color: var(--geo-orange); }} .geo-error {{ color: var(--geo-red); }}
    </style>
    """, unsafe_allow_html=True)
    return t

def setup_theme():
    """Convenience call: selector + apply. Use after st.set_page_config()."""
    selected = theme_selector()
    return selected, apply_theme(selected)

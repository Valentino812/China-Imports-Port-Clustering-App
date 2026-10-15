"""Application entry point: global configuration and routing.

The three main pages are registered explicitly with :func:`st.navigation` using
``position="hidden"`` so that the default Streamlit page menu never shows the
Beranda and Input Data pages. The Dashboard renders its own fixed sidebar for
its five internal views.

Flow: Beranda -> Input Data -> Dashboard.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Klasterisasi Pelabuhan Impor Tiongkok",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session defaults
# ---------------------------------------------------------------------------
_DEFAULTS = {
    "combined_bytes": None,
    "combined_name": None,
    "combined_df": None,
    "combined_valid": False,
    "combined_message": None,
    "provinces_bytes": None,
    "provinces_name": None,
    "provinces_df": None,
    "provinces_valid": False,
    "provinces_message": None,
    "conversion_bytes": None,
    "conversion_name": None,
    "conversion_df": None,
    "conversion_valid": False,
    "conversion_message": None,
    "conversion_source": None,
    "n_clusters": 4,
    "analysis_result": None,
    "analysis_signature": None,
    "processing_error": None,
    "dashboard_view": "Ringkasan",
    "pipeline_summary": None,
}
for key, value in _DEFAULTS.items():
    st.session_state.setdefault(key, value)

# ---------------------------------------------------------------------------
# Global styling
# ---------------------------------------------------------------------------
BASE_CSS = r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*='css'], [data-testid='stAppViewContainer'] {
    font-family: 'Inter', sans-serif;
}
.stApp, [data-testid='stAppViewContainer'] {
    background: #FAF9F9;
    color: #1A1C1C;
}
/* Hilangkan bar hitam kosong di bagian atas konten. */
header[data-testid='stHeader'] { display: none !important; height: 0 !important; min-height: 0 !important; }
div[data-testid='stDecoration'], [data-testid='stToolbar'] { display: none !important; height: 0 !important; }
.block-container { padding-top: 1.5rem !important; padding-bottom: 2.5rem !important; max-width: 1500px; }

.app-topbar {
    background: #FFFFFF; border: 1px solid #E2E2E2; border-radius: 8px;
    padding: 14px 20px; margin: 0 0 22px 0;
    display: flex; align-items: center; justify-content: space-between; gap: 16px;
}
.app-topbar-title { font-size: 18px; font-weight: 700; color: #1E1E1E; line-height: 1.4; }
.app-topbar-subtitle { font-size: 12px; color: #6B6B6B; white-space: nowrap; }

[data-testid='stMetric'] { background: #FFFFFF; border: 1px solid #E2E2E2; border-radius: 8px; padding: 14px 16px; }
[data-testid='stMetricLabel'] { color: #666666; }
[data-testid='stMetricValue'] { color: #1A1C1C; }
div[data-testid='stVerticalBlockBorderWrapper'] { border-radius: 8px; }

.hero-panel { background: #FFFFFF; border: 1px solid #D8D8D8; border-radius: 10px; padding: clamp(26px, 5vw, 58px); text-align: center; margin: 8px 0 22px 0; }
.hero-eyebrow { color: #666666; letter-spacing: .18em; font-size: 11px; font-weight: 700; margin-bottom: 15px; }
.hero-title { color: #1E1E1E; font-size: clamp(28px, 3vw, 42px); line-height: 1.2; font-weight: 700; margin: 0 auto 18px auto; max-width: 880px; }
.hero-copy { color: #616161; font-size: 16px; line-height: 1.7; max-width: 820px; margin: 0 auto 24px auto; }
.small-note { color: #666666; font-size: 12px; }
.section-card { background: #FFFFFF; border: 1px solid #E2E2E2; border-radius: 8px; padding: 18px 20px; height: 100%; }
.section-card h4 { margin: 0 0 8px 0; font-size: 15px; color: #1E1E1E; }
.section-card p { margin: 0; color: #616161; font-size: 14px; line-height: 1.6; }
.file-ok { color: #0B7285; font-weight: 600; }
.file-missing { color: #C92A2A; font-weight: 600; }

/* Secara bawaan sidebar disembunyikan; hanya Dashboard yang menampilkannya. */
[data-testid='stSidebar'] { display: none !important; }
[data-testid='stSidebarCollapsedControl'] { display: none !important; }

#MainMenu, footer { visibility: hidden; }
</style>
"""

st.markdown(BASE_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Page registration (hidden navigation)
# ---------------------------------------------------------------------------
pages = [
    st.Page("pages/1_Beranda.py", title="Beranda", default=True),
    st.Page("pages/2_Input_Data.py", title="Input Data"),
    st.Page("pages/3_Dashboard.py", title="Dashboard"),
]

navigation = st.navigation(pages, position="hidden")
navigation.run()

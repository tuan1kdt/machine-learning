"""Bài 3 — Demo Streamlit: phát hiện ransomware trên bộ dữ liệu BitcoinHeist.

Chạy:  streamlit run app.py   (từ thư mục proj1)
"""
from pathlib import Path

import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(page_title="BitcoinHeist — Ransomware Detection", page_icon="🛡️", layout="wide")

from Home import Home              # noqa: E402  (import sau set_page_config)
from Statistics import Statistics  # noqa: E402
from Models import Models          # noqa: E402
from FeatureSelection import FeatureSelection  # noqa: E402
from LinearClassification import LinearClassification  # noqa: E402
from Predict import Predict        # noqa: E402
from Team import Team              # noqa: E402

ROOT = Path(__file__).parent
st.markdown(f"<style>{(ROOT / 'styles.css').read_text()}</style>", unsafe_allow_html=True)

PAGES = {
    "Giới thiệu": ("house", Home),
    "Khám phá dữ liệu": ("bar-chart", Statistics),
    "So sánh mô hình": ("trophy", Models),
    "Chọn đặc trưng (Bài 2)": ("funnel", FeatureSelection),
    "Bài 2 — Phân lớp tuyến tính": ("bounding-box", LinearClassification),
    "Dự đoán": ("shield-check", Predict),
    "Nhóm thực hiện": ("people", Team),
}

with st.sidebar:
    st.image(str(ROOT / "logo.png"), width=150)
    selected = option_menu(
        menu_title="Menu",
        options=list(PAGES),
        icons=[icon for icon, _ in PAGES.values()],
        menu_icon="list",
        default_index=0,
    )
    st.caption("Học máy — Dự án giữa kỳ 2026 · Bài 3")

PAGES[selected][1]()

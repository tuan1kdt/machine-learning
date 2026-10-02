"""Hàm nạp dữ liệu / mô hình dùng chung (có cache)."""
import pandas as pd
import streamlit as st

from features import ROOT, load_artifact

DATA = ROOT / "data"
CLASSES = ["montrealCryptXXX", "montrealCryptoLocker", "otherRansom", "paduaCryptoWall",
           "princetonCerber", "princetonLocky", "white"]
CLASS_COLORS = {
    "white": "#9aa0a6",
    "princetonLocky": "#e45756",
    "princetonCerber": "#f58518",
    "paduaCryptoWall": "#4c78a8",
    "montrealCryptoLocker": "#b279a2",
    "montrealCryptXXX": "#54a24b",
    "otherRansom": "#eeca3b",
}
MODEL_ORDER = ["LogisticRegression", "KNN", "RandomForest", "HistGradientBoosting", "LightGBM", "XGBoost", "Ensemble"]


@st.cache_data
def test_sample() -> pd.DataFrame:
    return pd.read_csv(DATA / "test_sample.csv.gz")


@st.cache_data
def label_counts() -> pd.DataFrame:
    return pd.read_csv(DATA / "label_counts.csv")


@st.cache_data
def year_target_counts() -> pd.DataFrame:
    return pd.read_csv(DATA / "year_target_counts.csv")


@st.cache_data
def summary() -> pd.DataFrame:
    s = pd.read_csv(DATA / "results" / "summary.csv")
    return s.set_index("model").loc[[m for m in MODEL_ORDER if m in set(s["model"])]].reset_index()


@st.cache_data
def per_class() -> pd.DataFrame:
    return pd.read_csv(DATA / "results" / "per_class.csv")


@st.cache_resource(show_spinner="Đang nạp mô hình...")
def artifact(name: str) -> dict:
    return load_artifact(name)


def card(title: str, body: str = "") -> None:
    st.markdown(f'<div class="card"><div class="card-title">{title}</div>{body}</div>', unsafe_allow_html=True)

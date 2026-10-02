"""Tạo đặc trưng & dự đoán — giống hệt pipeline của Bài 1 (bai1.ipynb, mục 2, 3, 11)."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
MODEL_DIRS = [ROOT / "models", ROOT.parent / "outputs" / "models"]   # ưu tiên proj1/models

WHITE = "white"
RAW_COLS = ["year", "day", "length", "weight", "count", "looped", "neighbors", "income"]
NUM_BASE = ["length", "weight", "count", "looped", "neighbors", "income"]
NUM = NUM_BASE + ["income_tz", "income_per_count", "income_per_neighbor", "weight_per_count", "looped_ratio"]
CAT = ["year", "month", "weekday"]
MODEL_NAMES = ["XGBoost", "Ensemble", "HistGradientBoosting", "LightGBM", "RandomForest", "KNN", "LogisticRegression"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Thêm month, weekday và 5 đặc trưng mới từ 8 cột gốc."""
    df = df.copy()
    date = pd.to_datetime(df["year"].astype(int).astype(str) + "-01-01") + pd.to_timedelta(df["day"] - 1, unit="D")
    df["month"] = date.dt.month
    df["weekday"] = date.dt.dayofweek

    inc_int = df["income"].round().astype("int64")
    df["income_tz"] = sum((inc_int % 10**k == 0).astype("int8") for k in range(1, 13))
    df["income_per_count"] = df["income"] / df["count"]
    df["income_per_neighbor"] = df["income"] / df["neighbors"].clip(lower=1)
    df["weight_per_count"] = df["weight"] / df["count"]
    df["looped_ratio"] = df["looped"] / df["count"]
    return df


def model_path(name: str) -> Path | None:
    return next((d / f"{name}.joblib" for d in MODEL_DIRS if (d / f"{name}.joblib").exists()), None)


def available_models() -> list[str]:
    out = []
    for n in MODEL_NAMES:
        if not model_path(n):
            continue
        if n == "Ensemble" and not all(model_path(c) for c in joblib.load(model_path(n))["components"]):
            continue
        out.append(n)
    return out


def load_artifact(name: str) -> dict:
    art = joblib.load(model_path(name))
    if "components" in art:                                   # Ensemble = trung bình xác suất các mô hình con
        art["pipelines"] = [joblib.load(model_path(c))["pipeline"] for c in art["components"]]
    return art


def predict(art: dict, raw: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trả về (xác suất gốc p, điểm đã hiệu chỉnh p·w chuẩn hoá về tổng 1, chỉ số lớp dự đoán = argmax(p·w))."""
    X = add_features(raw)[NUM + CAT]
    pipes = art.get("pipelines") or [art["pipeline"]]
    P = np.mean([p.predict_proba(X) for p in pipes], axis=0)
    S = P * art["class_weights"]
    S = S / S.sum(axis=1, keepdims=True)
    return P, S, S.argmax(1)

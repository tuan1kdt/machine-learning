import time

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from common import CLASS_COLORS, artifact, test_sample
from features import NUM, RAW_COLS, WHITE, add_features, available_models, predict

DEFAULTS = {"year": 2016, "day": 120, "length": 0, "weight": 1.0, "count": 1, "looped": 0, "neighbors": 2,
            "income": 100_000_000.0}


def _init_state() -> None:
    for k, v in DEFAULTS.items():
        st.session_state.setdefault(f"in_{k}", v)
    st.session_state.setdefault("true_label", None)


def _fill_random(cls: str) -> None:
    df = test_sample()
    pool = df if cls == "Bất kỳ" else df[df["target"] == cls]
    row = pool.sample(1).iloc[0]
    for k in RAW_COLS:
        v = row[k]
        st.session_state[f"in_{k}"] = float(v) if k in ("weight", "income") else int(v)
    st.session_state["true_label"] = (row["target"], row["label"], row["address"],
                                      {k: st.session_state[f"in_{k}"] for k in RAW_COLS})


def _reset() -> None:
    for k, v in DEFAULTS.items():
        st.session_state[f"in_{k}"] = v
    st.session_state["true_label"] = None


def _verdict(cls: str, score: float) -> None:
    is_white = cls == WHITE
    color = "#2e7d32" if is_white else "#c62828"
    title = "✅ Địa chỉ bình thường" if is_white else "🚨 Nghi ngờ ransomware"
    st.markdown(f'<div class="verdict" style="background:{color}"><div>{title}</div>'
                f'<div class="big">{cls}</div><div>Điểm tin cậy (đã hiệu chỉnh): {score:.1%}</div></div>',
                unsafe_allow_html=True)


def _single(art: dict, classes: list[str]) -> None:
    _init_state()
    a, b, c = st.columns([1.4, 1, 1], vertical_alignment="bottom")
    cls = a.selectbox("Lấy ngẫu nhiên 1 dòng thật từ tập test, thuộc lớp", ["Bất kỳ"] + classes)
    b.button("🎲 Lấy mẫu ngẫu nhiên", on_click=_fill_random, args=(cls,), width="stretch")
    c.button("↺ Đặt lại", on_click=_reset, width="stretch")

    with st.form("single"):
        r1 = st.columns(4)
        r1[0].selectbox("year", list(range(2011, 2019)), key="in_year")
        r1[1].number_input("day (1–365)", 1, 366, step=1, key="in_day")
        r1[2].number_input("length", 0, step=1, key="in_length")
        r1[3].number_input("weight", 0.0, format="%.6f", key="in_weight")
        r2 = st.columns(4)
        r2[0].number_input("count", 1, step=1, key="in_count")
        r2[1].number_input("looped", 0, step=1, key="in_looped")
        r2[2].number_input("neighbors", 1, step=1, key="in_neighbors")
        r2[3].number_input("income (satoshi)", 0.0, step=1e6, format="%.0f", key="in_income",
                           help="1 BTC = 100.000.000 satoshi")
        submitted = st.form_submit_button("🔍 Dự đoán", type="primary", width="stretch")

    raw = pd.DataFrame([{k: st.session_state[f"in_{k}"] for k in RAW_COLS}])
    st.caption(f"income = {raw['income'].iloc[0] / 1e8:,.8f} BTC")
    true = st.session_state["true_label"]
    if true and true[3] != raw.iloc[0].to_dict():      # người dùng đã sửa input → không còn là dòng thật
        true = None
    if true:
        st.caption(f"Dòng đang hiển thị lấy từ tập test — địa chỉ `{true[2]}`, nhãn gốc `{true[1]}`")

    if not submitted:
        st.info("Nhập thông tin hoặc bấm **Lấy mẫu ngẫu nhiên**, sau đó bấm **Dự đoán**.")
        return

    t0 = time.perf_counter()
    P, S, idx = predict(art, raw)
    dt = time.perf_counter() - t0
    pred = classes[idx[0]]

    left, right = st.columns([1, 1.4], gap="large")
    with left:
        _verdict(pred, S[0, idx[0]])
        if true:
            ok = true[0] == pred
            (st.success if ok else st.error)(f"Nhãn thật: **{true[0]}** → dự đoán {'ĐÚNG' if ok else 'SAI'}")
        st.caption(f"Thời gian dự đoán: {dt * 1000:.1f} ms")
        with st.expander("Đặc trưng đưa vào mô hình"):
            feats = add_features(raw)[NUM + ["month", "weekday"]].T.rename(columns={0: "giá trị"})
            st.dataframe(feats, width="stretch")
    with right:
        res = pd.DataFrame({"class": classes, "Điểm đã hiệu chỉnh p·w": S[0], "Xác suất gốc p": P[0]})
        view = st.radio("Hiển thị", list(res.columns[1:]), horizontal=True,
                        help="Dự đoán = argmax(p·w); w là trọng số lớp tìm trên validation để bù cho undersampling")
        fig = px.bar(res.sort_values(view), x=view, y="class", orientation="h", color="class",
                     color_discrete_map=CLASS_COLORS, text_auto=".1%", range_x=[0, 1])
        fig.update_layout(showlegend=False, height=360, xaxis_tickformat=".0%")
        st.plotly_chart(fig, width="stretch")


def _batch(art: dict, classes: list[str]) -> None:
    src = st.radio("Nguồn dữ liệu", ["Mẫu từ tập test", "Tải lên file CSV"], horizontal=True)
    if src == "Tải lên file CSV":
        st.caption("File cần các cột: " + ", ".join(f"`{c}`" for c in RAW_COLS) +
                   ". Nếu có cột `label` (nhãn gốc) hoặc `target` (7 lớp), demo sẽ tính độ đo.")
        up = st.file_uploader("Chọn file CSV", type=["csv"])
        if up is None:
            return
        df = pd.read_csv(up)
        missing = [c for c in RAW_COLS if c not in df.columns]
        if missing:
            st.error(f"Thiếu cột: {missing}")
            return
    else:
        full = test_sample()
        a, b = st.columns(2)
        n = a.slider("Số dòng", 100, len(full), 5000, step=100)
        frac = b.slider("Tỉ lệ ransomware trong mẫu", 0.0, 1.0, 0.3, 0.05,
                        help="Tập test thật chỉ có 1,4% ransomware; tăng tỉ lệ để thấy rõ các lớp ransomware")
        n_r = min(int(n * frac), int((full["target"] != WHITE).sum()))
        df = pd.concat([full[full["target"] != WHITE].sample(n_r, random_state=0),
                        full[full["target"] == WHITE].sample(n - n_r, random_state=0)]).sample(frac=1, random_state=0)

    if "target" not in df.columns and "label" in df.columns:
        top = set(classes) - {WHITE, "otherRansom"}
        df["target"] = np.where(df["label"] == WHITE, WHITE,
                                np.where(df["label"].isin(top), df["label"], "otherRansom"))

    t0 = time.perf_counter()
    _, S, idx = predict(art, df[RAW_COLS])
    dt = time.perf_counter() - t0
    out = df.copy()
    out["prediction"] = np.array(classes)[idx]
    out["score"] = S[np.arange(len(S)), idx]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Số dòng", f"{len(out):,}")
    c2.metric("Dự đoán ransomware", f"{(out['prediction'] != WHITE).sum():,}")
    c3.metric("Thời gian dự đoán", f"{dt:.2f}s")
    c4.metric("Tốc độ", f"{len(out) / max(dt, 1e-9):,.0f} dòng/s")

    if "target" in out.columns:
        y, p = out["target"], out["prediction"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{(y == p).mean():.4f}")
        c2.metric("Weighted F1", f"{f1_score(y, p, labels=classes, average='weighted', zero_division=0):.4f}")
        c3.metric("Macro F1", f"{f1_score(y, p, labels=classes, average='macro', zero_division=0):.4f}")
        c4.metric("Binary F1 (ransomware)", f"{f1_score(y != WHITE, p != WHITE, zero_division=0):.4f}")

        a, b = st.columns([1.2, 1], gap="large")
        with a:
            norm = st.toggle("Chuẩn hoá theo hàng (recall)", value=True)
            cm = confusion_matrix(y, p, labels=classes)
            z = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) if norm else cm
            fig = px.imshow(z, x=classes, y=classes, text_auto=".2f" if norm else "d", color_continuous_scale="Blues",
                            labels={"x": "Dự đoán", "y": "Thực tế"}, aspect="auto", title="Ma trận nhầm lẫn", height=520)
            st.plotly_chart(fig, width="stretch")
        with b:
            rep = classification_report(y, p, labels=classes, output_dict=True, zero_division=0)
            rep = pd.DataFrame(rep).T.loc[classes + ["macro avg", "weighted avg"]]
            st.dataframe(rep.style.format({"precision": "{:.3f}", "recall": "{:.3f}", "f1-score": "{:.3f}",
                                           "support": "{:.0f}"}), width="stretch")
        if src == "Mẫu từ tập test":
            st.caption("Lưu ý: mẫu đã được tăng tỉ lệ ransomware nên các độ đo khác với kết quả trên toàn tập test "
                       "(xem trang So sánh mô hình).")
    else:
        vc = out["prediction"].value_counts().rename_axis("class").reset_index(name="count")
        fig = px.bar(vc, x="class", y="count", color="class", color_discrete_map=CLASS_COLORS,
                     title="Phân phối dự đoán")
        st.plotly_chart(fig, width="stretch")

    st.dataframe(out.head(1000), width="stretch")
    st.download_button("⬇️ Tải kết quả (CSV)", out.to_csv(index=False).encode("utf-8"),
                       "predictions.csv", "text/csv")


def Predict() -> None:
    st.header("🔮 Dự đoán")
    models = available_models()
    if not models:
        st.error("Không tìm thấy mô hình. Chạy `python prepare_data.py` hoặc notebook Bài 1 trước.")
        return

    a, b = st.columns([1, 2.2], vertical_alignment="bottom")
    name = a.selectbox("Mô hình", models, index=0)
    try:
        art = artifact(name)
    except Exception as e:  # vd thiếu libomp cho XGBoost/LightGBM trên macOS
        st.error(f"Không nạp được mô hình {name}: {e}")
        return
    m = art.get("metrics_test", {})
    if m:
        b.markdown(" ".join(f'<span class="pill">{k}: <b>{m[v]:.3f}</b></span>' for k, v in
                            [("Macro F1 (test)", "macro_f1"), ("Weighted F1", "weighted_f1"),
                             ("Accuracy", "accuracy"), ("Binary F1", "binary_f1_ransom")]),
                   unsafe_allow_html=True)
    classes = list(art["classes"])

    tab1, tab2 = st.tabs(["Một địa chỉ", "Hàng loạt (CSV)"])
    with tab1:
        _single(art, classes)
    with tab2:
        _batch(art, classes)

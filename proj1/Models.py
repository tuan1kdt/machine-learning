import pandas as pd
import plotly.express as px
import streamlit as st

from common import CLASSES, MODEL_ORDER, per_class, summary, year_target_counts

METRICS = {"accuracy": "Accuracy", "weighted_precision": "Weighted precision", "weighted_recall": "Weighted recall",
           "weighted_f1": "Weighted F1", "macro_f1": "Macro F1", "binary_f1_ransom": "Binary F1 (ransomware)"}


def Models() -> None:
    st.header("🏆 So sánh mô hình (kết quả Bài 1)")
    st.caption("Tập test 583.340 dòng, giữ phân phối thật (98,6% white). Mô hình train trên tập train đã "
               "undersample (100k white) và được hiệu chỉnh trọng số lớp trên validation.")

    s = summary()
    best = s.loc[s["model"] != "Ensemble"].sort_values("macro_f1", ascending=False).iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mô hình đơn tốt nhất", best["model"])
    c2.metric("Macro F1", f"{best['macro_f1']:.4f}")
    c3.metric("Weighted F1", f"{best['weighted_f1']:.4f}")
    c4.metric("Thời gian test", f"{best['test_time_s']:.2f}s")

    # ---------------- Bảng tổng hợp
    st.subheader("Độ đo tổng hợp & thời gian")
    yt = year_target_counts()
    p = yt.loc[yt["target"] == "white", "count"].sum() / yt["count"].sum()   # tỉ lệ white (stratified → như test)
    f1_white = 2 * p / (1 + p)
    baseline_macro = f1_white / len(CLASSES)
    baseline = pd.DataFrame([{"model": "Luôn đoán white (mốc)", "accuracy": p, "weighted_precision": p * p,
                              "weighted_recall": p, "weighted_f1": p * f1_white, "macro_f1": baseline_macro,
                              "binary_f1_ransom": 0.0}])
    show = pd.concat([s, baseline], ignore_index=True)
    st.dataframe(
        show.style.format({c: "{:.4f}" for c in METRICS} |
                          {c: lambda v: "–" if pd.isna(v) else f"{v:.2f}" for c in ["train_time_s", "test_time_s"]})
            .background_gradient(cmap="Greens", subset=["macro_f1", "binary_f1_ransom"])
            .background_gradient(cmap="Reds", subset=["train_time_s", "test_time_s"]),
        hide_index=True, width="stretch")

    a, b = st.columns(2, gap="large")
    with a:
        chosen = st.multiselect("Độ đo", list(METRICS), default=["accuracy", "weighted_f1", "macro_f1"],
                                format_func=METRICS.get)
        long = s.melt(id_vars="model", value_vars=chosen, var_name="metric", value_name="value")
        long["metric"] = long["metric"].map(METRICS)
        fig = px.bar(long, x="model", y="value", color="metric", barmode="group", range_y=[0, 1.05],
                     category_orders={"model": MODEL_ORDER}, title="Độ đo trên tập test")
        fig.add_hline(y=baseline_macro, line_dash="dash", line_color="gray",
                      annotation_text="macro F1 của 'luôn đoán white'")
        st.plotly_chart(fig, width="stretch")
    with b:
        log_y = st.toggle("Thang log", value=True)
        t = s[s["model"] != "Ensemble"].melt(id_vars="model", value_vars=["train_time_s", "test_time_s"],
                                             var_name="phase", value_name="seconds")
        t["phase"] = t["phase"].map({"train_time_s": "Training", "test_time_s": "Testing"})
        fig = px.bar(t, x="model", y="seconds", color="phase", barmode="group", log_y=log_y,
                     category_orders={"model": MODEL_ORDER}, title="Thời gian training / testing (giây)")
        st.plotly_chart(fig, width="stretch")

    st.info("**Accuracy & weighted F1 không phù hợp** với dữ liệu mất cân bằng: mô hình 'luôn đoán white' có "
            "accuracy 0,986 nhưng không phát hiện được ransomware nào. **Macro F1** (trung bình F1 các lớp, "
            "không trọng số) phản ánh đúng khả năng nhận diện các họ ransomware.")

    # ---------------- Theo từng lớp
    st.subheader("Precision / Recall / F1 của từng lớp")
    pc = per_class()
    a, b = st.columns([1, 1.3], gap="large")
    with a:
        model = st.selectbox("Mô hình", [m for m in MODEL_ORDER if m in set(pc["model"])],
                             index=MODEL_ORDER.index("XGBoost"))
        m = pc[pc["model"] == model].drop(columns="model").set_index("class").loc[CLASSES]
        st.dataframe(m.style.format("{:.3f}").background_gradient(cmap="Greens", vmin=0, vmax=1),
                     width="stretch")
        fig = px.bar(m.reset_index().melt(id_vars="class", var_name="metric"), x="class", y="value",
                     color="metric", barmode="group", range_y=[0, 1], title=model)
        st.plotly_chart(fig, width="stretch")
    with b:
        metric = st.radio("Độ đo", ["f1-score", "precision", "recall"], horizontal=True)
        heat = pc.pivot(index="model", columns="class", values=metric).loc[
            [x for x in MODEL_ORDER if x in set(pc["model"])], CLASSES]
        fig = px.imshow(heat, text_auto=".2f", color_continuous_scale="Greens", zmin=0, zmax=1, aspect="auto",
                        title=f"{metric} theo lớp × mô hình", height=520)
        st.plotly_chart(fig, width="stretch")

    st.markdown("""
**Nhận xét**
1. Các mô hình **boosting** tốt nhất; **XGBoost** là mô hình đơn tốt nhất và dự đoán nhanh. Ensemble chỉ nhỉnh hơn
   rất ít nhưng tốn gấp ~4 lần thời gian.
2. **Logistic Regression** và **KNN** yếu hơn rõ rệt: ranh giới giữa các lớp phi tuyến, các họ ransomware chồng lấn.
3. **KNN** gần như không tốn thời gian train nhưng chậm nhất khi test (tính khoảng cách tới toàn bộ tập train).
4. Nhận diện tốt **CryptXXX, Locky, Cerber**; khó với **CryptoLocker** và **otherRansom** (gộp nhiều họ hiếm).
""")

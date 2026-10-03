"""Bài 2 (bản phân lớp) — Chọn đặc trưng dựa trên tương quan cho bộ phân lớp tuyến tính.

Linear Regression trên nhãn one-hot (7 lớp của Bài 1) → argmax, so sánh các tập đặc trưng bằng MAE (+ accuracy,
macro F1). Dữ liệu data/bai2_cls/ do notebook ../bai2_classification.ipynb tạo ra (ô "Xuất kết quả cho demo").
"""
import time

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import CLASS_COLORS, CLASSES, DATA
from FeatureSelection import _delta, _pills, drop_redundant

B2C = DATA / "bai2_cls"
FILES = ["corr_classes.csv", "corr_matrix.csv", "topk.csv", "subsets.csv", "cross.csv", "sample.csv.gz"]
K = len(CLASSES)
GOAL = 90            # k nhỏ nhất đạt ≥ 90% mức cải thiện MAE (như notebook)
RED_THR = 0.95       # ngưỡng loại dư thừa của tập đề xuất (như notebook)
BLUE, RED, ORANGE, GRAY, PURPLE = "#0068c9", "#e45756", "#f58518", "#9aa0a6", "#8e6bbf"
LIN, LOG = "Linear Regression (one-hot + argmax)", "Logistic Regression (softmax)"
METHODS = ["Top-k theo R", "Ngưỡng R", "Tự chọn"]


@st.cache_data(show_spinner="Đang nạp dữ liệu Bài 2 (phân lớp)...")
def _read(name: str, index_col: int | None = None) -> pd.DataFrame:
    return pd.read_csv(B2C / name, index_col=index_col)


@st.cache_data(show_spinner=False)
def _split() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = _read("sample.csv.gz")
    df["y"] = df["target"].map({c: i for i, c in enumerate(CLASSES)})
    return df[df["split"] == "train"], df[df["split"] == "test"]


def _make(model: str) -> Pipeline:
    clf = LinearRegression() if model == LIN else LogisticRegression(max_iter=2000)
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


@st.cache_resource(show_spinner=False)
def _train(cols: tuple[str, ...], model: str) -> tuple[Pipeline, float]:
    """Train trên toàn bộ tập train cân bằng của notebook. Linear Regression học 7 đầu ra one-hot."""
    tr, _ = _split()
    pipe = _make(model)
    target = np.eye(K)[tr["y"]] if model == LIN else tr["y"]
    t0 = time.perf_counter()
    pipe.fit(tr[list(cols)], target)
    return pipe, time.perf_counter() - t0


def _scores(pipe: Pipeline | None, X: pd.DataFrame) -> np.ndarray:
    """Điểm 7 lớp: ŝ_k của Linear Regression hoặc xác suất của Logistic Regression.
    pipe = None → baseline: điểm = tỉ lệ các lớp trên tập train."""
    if pipe is None:
        tr, _ = _split()
        return np.tile(np.bincount(tr["y"], minlength=K) / len(tr), (len(X), 1))
    return pipe.predict(X) if isinstance(pipe.named_steps["clf"], LinearRegression) else pipe.predict_proba(X)


@st.cache_data(show_spinner=False)
def _evaluate(cols: tuple[str, ...], model: str) -> dict:
    _, te = _split()
    pipe, t_train = _train(cols, model) if cols else (None, 0.0)
    S = _scores(pipe, te[list(cols)])
    pred = S.argmax(axis=1)
    coef = (pd.DataFrame(np.atleast_2d(pipe.named_steps["clf"].coef_).T, index=list(cols), columns=CLASSES)
            if pipe is not None else pd.DataFrame())
    return {"n": len(cols), "MAE": mean_absolute_error(np.eye(K)[te["y"]], S),
            "accuracy": accuracy_score(te["y"], pred), "macro_f1": f1_score(te["y"], pred, average="macro"),
            "time": t_train, "coef": coef,
            "cm": confusion_matrix(te["y"], pred, labels=range(K), normalize="true")}


def _proposed(ranking: list[str], cm: pd.DataFrame, k_best: int) -> list[str]:
    return drop_redundant(ranking[:k_best], cm, RED_THR)[0]


# ---------------- Tab "Mô hình & dữ liệu"
def _model_tab(ranking: list[str], cm: pd.DataFrame, k_best: int) -> None:
    a, b = st.columns([1.15, 1], gap="large")
    with a:
        st.markdown(r"""
##### Linear Regression như một bộ phân lớp tuyến tính
1. **Mã hoá one-hot:** nhãn $y = k$ → vector $\mathbf{Y} \in \{0,1\}^7$ với $Y_k = 1$.
2. **Huấn luyện:** `LinearRegression` hồi quy đồng thời 7 đầu ra (bình phương tối thiểu):
   $\hat s_k(\mathbf{x}) = w_{k0} + \mathbf{w}_k^\top\mathbf{x}$ — xấp xỉ tuyến tính của $P(y = k \mid \mathbf{x})$.
3. **Dự đoán:** $\hat y = \arg\max_k \hat s_k(\mathbf{x})$. Ranh giới giữa hai lớp $\hat s_j = \hat s_k$ là một
   **siêu phẳng** → bộ phân lớp **tuyến tính** (chính là `RidgeClassifier` khi hệ số L2 → 0).
4. **MAE** giữa điểm và nhãn one-hot: $\text{MAE} = \frac{1}{7n}\sum_i\sum_k |Y_{ik} - \hat s_k(\mathbf{x}_i)|$
   (càng nhỏ càng tốt); kèm **accuracy** và **macro F1** như Bài 1.
""")
        st.info("**Vì sao dùng tập cân bằng?** Trên phân phối thật (98,6% `white`), bộ phân lớp bình phương tối thiểu "
                "dự đoán `white` cho **100%** dòng test (macro F1 = 0,142) với mọi tập đặc trưng → không so sánh được. "
                "Notebook dùng **toàn bộ 41.413 dòng ransomware + 41.413 dòng `white`** lấy ngẫu nhiên, chia 80/20 "
                "(stratified). Tương quan chỉ tính trên tập train.")
    with b:
        tr, te = _split()
        counts = (pd.concat([tr["target"].value_counts().rename("train"), te["target"].value_counts().rename("test")],
                            axis=1).rename_axis("lớp").reset_index().melt("lớp", var_name="tập", value_name="số dòng"))
        fig = px.bar(counts, x="số dòng", y="lớp", color="tập", orientation="h", barmode="group", text_auto=True,
                     color_discrete_map={"train": BLUE, "test": ORANGE},
                     category_orders={"lớp": tr["target"].value_counts().index.tolist()},
                     title=f"Tập thử nghiệm cân bằng — {len(tr):,} train / {len(te):,} test", height=380)
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_yaxes(title=None)
        st.plotly_chart(fig, width="stretch")

    st.markdown("##### Minh hoạ: 7 điểm của một dòng test → lớp dự đoán")
    feats = _proposed(ranking, cm, k_best)
    _, te = _split()
    if "lc_row" not in st.session_state:
        st.session_state["lc_row"] = 0
    a, b = st.columns([1, 2], gap="large")
    with a:
        only = st.selectbox("Lấy dòng thuộc lớp", ["(bất kỳ)"] + CLASSES, key="lc_row_class")
        if st.button("🎲 Lấy ngẫu nhiên một dòng test", width="stretch"):
            pool = te.index if only == "(bất kỳ)" else te.index[te["target"] == only]
            st.session_state["lc_row"] = int(np.random.default_rng().choice(pool))
        idx = st.session_state["lc_row"] if st.session_state["lc_row"] in te.index else te.index[0]
        row = te.loc[[idx]]
        st.caption(f"Mô hình: Linear Regression trên **tập đề xuất** ({len(feats)} đặc trưng, train trên tập train "
                   "cân bằng). Đặc trưng số lệch mạnh đã `log1p`.")
        st.dataframe(row[feats].T.rename(columns={idx: "giá trị"}).style.format("{:.4g}"), width="stretch")
    with b:
        pipe, _ = _train(tuple(feats), LIN)
        s = _scores(pipe, row[feats])[0]
        pred, true = CLASSES[int(s.argmax())], row["target"].iloc[0]
        d = pd.DataFrame({"lớp": CLASSES, "điểm ŝ_k": s}).sort_values("điểm ŝ_k")
        fig = px.bar(d, x="điểm ŝ_k", y="lớp", orientation="h", color="lớp", text_auto=".3f",
                     color_discrete_map=CLASS_COLORS, height=360,
                     title="Điểm Linear Regression của từng lớp (có thể < 0 hoặc > 1 — không phải xác suất thật)")
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(showlegend=False)
        fig.update_yaxes(title=None)
        fig.add_vline(x=0, line_color=GRAY)
        st.plotly_chart(fig, width="stretch")
        verdict = "✅ Đúng" if pred == true else "❌ Sai"
        st.markdown(f"**Dự đoán (argmax):** `{pred}` · **Nhãn thật:** `{true}` → {verdict}")


# ---------------- Tab "Tương quan"
def _correlation(cc: pd.DataFrame, cm: pd.DataFrame, by_k: pd.DataFrame) -> None:
    ranking = cc.index.tolist()
    a, b = st.columns([1.6, 1], gap="large")
    with a:
        fig = px.imshow(cc[CLASSES], text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-0.45, zmax=0.45,
                        aspect="auto", height=560,
                        labels={"x": "lớp", "y": "đặc trưng", "color": "r"},
                        title="Tương quan point-biserial r(đặc trưng, 1[y = lớp]) — tập train, xếp theo R giảm dần")
        st.plotly_chart(fig, width="stretch")
    with b:
        thr = st.slider("Ngưỡng R", 0.0, 0.25, 0.05, 0.005, format="%.3f", key="lc_corr_thr",
                        help="Giữ các đặc trưng có điểm tương quan tổng hợp R ≥ ngưỡng")
        sel = cc["R"] >= thr
        k = int(sel.sum())
        d = cc.reset_index().assign(**{"Trạng thái": np.where(sel.values, "Được chọn", "Bị loại")})
        fig = px.bar(d, x="R", y="feature", orientation="h", color="Trạng thái", text_auto=".3f",
                     color_discrete_map={"Được chọn": BLUE, "Bị loại": GRAY},
                     category_orders={"feature": ranking, "Trạng thái": ["Được chọn", "Bị loại"]},
                     hover_data={"eta": ":.3f", "rank": True}, labels={"feature": "đặc trưng", "eta": "η"},
                     title="R = √(trung bình r² theo 7 lớp)", height=420, range_x=[0, 1.25 * cc["R"].max()])
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_yaxes(title=None)
        fig.update_layout(legend={"orientation": "h", "y": -0.18, "title": None})
        fig.add_vline(x=thr, line_dash="dash", line_color=GRAY)
        st.plotly_chart(fig, width="stretch")
        st.markdown(f"**Được chọn: {k}/{len(cc)} đặc trưng** — tập này là Top-{k} theo R"
                    + (f": MAE (test) = {by_k.loc[k, 'MAE']:.4f}, macro F1 = {by_k.loc[k, 'macro_f1']:.3f}, "
                       f"đạt {by_k.loc[k, 'improvement_pct']:.1f}% mức cải thiện MAE." if k else "."))
    st.info("**Point-biserial** = Pearson giữa đặc trưng và biến chỉ thị 1[y = k]. |r| lớn ⇔ trung bình đặc trưng "
            "trong lớp k khác hẳn các lớp còn lại ⇔ đặc trưng giúp tách lớp k (đúng việc đầu ra ŝ_k phải làm). "
            "`income_tz` tách `white` (r = −0,41) khỏi ransomware — tiền chuộc thường là số **tròn**; `year` tách "
            "các **họ** ransomware với nhau (CryptoLocker 2013 ↔ Cerber/Locky 2016–2017). `count`, `length`, "
            "`weekday`… gần như không tách được lớp nào (R < 0,05).")

    st.markdown("##### Tương quan giữa các đặc trưng — phát hiện đặc trưng dư thừa")
    a, b = st.columns([1.6, 1], gap="large")
    with a:
        mask = np.triu(np.ones(cm.shape, dtype=bool), k=1)
        fig = px.imshow(cm.mask(mask), text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                        aspect="auto", height=600, title="Ma trận tương quan Pearson giữa các đặc trưng (tập train)")
        st.plotly_chart(fig, width="stretch")
    with b:
        feats = cm.columns.tolist()
        pairs = pd.DataFrame([(f, g, cm.loc[f, g]) for i, f in enumerate(feats) for g in feats[i + 1:]],
                             columns=["Đặc trưng 1", "Đặc trưng 2", "r"])
        st.markdown("**8 cặp đặc trưng tương quan mạnh nhất**")
        st.dataframe(pairs.sort_values("r", key=abs, ascending=False).head(8).style.format({"r": "{:+.3f}"}),
                     hide_index=True, width="stretch")
        st.info("`day` – `month` (r = 0,996): bỏ `day` thì kết quả **không đổi**. `income` – `income_per_neighbor` "
                "(0,95) tuy tương quan rất mạnh nhưng bỏ `income` làm macro F1 giảm 0,352 → 0,317 (mô hình dùng "
                "**hiệu** của hai đặc trưng) → ngưỡng dư thừa nên đặt **cao** (≈ 0,95).")

    st.markdown("##### R (point-biserial) và η (tỉ số tương quan — `f_classif` của scikit-learn)")
    long = cc.reset_index().melt(id_vars="feature", value_vars=["R", "eta"], var_name="hệ số", value_name="giá trị")
    long["hệ số"] = long["hệ số"].map({"R": "R (point-biserial)", "eta": "η (từ f_classif)"})
    fig = px.bar(long, x="feature", y="giá trị", color="hệ số", barmode="group", height=380,
                 category_orders={"feature": ranking},
                 color_discrete_map={"R (point-biserial)": BLUE, "η (từ f_classif)": ORANGE},
                 title="Hai thước đo tương quan đặc trưng – lớp")
    fig.update_xaxes(title=None)
    st.plotly_chart(fig, width="stretch")
    st.caption("η² = phương sai giữa các lớp / phương sai toàn phần; ANOVA F = (η²/(K−1)) / ((1−η²)/(n−K)) tăng theo η "
               "→ `SelectKBest(f_classif, k)` chọn k đặc trưng có η lớn nhất. Thứ hạng theo η gần như trùng R: "
               "Top-9 theo η và theo R là **cùng một tập**.")


# ---------------- Tab "Kết quả MAE"
def _results(tk: pd.DataFrame, sb: pd.DataFrame, cr: pd.DataFrame, k_best: int, prop: pd.Series) -> None:
    st.caption("Kết quả của notebook `bai2_classification.ipynb`: train 66.260 dòng, test 16.566 dòng (tập cân bằng).")
    n = int(tk["n_features"].max())
    base, full = tk.iloc[0], tk.iloc[-1]
    best = tk.set_index("n_features").loc[k_best]
    a, b = st.columns(2, gap="large")
    with a:
        fig = px.line(tk, x="n_features", y="MAE", markers=True, height=440,
                      hover_data={"added": True, "improvement_pct": ":.1f", "macro_f1": ":.3f"},
                      labels={"n_features": "k — số đặc trưng (thêm dần theo R giảm dần)",
                              "added": "đặc trưng thêm vào", "improvement_pct": "% cải thiện", "macro_f1": "macro F1"},
                      title="MAE trên tập test theo số đặc trưng (Top-k theo R)")
        fig.add_hline(y=base["MAE"], line_dash="dash", line_color=GRAY, annotation_position="bottom right",
                      annotation_text=f"Baseline: {base['MAE']:.4f}")
        fig.add_hline(y=full["MAE"], line_dash="dot", line_color=GRAY, annotation_position="top right",
                      annotation_text=f"Tất cả {n} đặc trưng: {full['MAE']:.4f}")
        fig.add_scatter(x=[k_best], y=[best["MAE"]], mode="markers", hoverinfo="skip", showlegend=False,
                        marker={"size": 20, "symbol": "circle-open", "color": ORANGE, "line": {"width": 3}})
        fig.add_annotation(x=k_best, y=best["MAE"], ax=0, ay=-90, arrowcolor=ORANGE,
                           text=f"<b>Top-{k_best}</b>: MAE {best['MAE']:.4f}<br>{best['improvement_pct']:.0f}% mức "
                                f"cải thiện với {100 * k_best / n:.0f}% số đặc trưng")
        fig.update_xaxes(dtick=1)
        st.plotly_chart(fig, width="stretch")
    with b:
        long = tk.melt(id_vars=["n_features", "added"], value_vars=["accuracy", "macro_f1"], var_name="độ đo",
                       value_name="giá trị")
        long["độ đo"] = long["độ đo"].map({"accuracy": "accuracy", "macro_f1": "macro F1"})
        fig = px.line(long, x="n_features", y="giá trị", color="độ đo", markers=True, height=440,
                      color_discrete_map={"accuracy": BLUE, "macro F1": ORANGE}, range_y=[0, 0.7],
                      hover_data={"added": True}, labels={"n_features": "k — số đặc trưng", "added": "thêm vào"},
                      title="Accuracy và macro F1 trên tập test theo số đặc trưng")
        fig.add_vline(x=k_best, line_dash="dot", line_color=RED)
        fig.update_xaxes(dtick=1)
        fig.update_layout(legend={"orientation": "h", "y": -0.2, "title": None})
        st.plotly_chart(fig, width="stretch")
    st.info(f"Chỉ `income_tz` đã đạt 71% mức cải thiện MAE (tách `white` khỏi ransomware) nhưng macro F1 vẫn bằng "
            f"baseline — cần thêm `year` để phân biệt các **họ** ransomware → MAE (trên điểm) và macro F1 (trên nhãn) "
            f"không phải lúc nào cũng cùng chiều. Từ k ≈ 7 cả hai **bão hoà**: Top-{k_best} đạt "
            f"{best['improvement_pct']:.0f}% mức cải thiện MAE và macro F1 {best['macro_f1']:.3f} so với "
            f"{full['macro_f1']:.3f} khi dùng cả {n} đặc trưng.")

    st.markdown("##### Tổng hợp các tập đặc trưng")
    show = (sb[["subset", "n_features", "MAE", "mae_vs_all", "accuracy", "macro_f1", "improvement_pct", "features"]]
            .fillna({"features": "—"})
            .rename(columns={"subset": "Tập đặc trưng", "n_features": "Số đặc trưng",
                             "mae_vs_all": "MAE chênh so với tất cả", "accuracy": "Accuracy", "macro_f1": "Macro F1",
                             "improvement_pct": "% cải thiện MAE", "features": "Đặc trưng"}))
    st.dataframe(show.style.format({"MAE": "{:.4f}", "MAE chênh so với tất cả": "{:+.4f}", "Accuracy": "{:.3f}",
                                    "Macro F1": "{:.3f}", "% cải thiện MAE": "{:.1f}"})
                     .background_gradient(cmap="Greens_r", subset=["MAE"])
                     .background_gradient(cmap="Greens", subset=["Macro F1"]),
                 hide_index=True, width="stretch")

    a, b = st.columns([1.3, 1], gap="large")
    with a:
        long = cr.melt(id_vars=["subset", "model"], value_vars=["MAE", "macro_f1"], var_name="độ đo",
                       value_name="giá trị")
        long["độ đo"] = long["độ đo"].map({"MAE": "MAE (càng nhỏ càng tốt)", "macro_f1": "macro F1 (càng lớn càng tốt)"})
        fig = px.bar(long, x="giá trị", y="subset", color="model", facet_col="độ đo", orientation="h",
                     barmode="group", text_auto=".3f", height=430,
                     color_discrete_map={"LinearRegression": BLUE, "LogisticRegression": PURPLE},
                     category_orders={"subset": cr["subset"].drop_duplicates().tolist()},
                     title="Kiểm chứng chéo với một bộ phân lớp tuyến tính khác")
        fig.update_xaxes(matches=None, title=None)
        fig.update_yaxes(title=None)
        fig.for_each_annotation(lambda t: t.update(text=t.text.split("=")[-1]))
        fig.update_layout(legend={"orientation": "h", "y": -0.12, "title": None})
        st.plotly_chart(fig, width="stretch")
    with b:
        st.markdown(f"""
**Nhận xét**
1. **Tập đề xuất — Top-{k_best} rồi bỏ đặc trưng dư thừa:** {prop['n_features']} đặc trưng, MAE chỉ tăng
   {prop['mae_vs_all']:.4f} so với dùng cả {n}, macro F1 đạt {100 * prop['macro_f1'] / full['macro_f1']:.0f}%.
2. **Đối chứng:** các đặc trưng có R **thấp nhất** (chính là những đặc trưng bị bỏ) cho accuracy 0,500 và macro F1
   bằng baseline → mô hình vẫn luôn đoán `white`.
3. **Logistic Regression** cho cùng thứ tự các tập đặc trưng → kết quả chọn đặc trưng không phụ thuộc một mô hình
   cụ thể; với tập đề xuất, Logistic Regression train **nhanh khoảng 2–3 lần**.
4. **Hạn chế:** bộ phân lớp tuyến tính chỉ đạt accuracy ~0,6, macro F1 ~0,35–0,40 (XGBoost ở Bài 1 tốt hơn nhiều);
   cần tập cân bằng; tương quan chỉ xét từng cặp và chỉ đo quan hệ tuyến tính.
""")


# ---------------- Tab "Thử nghiệm trực tiếp"
def _live(cc: pd.DataFrame, cm: pd.DataFrame, k_best: int) -> None:
    ranking = cc.index.tolist()
    st.caption("Train lại ngay trong demo trên **đúng** tập train/test cân bằng của notebook "
               "(66.260 / 16.566 dòng) → số liệu khớp tab *Kết quả MAE*.")
    left, right = st.columns([1, 2], gap="large")
    with left:
        model = st.radio("Bộ phân lớp tuyến tính", [LIN, LOG], key="lc_model",
                         captions=["đúng yêu cầu đề bài (Linear Regression)", "để kiểm chứng chéo"])
        method = st.radio("Cách chọn đặc trưng", METHODS, key="lc_method",
                          captions=["k đặc trưng có R lớn nhất", "giữ các đặc trưng có R ≥ ngưỡng", "chọn tuỳ ý"])
        if method == METHODS[0]:
            chosen = ranking[:st.slider("k — số đặc trưng", 1, len(ranking), k_best, key="lc_k")]
        elif method == METHODS[1]:
            t = st.slider("Ngưỡng R", 0.0, 0.25, 0.05, 0.005, format="%.3f", key="lc_live_thr")
            chosen = [f for f in ranking if cc.loc[f, "R"] >= t]
        else:
            picked = st.multiselect("Đặc trưng", ranking, default=_proposed(ranking, cm, k_best), key="lc_custom")
            chosen = [f for f in ranking if f in picked]                      # sắp theo R giảm dần
        dedup = st.checkbox("Loại đặc trưng dư thừa", value=True, key="lc_dedup",
                            help="Duyệt theo R giảm dần; bỏ một đặc trưng nếu |r| của nó với một đặc trưng đã giữ ≥ "
                                 "ngưỡng (ma trận tương quan của tập train)")
        thr = st.slider("Ngưỡng |r| giữa hai đặc trưng", 0.5, 0.99, RED_THR, 0.01, key="lc_red_thr", disabled=not dedup)
        if dedup:
            chosen, removed = drop_redundant(chosen, cm, thr)
            if removed:
                st.markdown("**Đã loại:** " + "; ".join(f"`{f}` (r = {r:+.3f} với `{g}`)" for f, g, r in removed))
            else:
                st.caption("Không có cặp đặc trưng nào vượt ngưỡng → không loại đặc trưng nào.")
        st.markdown(f"**Dùng {len(chosen)}/{len(ranking)} đặc trưng:**")
        _pills(chosen)

    with right:
        if not chosen:
            st.warning("Chưa có đặc trưng nào được chọn — hãy giảm ngưỡng R hoặc chọn thêm đặc trưng.")
            return
        with st.spinner("Đang train..."):
            base, full, res = _evaluate((), model), _evaluate(tuple(ranking), model), _evaluate(tuple(chosen), model)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Số đặc trưng", f"{res['n']}/{full['n']}")
        c2.metric("MAE (test)", f"{res['MAE']:.4f}", _delta(res["MAE"] - full["MAE"]), delta_color="inverse",
                  help=f"Chênh so với tất cả {full['n']} đặc trưng (MAE {full['MAE']:.4f}); baseline {base['MAE']:.4f}")
        c3.metric("Accuracy", f"{res['accuracy']:.3f}", _delta(res["accuracy"] - full["accuracy"], "+.3f"))
        c4.metric("Macro F1", f"{res['macro_f1']:.3f}", _delta(res["macro_f1"] - full["macro_f1"], "+.3f"),
                  help=f"Tất cả {full['n']} đặc trưng: {full['macro_f1']:.3f}; baseline {base['macro_f1']:.3f}")
        c5.metric("Thời gian train", f"{1000 * res['time']:.0f} ms",
                  help=f"Tất cả {full['n']} đặc trưng: {1000 * full['time']:.0f} ms (đo ở lần train đầu tiên, "
                       "sau đó kết quả được cache)")

        rows = [("Baseline (không dùng đặc trưng)", base), (f"Tất cả {full['n']} đặc trưng", full),
                ("Lựa chọn hiện tại", res)]
        cmp_ = pd.DataFrame([{"Mô hình": name, "Số đặc trưng": r["n"], "MAE": r["MAE"], "Accuracy": r["accuracy"],
                              "Macro F1": r["macro_f1"], "Train (ms)": 1000 * r["time"],
                              "% cải thiện MAE": 100 * (base["MAE"] - r["MAE"]) / (base["MAE"] - full["MAE"])}
                             for name, r in rows])
        st.dataframe(cmp_.style.format({"MAE": "{:.4f}", "Accuracy": "{:.3f}", "Macro F1": "{:.3f}",
                                        "Train (ms)": "{:.0f}", "% cải thiện MAE": "{:.1f}"}),
                     hide_index=True, width="stretch")

    a, b = st.columns(2, gap="large")       # hàng riêng, rộng toàn trang
    with a:
        fig = px.imshow(res["cm"], x=CLASSES, y=CLASSES, text_auto=".2f", color_continuous_scale="Blues",
                        zmin=0, zmax=1, height=520, labels={"x": "dự đoán", "y": "nhãn thật", "color": "tỉ lệ"},
                        title="Ma trận nhầm lẫn (chuẩn hoá theo hàng = recall)")
        fig.update_coloraxes(showscale=False)
        fig.update_xaxes(tickangle=30)
        st.plotly_chart(fig, width="stretch")
    with b:
        coef = res["coef"]
        m = float(np.abs(coef.values).max()) or 1.0
        fig = px.imshow(coef, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-m, zmax=m, height=520,
                        labels={"x": "lớp", "y": "đặc trưng", "color": "w"},
                        title=f"Hệ số w của {model.split(' (')[0]} — mỗi cột là một lớp (đặc trưng đã chuẩn hoá)")
        fig.update_coloraxes(showscale=False)
        fig.update_xaxes(tickangle=30)
        st.plotly_chart(fig, width="stretch")
    st.caption("Ranh giới quyết định tuyến tính không đủ tách các lớp chồng lấn: `montrealCryptXXX` hay bị nhầm "
               "sang `princetonCerber` (cùng năm 2016); `otherRansom` gộp nhiều họ khác nhau nên gần như không "
               "được nhận ra.")


def LinearClassification() -> None:
    st.header("🧭 Chọn đặc trưng cho bộ phân lớp tuyến tính (Bài 2 — bản phân lớp)")
    st.caption("Bài toán **phân lớp 7 lớp** như Bài 1. **Linear Regression** được dùng như bộ phân lớp tuyến tính "
               "(nhãn one-hot → 7 điểm → `argmax`); chọn đặc trưng bằng **tương quan point-biserial** giữa đặc trưng "
               "và từng lớp; so sánh các tập đặc trưng bằng **MAE** (giữa điểm và nhãn one-hot) kèm accuracy, "
               "macro F1. Bản hồi quy `log_income` nằm ở trang *Chọn đặc trưng (Bài 2)*.")
    missing = [f for f in FILES if not (B2C / f).exists()]
    if missing:
        st.error(f"Thiếu dữ liệu trong `data/bai2_cls/`: {', '.join(missing)}. "
                 "Chạy notebook `../bai2_classification.ipynb` trước.")
        return

    cc, cm, tk = _read("corr_classes.csv", 0), _read("corr_matrix.csv", 0), _read("topk.csv")
    sb, cr = _read("subsets.csv"), _read("cross.csv")
    ranking, n = cc.index.tolist(), len(cc)
    by_k = tk.set_index("n_features")
    k_best = int(tk.loc[tk["improvement_pct"] >= GOAL, "n_features"].min())
    prop = sb[sb["subset"].str.startswith(f"Top-{k_best} +")].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Số đặc trưng", n, help="10 đặc trưng số đã log1p, income_tz, year, day, month, weekday")
    c2.metric(f"MAE — tất cả {n} đặc trưng", f"{by_k.loc[n, 'MAE']:.4f}",
              help=f"Baseline (điểm = tỉ lệ lớp): {by_k.loc[0, 'MAE']:.4f}")
    c3.metric(f"MAE — tập đề xuất ({prop['n_features']} đặc trưng)", f"{prop['MAE']:.4f}",
              f"{prop['MAE'] - by_k.loc[n, 'MAE']:+.4f} so với tất cả", delta_color="inverse",
              help=f"Top-{k_best} theo R (k nhỏ nhất đạt ≥ {GOAL}% mức cải thiện MAE) rồi bỏ đặc trưng dư thừa "
                   f"(|r| ≥ {RED_THR})")
    c4.metric("Macro F1 — tập đề xuất", f"{prop['macro_f1']:.3f}",
              f"{100 * prop['macro_f1'] / by_k.loc[n, 'macro_f1']:.0f}% so với tất cả", delta_color="off",
              delta_arrow="off")

    tab0, tab1, tab2, tab3 = st.tabs(["Mô hình & dữ liệu", "Tương quan", "Kết quả MAE", "Thử nghiệm trực tiếp"])
    with tab0:
        _model_tab(ranking, cm, k_best)
    with tab1:
        _correlation(cc, cm, by_k)
    with tab2:
        _results(tk, sb, cr, k_best, prop)
    with tab3:
        _live(cc, cm, k_best)

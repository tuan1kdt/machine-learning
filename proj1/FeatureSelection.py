"""Bài 2 — Chọn đặc trưng dựa trên tương quan (Correlation-based Feature Selection).

Hồi quy log_income = log1p(income) bằng Linear Regression, so sánh các tập đặc trưng bằng MAE.
Dữ liệu data/bai2/ do notebook ../bai2.ipynb tạo ra (ô "Xuất kết quả cho demo Streamlit").
"""
import time

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import DATA

B2 = DATA / "bai2"
FILES = ["corr_target.csv", "corr_matrix.csv", "topk.csv", "subsets.csv", "sample.csv.gz"]
TARGET = "log_income"
BLUE, RED, ORANGE, GRAY = "#0068c9", "#e45756", "#f58518", "#9aa0a6"
METHODS = ["Top-k theo |r|", "Ngưỡng |r|", "Tự chọn"]


@st.cache_data(show_spinner="Đang nạp dữ liệu Bài 2...")
def _read(name: str, index_col: int | None = None) -> pd.DataFrame:
    return pd.read_csv(B2 / name, index_col=index_col)


def _fit(df: pd.DataFrame, cols: list[str]) -> dict:
    """StandardScaler → LinearRegression: train trên split == "train", đo trên split == "test".
    Không có đặc trưng nào → baseline: luôn dự đoán trung bình log_income của tập train."""
    tr, te = df[df["split"] == "train"], df[df["split"] == "test"]
    if cols:
        model = Pipeline([("scaler", StandardScaler()), ("lr", LinearRegression())])
        t0 = time.perf_counter()
        model.fit(tr[cols], tr[TARGET])
        t_train = time.perf_counter() - t0
        pred, coef = model.predict(te[cols]), pd.Series(model.named_steps["lr"].coef_, index=cols)
    else:
        t_train, pred, coef = 0.0, np.full(len(te), tr[TARGET].mean()), pd.Series(dtype=float)
    return {"n": len(cols), "MAE": mean_absolute_error(te[TARGET], pred), "R2": r2_score(te[TARGET], pred),
            "time": t_train, "coef": coef}


@st.cache_data(show_spinner=False)
def _reference(features: tuple[str, ...]) -> tuple[dict, dict]:
    """Baseline và mô hình dùng tất cả đặc trưng — cùng mẫu với thử nghiệm trực tiếp, chỉ tính 1 lần."""
    df = _read("sample.csv.gz")
    return _fit(df, []), _fit(df, list(features))


def drop_redundant(feats: list[str], corr: pd.DataFrame, thr: float) -> tuple[list[str], list[tuple[str, str, float]]]:
    """Tham lam (như notebook): duyệt feats theo |r| với mục tiêu giảm dần, chỉ giữ f nếu |r(f, g)| < thr với mọi g
    đã giữ. Trả về (đặc trưng giữ lại, [(đặc trưng bị loại, đặc trưng đã giữ trùng thông tin, r)])."""
    kept, removed = [], []
    for f in feats:
        g = max(kept, key=lambda h: abs(corr.loc[f, h]), default=None)
        if g is not None and abs(corr.loc[f, g]) >= thr:
            removed.append((f, g, corr.loc[f, g]))
        else:
            kept.append(f)
    return kept, removed


def _pills(items: list[str]) -> None:
    st.markdown(" ".join(f'<span class="pill">{t}</span>' for t in items) or "—", unsafe_allow_html=True)


def _delta(d: float, fmt: str = "+.4f") -> str | None:
    """Delta cho st.metric; ẩn khi chênh lệch (sau làm tròn) bằng 0 — tránh mũi tên màu cho '+0.0000'."""
    s = f"{d:{fmt}}"
    return None if float(s) == 0 else s


# ---------------- Tab "Tương quan" (tính trên tập train của notebook)
def _correlation(ct: pd.DataFrame, cm: pd.DataFrame, by_k: pd.DataFrame) -> None:
    ranking = ct["feature"].tolist()
    a, b = st.columns([1.6, 1], gap="large")
    with b:
        thr = st.slider("Ngưỡng |r|", 0.0, 0.3, 0.03, 0.005, format="%.3f", key="fs_corr_thr",
                        help="Giữ các đặc trưng có |r| với log_income ≥ ngưỡng")
        sel = ct["abs_pearson"] >= thr
        k = int(sel.sum())
        st.markdown(f"**Được chọn: {k}/{len(ct)} đặc trưng**")
        _pills(ct.loc[sel, "feature"].tolist())
        if k < len(ct):
            st.caption("Bị loại: " + ", ".join(ct.loc[~sel, "feature"]))
        if k:   # chọn theo ngưỡng luôn là một tiền tố của thứ hạng |r| → trùng Top-k
            st.caption(f"Tập này chính là Top-{k} theo |r| → MAE (test) = {by_k.loc[k, 'MAE']:.4f}, đạt "
                       f"{by_k.loc[k, 'improvement_pct']:.1f}% mức cải thiện (xem tab *Kết quả MAE*).")
        else:
            st.caption(f"Không còn đặc trưng nào → chỉ còn baseline, MAE = {by_k.loc[0, 'MAE']:.4f}.")
        st.info("**Liên quan (relevance):** giữ các đặc trưng có |r| với mục tiêu **lớn** — chúng mang nhiều thông tin "
                "(tuyến tính) về `log_income`. Chọn theo **Top-k** (k đặc trưng có |r| lớn nhất) hoặc theo **ngưỡng** "
                "|r| ≥ t. `year` (r = −0,285) và `neighbors` (+0,198) liên quan nhất; `weekday`, `looped` gần như "
                "không liên quan. Mọi |r| đều < 0,3 → quan hệ tuyến tính khá yếu.")
    with a:
        d = ct.assign(**{"Trạng thái": np.where(sel, "Được chọn", "Bị loại")})
        m = 1.35 * ct["abs_pearson"].max()
        fig = px.bar(d, x="pearson", y="feature", orientation="h", color="Trạng thái", text_auto=".3f",
                     color_discrete_map={"Được chọn": BLUE, "Bị loại": GRAY}, range_x=[-m, m],
                     category_orders={"feature": ranking, "Trạng thái": ["Được chọn", "Bị loại"]},
                     hover_data={"rank": True, "spearman": ":.3f"},
                     labels={"pearson": "r (Pearson)", "feature": "đặc trưng", "rank": "hạng", "spearman": "Spearman"},
                     title="Tương quan Pearson giữa từng đặc trưng và log_income (tập train)", height=440)
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_yaxes(title=None)
        for x in (-thr, thr):
            fig.add_vline(x=x, line_dash="dash", line_color=GRAY)
        st.plotly_chart(fig, width="stretch")

    st.markdown("##### Tương quan giữa các đặc trưng — phát hiện đặc trưng dư thừa")
    a, b = st.columns([1.6, 1], gap="large")
    with a:
        fig = px.imshow(cm, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto",
                        title="Ma trận tương quan Pearson (tập train) — hàng/cột cuối: log_income", height=620)
        st.plotly_chart(fig, width="stretch")
    with b:
        feats = [f for f in cm.columns if f != TARGET]
        pairs = pd.DataFrame([(f, g, cm.loc[f, g]) for i, f in enumerate(feats) for g in feats[i + 1:]],
                             columns=["Đặc trưng 1", "Đặc trưng 2", "r"])
        st.markdown("**8 cặp đặc trưng tương quan mạnh nhất**")
        st.dataframe(pairs.sort_values("r", key=abs, ascending=False).head(8).style.format({"r": "{:+.3f}"}),
                     hide_index=True, width="stretch")
        st.info("**Không dư thừa (redundancy):** hai đặc trưng tương quan rất mạnh *với nhau* mang thông tin trùng lặp "
                "→ chỉ giữ một (đặc trưng có |r| với mục tiêu cao hơn). Ví dụ `day` – `month` (r = 0,996): bỏ `day` "
                "thì MAE không đổi. `looped` – `looped_ratio` (0,87), `count` – `length` (0,76) tương quan mạnh nhưng "
                "vẫn mang thông tin riêng.")

    st.markdown("##### Pearson (quan hệ tuyến tính) và Spearman (quan hệ đơn điệu)")
    long = ct.melt(id_vars="feature", value_vars=["pearson", "spearman"], var_name="method", value_name="r")
    long["|r|"] = long["r"].abs()
    long["method"] = long["method"].map({"pearson": "|Pearson|", "spearman": "|Spearman|"})
    fig = px.bar(long, x="feature", y="|r|", color="method", barmode="group", category_orders={"feature": ranking},
                 color_discrete_map={"|Pearson|": BLUE, "|Spearman|": ORANGE}, hover_data={"r": ":.4f"},
                 labels={"feature": "đặc trưng", "method": "hệ số", "|r|": "độ lớn |r|"},
                 title="Độ lớn tương quan với log_income",
                 height=400)
    fig.update_xaxes(title=None)
    st.plotly_chart(fig, width="stretch")
    st.info("Spearman = Pearson tính trên **thứ hạng** → đo quan hệ **đơn điệu** (không cần tuyến tính), ít nhạy với "
            "ngoại lai. Hai hệ số thống nhất ở `year`, `neighbors`; Spearman xếp `weight` thấp hơn nhiều "
            "(0,089 → 0,013) → quan hệ không tuyến tính. Mô hình là Linear Regression nên dùng **Pearson** làm tiêu "
            "chí chính (Top-6 theo Spearman cho MAE kém hơn: 1,3476 so với 1,3325).")
    with st.expander("Bảng hệ số tương quan với log_income"):
        st.dataframe(ct.style.format({"pearson": "{:+.4f}", "spearman": "{:+.4f}", "abs_pearson": "{:.4f}"}),
                     hide_index=True, width="stretch")


# ---------------- Tab "Kết quả MAE" (notebook, toàn bộ dữ liệu)
def _results(tk: pd.DataFrame, sb: pd.DataFrame, k_best: int) -> None:
    st.caption("Kết quả của notebook Bài 2 trên toàn bộ dữ liệu: train 2.333.357 dòng (80%), "
               "đo MAE trên 583.340 dòng test (20%).")
    n = int(tk["n_features"].max())
    base, full = tk["MAE"].iloc[0], tk["MAE"].iloc[-1]
    best = tk.set_index("n_features").loc[k_best]
    fig = px.line(tk, x="n_features", y="MAE", markers=True, height=460,
                  hover_data={"added": True, "improvement_pct": ":.1f", "R2": ":.4f"},
                  labels={"n_features": "k — số đặc trưng (thêm dần theo |r| giảm dần)", "added": "đặc trưng thêm vào",
                          "improvement_pct": "% cải thiện", "R2": "R²"},
                  title="MAE trên tập test theo số đặc trưng (Top-k theo |r|)")
    fig.add_hline(y=base, line_dash="dash", line_color=GRAY, annotation_position="bottom right",
                  annotation_text=f"Baseline — luôn dự đoán trung bình: {base:.4f}")
    fig.add_hline(y=full, line_dash="dot", line_color=GRAY, annotation_position="top left",
                  annotation_text=f"Tất cả {n} đặc trưng: {full:.4f}")
    fig.add_scatter(x=[k_best], y=[best["MAE"]], mode="markers", hoverinfo="skip", showlegend=False,
                    marker={"size": 20, "symbol": "circle-open", "color": ORANGE, "line": {"width": 3}})
    fig.add_annotation(x=k_best, y=best["MAE"], ax=50, ay=-70, arrowcolor=ORANGE, align="left",
                       text=f"<b>Top-{k_best}</b>: MAE {best['MAE']:.4f}<br>{best['improvement_pct']:.0f}% mức cải "
                            f"thiện với {100 * k_best / n:.0f}% số đặc trưng")
    fig.update_xaxes(dtick=1)
    st.plotly_chart(fig, width="stretch")

    st.markdown("##### Tổng hợp các tập đặc trưng")
    show = (sb[["subset", "n_features", "MAE", "mae_vs_all", "R2", "improvement_pct", "train_time_s", "features"]]
            .fillna({"features": "—"})
            .rename(columns={"subset": "Tập đặc trưng", "n_features": "Số đặc trưng",
                             "mae_vs_all": "MAE chênh so với tất cả", "R2": "R²", "improvement_pct": "% cải thiện",
                             "train_time_s": "Train (s)", "features": "Đặc trưng"}))
    st.dataframe(show.style.format({"MAE": "{:.4f}", "MAE chênh so với tất cả": "{:+.4f}", "R²": "{:.4f}",
                                    "% cải thiện": "{:.1f}", "Train (s)": "{:.3f}"})
                     .background_gradient(cmap="Greens_r", subset=["MAE"]),
                 hide_index=True, width="stretch")

    a, b = st.columns([1.4, 1], gap="large")
    with a:
        d = sb.assign(group=np.where(sb["subset"].str.startswith("Top-") & (sb["n_features"] == k_best),
                                     "Đề xuất", "Khác"))
        lo, hi = d["MAE"].min(), d["MAE"].max()
        fig = px.bar(d, x="MAE", y="subset", orientation="h", color="group", text_auto=".4f",
                     color_discrete_map={"Đề xuất": BLUE, "Khác": GRAY}, range_x=[lo - 0.03, hi + 0.03],
                     category_orders={"subset": d["subset"].tolist(), "group": ["Đề xuất", "Khác"]},
                     hover_data={"n_features": True, "improvement_pct": ":.1f"},
                     labels={"subset": "tập đặc trưng", "group": "nhóm", "n_features": "số đặc trưng",
                             "improvement_pct": "% cải thiện", "MAE": "MAE (trục không bắt đầu từ 0)"},
                     title="MAE của Linear Regression theo tập đặc trưng (càng nhỏ càng tốt)", height=440)
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_yaxes(title=None)
        st.plotly_chart(fig, width="stretch")
    with b:
        st.markdown("""
**Nhận xét**
1. MAE giảm nhanh ở vài đặc trưng đầu rồi **bão hoà**: chỉ `year` đã đạt ~55% mức cải thiện, `year + neighbors`
   ~85%; **Top-6 đạt ~98%** (MAE 1,3325 so với 1,3300) với **50% số đặc trưng** và train nhanh hơn ~2 lần.
2. Top-6 theo |r| trùng với `SelectKBest(f_regression, k=6)` của scikit-learn.
3. Bỏ `day` (dư thừa với `month`, r = 0,996) → MAE **không đổi**.
4. **Đối chứng:** 6 đặc trưng có |r| thấp nhất cho MAE 1,4607 ≈ baseline 1,4641 → |r| xác định đúng các đặc trưng
   quan trọng.
5. **Hạn chế:** R² chỉ ≈ 0,15 — quan hệ tuyến tính với income yếu; tương quan chỉ xét từng cặp và chỉ đo quan hệ
   tuyến tính.
""")


# ---------------- Tab "Thử nghiệm trực tiếp" (train lại trên mẫu)
def _live(ct: pd.DataFrame, cm: pd.DataFrame, k_best: int) -> None:
    ranking = ct["feature"].tolist()
    abs_r = ct.set_index("feature")["abs_pearson"]
    st.caption("Train lại `StandardScaler → LinearRegression` ngay trong demo trên mẫu 100.000 dòng train / "
               "25.000 dòng test (lấy ngẫu nhiên từ phép chia train/test của notebook) → MAE hơi khác tab "
               "*Kết quả MAE*.")
    left, right = st.columns([1, 2], gap="large")
    with left:
        method = st.radio("Cách chọn đặc trưng", METHODS, key="fs_method",
                          captions=["k đặc trưng có |r| lớn nhất", "giữ các đặc trưng có |r| ≥ ngưỡng", "chọn tuỳ ý"])
        if method == METHODS[0]:
            chosen = ranking[:st.slider("k — số đặc trưng", 1, len(ranking), k_best, key="fs_k")]
        elif method == METHODS[1]:
            t = st.slider("Ngưỡng |r|", 0.0, 0.3, 0.03, 0.005, format="%.3f", key="fs_live_thr")
            chosen = [f for f in ranking if abs_r[f] >= t]
        else:
            picked = st.multiselect("Đặc trưng", ranking, default=ranking[:k_best], key="fs_custom")
            chosen = [f for f in ranking if f in picked]                      # sắp theo |r| giảm dần

        dedup = st.checkbox("Loại đặc trưng dư thừa", key="fs_dedup",
                            help="Duyệt theo |r| với mục tiêu giảm dần; bỏ một đặc trưng nếu |r| của nó với một đặc "
                                 "trưng đã giữ ≥ ngưỡng (ma trận tương quan của tập train)")
        thr = st.slider("Ngưỡng |r| giữa hai đặc trưng", 0.5, 0.99, 0.9, 0.01, key="fs_red_thr", disabled=not dedup)
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
            st.warning("Chưa có đặc trưng nào được chọn — hãy giảm ngưỡng |r| hoặc chọn thêm đặc trưng.")
            return
        base, full = _reference(tuple(ranking))
        res = _fit(_read("sample.csv.gz"), chosen)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Số đặc trưng", f"{res['n']}/{full['n']}")
        c2.metric("MAE (test)", f"{res['MAE']:.4f}", _delta(res["MAE"] - full["MAE"]), delta_color="inverse",
                  help=f"Mức chênh so với tất cả {full['n']} đặc trưng (MAE {full['MAE']:.4f}); "
                       f"baseline: {base['MAE']:.4f}")
        c3.metric("R²", f"{res['R2']:.4f}", _delta(res["R2"] - full["R2"]),
                  help=f"Mức chênh so với tất cả {full['n']} đặc trưng (R² {full['R2']:.4f})")
        c4.metric("Thời gian train", f"{1000 * res['time']:.0f} ms",
                  help=f"Tất cả {full['n']} đặc trưng: {1000 * full['time']:.0f} ms")

        rows = [("Baseline (dự đoán trung bình)", base), (f"Tất cả {full['n']} đặc trưng", full),
                ("Lựa chọn hiện tại", res)]
        cmp_ = pd.DataFrame([{"Mô hình": name, "Số đặc trưng": r["n"], "MAE": r["MAE"], "R²": r["R2"],
                              "Train (ms)": 1000 * r["time"],
                              "% cải thiện": 100 * (base["MAE"] - r["MAE"]) / (base["MAE"] - full["MAE"])}
                             for name, r in rows])
        st.dataframe(cmp_.style.format({"MAE": "{:.4f}", "R²": "{:.4f}", "Train (ms)": "{:.0f}",
                                        "% cải thiện": "{:.1f}"}),
                     hide_index=True, width="stretch")

        coef = res["coef"].sort_values(ascending=False)       # px vẽ phần tử đầu ở trên cùng
        d = pd.DataFrame({"feature": coef.index, "w": coef.values,
                          "dấu": np.where(coef.values >= 0, "dương (+)", "âm (−)")})
        m = 1.35 * coef.abs().max() or 1.0
        fig = px.bar(d, x="w", y="feature", orientation="h", color="dấu", text_auto=".3f", range_x=[-m, m],
                     color_discrete_map={"dương (+)": BLUE, "âm (−)": RED},
                     category_orders={"feature": d["feature"].tolist(), "dấu": ["dương (+)", "âm (−)"]},
                     labels={"w": "hệ số w", "feature": "đặc trưng"}, height=140 + 32 * len(d),
                     title="Hệ số Linear Regression (đặc trưng đã chuẩn hoá)")
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_yaxes(title=None)
        st.plotly_chart(fig, width="stretch")
        st.caption("Đặc trưng đã chuẩn hoá → |w| so sánh được: w là mức thay đổi dự đoán của log_income khi đặc trưng "
                   "tăng 1 độ lệch chuẩn (các đặc trưng khác giữ nguyên). Các đặc trưng tương quan với nhau "
                   "(vd `count` – `length`) có thể làm hệ số khó diễn giải.")


def FeatureSelection() -> None:
    st.header("🧪 Chọn đặc trưng dựa trên tương quan (Bài 2)")
    st.caption("Bài toán **hồi quy**: dự đoán `log_income = log1p(income)` — lượng satoshi một địa chỉ Bitcoin nhận "
               "trong một ngày — từ 12 đặc trưng bằng **Linear Regression**; các tập đặc trưng được so sánh bằng "
               "**MAE** trên tập test (càng nhỏ càng tốt). Không dùng nhãn 7 lớp của Bài 1 vì Linear Regression và MAE "
               "cần biến mục tiêu **liên tục** — đánh số các lớp danh nghĩa (không có thứ tự) rồi hồi quy là vô nghĩa.")
    missing = [f for f in FILES if not (B2 / f).exists()]
    if missing:
        st.error(f"Thiếu dữ liệu Bài 2 trong `data/bai2/`: {', '.join(missing)}. Chạy notebook `../bai2.ipynb` trước.")
        return

    ct, cm, tk = _read("corr_target.csv"), _read("corr_matrix.csv", 0), _read("topk.csv")
    n = len(ct)
    by_k = tk.set_index("n_features")
    k_best = int(tk.loc[tk["improvement_pct"] >= 95, "n_features"].min())   # k nhỏ nhất đạt ≥ 95% (như notebook)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Số đặc trưng", n, help="7 đặc trưng số đã log1p (length, weight, count, looped, neighbors, "
                                      "weight_per_count, looped_ratio), year, day, month, weekday và is_ransom")
    c2.metric(f"MAE — tất cả {n} đặc trưng", f"{by_k.loc[n, 'MAE']:.4f}",
              help=f"Baseline (luôn dự đoán trung bình): {by_k.loc[0, 'MAE']:.4f}")
    c3.metric(f"MAE — Top-{k_best} theo |r|", f"{by_k.loc[k_best, 'MAE']:.4f}",
              f"{by_k.loc[k_best, 'MAE'] - by_k.loc[n, 'MAE']:+.4f} so với tất cả", delta_color="inverse")
    c4.metric(f"% cải thiện đạt được (Top-{k_best})", f"{by_k.loc[k_best, 'improvement_pct']:.1f}%",
              f"chỉ dùng {k_best}/{n} đặc trưng", delta_color="off", delta_arrow="off",
              help="(MAE baseline − MAE Top-k) / (MAE baseline − MAE tất cả) × 100")

    tab1, tab2, tab3 = st.tabs(["Tương quan", "Kết quả MAE", "Thử nghiệm trực tiếp"])
    with tab1:
        _correlation(ct, cm, by_k)
    with tab2:
        _results(tk, _read("subsets.csv"), k_best)
    with tab3:
        _live(ct, cm, k_best)

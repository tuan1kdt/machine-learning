import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from common import CLASS_COLORS, CLASSES, label_counts, test_sample, year_target_counts
from features import NUM, add_features


def Statistics() -> None:
    st.header("📊 Khám phá dữ liệu")

    lc = label_counts()
    total = int(lc["count"].sum())
    n_white = int(lc.loc[lc["label"] == "white", "count"].sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Số dòng", f"{total:,}")
    c2.metric("Số cột", "10", help="address, year, day, 6 đặc trưng số, label")
    c3.metric("Nhãn gốc", f"{len(lc)}")
    c4.metric("Tỉ lệ ransomware", f"{100 * (total - n_white) / total:.2f}%")

    tab1, tab2, tab3 = st.tabs(["Nhãn", "Đặc trưng", "Tương quan"])

    # ---------------- Nhãn (toàn bộ 2,9 triệu dòng)
    with tab1:
        a, b = st.columns([1.3, 1], gap="large")
        with a:
            fig = px.bar(lc.sort_values("count"), x="count", y="label", orientation="h", log_x=True,
                         title=f"Phân phối {len(lc)} nhãn gốc (thang log)", height=720)
            st.plotly_chart(fig, width="stretch")
        with b:
            yt = year_target_counts()
            tgt = yt.groupby("target")["count"].sum().reset_index()
            fig = px.pie(tgt[tgt["target"] != "white"], names="target", values="count", color="target",
                         color_discrete_map=CLASS_COLORS, hole=0.45,
                         title="Cơ cấu 6 lớp ransomware (white = 98,6% không vẽ)")
            st.plotly_chart(fig, width="stretch")
            st.dataframe(tgt.assign(pct=lambda d: (100 * d["count"] / total).round(3))
                         .sort_values("count", ascending=False), hide_index=True, width="stretch")

        yt = year_target_counts()
        ransom = yt[yt["target"] != "white"]
        a, b = st.columns(2, gap="large")
        with a:
            fig = px.bar(ransom, x="year", y="count", color="target", color_discrete_map=CLASS_COLORS,
                         title="Số dòng ransomware theo năm")
            st.plotly_chart(fig, width="stretch")
        with b:
            pct = yt.assign(r=yt["target"] != "white").groupby(["year", "r"])["count"].sum().unstack()
            pct = (100 * pct[True] / pct.sum(axis=1)).rename("ransom_pct").reset_index()
            fig = px.line(pct, x="year", y="ransom_pct", markers=True, title="Tỉ lệ ransomware theo năm (%)")
            st.plotly_chart(fig, width="stretch")
        st.info("`white` được lấy mẫu cố định 1.000 địa chỉ/ngày; tỉ lệ ransomware thay đổi rất mạnh theo năm "
                "(cao nhất 2016) → `year` là đặc trưng quan trọng.")

    # ---------------- Đặc trưng (mẫu từ tập test)
    df = add_features(test_sample())
    with tab2:
        st.caption(f"Mẫu từ tập test: {len(df):,} dòng — toàn bộ {int((df['target'] != 'white').sum()):,} "
                   "dòng ransomware + 20.000 dòng white.")
        a, b, c = st.columns([1, 1, 1])
        col = a.selectbox("Đặc trưng", NUM, index=NUM.index("income"))
        use_log = b.toggle("Áp dụng log1p", value=True, help="Các đặc trưng lệch rất mạnh → log1p trước khi chuẩn hoá")
        classes = c.multiselect("Lớp", CLASSES, default=["white", "princetonLocky", "montrealCryptXXX"])
        d = df[df["target"].isin(classes)].copy()
        x = f"log1p({col})" if use_log else col
        d[x] = np.log1p(d[col]) if use_log else d[col]

        a, b = st.columns(2, gap="large")
        with a:
            fig = px.histogram(d, x=x, color="target", color_discrete_map=CLASS_COLORS, nbins=60,
                               histnorm="probability density", barmode="overlay", opacity=0.6,
                               title=f"Phân phối {x} theo lớp")
            st.plotly_chart(fig, width="stretch")
        with b:
            fig = px.box(d, x="target", y=x, color="target", color_discrete_map=CLASS_COLORS,
                         title=f"Box plot {x} theo lớp")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, width="stretch")

        st.markdown("##### Độ “tròn” của income (số chữ số 0 ở cuối, đơn vị satoshi)")
        tz = (df.assign(group=np.where(df["target"] == "white", "white", "ransomware"),
                        income_tz=df["income_tz"].clip(upper=10))
                .groupby("group")["income_tz"].value_counts(normalize=True).rename("ratio").reset_index())
        fig = px.bar(tz, x="income_tz", y="ratio", color="group", barmode="group",
                     color_discrete_map={"white": CLASS_COLORS["white"], "ransomware": "#e45756"})
        fig.update_layout(xaxis_title="income_tz (≥10 gộp chung)", yaxis_title="Tỉ lệ trong nhóm")
        st.plotly_chart(fig, width="stretch")
        st.caption("Tiền chuộc thường là số tròn (vd 1e8 satoshi = 1 BTC) → đặc trưng mới `income_tz`.")

        with st.expander("Xem dữ liệu mẫu"):
            st.dataframe(df.head(500), width="stretch")

    # ---------------- Tương quan
    with tab3:
        a, b = st.columns([1, 2])
        method = a.radio("Phương pháp", ["pearson", "spearman"], horizontal=True,
                         help="Pearson tính trên log1p(đặc trưng); Spearman dựa trên thứ hạng")
        feats = b.multiselect("Đặc trưng", NUM + ["year", "month", "weekday"], default=NUM)
        if len(feats) >= 2:
            X = df[feats].astype(float)
            if method == "pearson":
                X = np.log1p(X.clip(lower=0))
            corr = X.corr(method=method)
            fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                            aspect="auto", title=f"Ma trận tương quan ({method})", height=650)
            st.plotly_chart(fig, width="stretch")

            pairs = (corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
                     .rename("corr").reset_index().rename(columns={"level_0": "feature_1", "level_1": "feature_2"}))
            pairs = pairs.reindex(pairs["corr"].abs().sort_values(ascending=False).index).head(10)
            st.markdown("##### 10 cặp đặc trưng tương quan mạnh nhất")
            st.dataframe(pairs, hide_index=True, width="stretch")
        else:
            st.info("Chọn ít nhất 2 đặc trưng.")

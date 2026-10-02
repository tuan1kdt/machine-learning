# app.py
# Single-page Streamlit dashboard: Data Statistics
# Loads CSV from: data/insurance.csv

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

# Optional shadcn-ui
try:
    import streamlit_shadcn_ui as ui
except Exception:
    ui = None


def Statistics() -> None:
    st.header("📊 DATA STATISTICS")

    DATA_PATH = "data/insurance.csv"

    # ----------------------------
    # Load data
    # ----------------------------
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        st.error(f"File not found: `{DATA_PATH}`")
        st.stop()
    except Exception as e:
        st.error(f"Failed to load `{DATA_PATH}`: {e}")
        st.stop()

    df.columns = [c.strip() for c in df.columns]
    if df.empty:
        st.error("Dataset is empty.")
        st.stop()

    # ----------------------------
    # Key factory (for component instances)
    # ----------------------------
    page = "data_stats"
    _k = {"i": 0}

    def k(name: str) -> str:
        _k["i"] += 1
        return f"{page}_{name}_{_k['i']}"

    # ----------------------------
    # Helpers
    # ----------------------------
    def numeric_cols(d: pd.DataFrame):
        return [c for c in d.columns if pd.api.types.is_numeric_dtype(d[c])]

    def categorical_cols(d: pd.DataFrame):
        return [c for c in d.columns if not pd.api.types.is_numeric_dtype(d[c])]

    def show_table(d: pd.DataFrame, name: str):
        if ui is not None:
            try:
                ui.table(d, key=k(name))
                return
            except Exception:
                pass
        st.dataframe(d, use_container_width=True)

    num_cols = numeric_cols(df)
    cat_cols = categorical_cols(df)

    # ----------------------------
    # KPIs
    # ----------------------------
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Rows", f"{len(df):,}", help="Total records")
    with c2:
        st.metric("Columns", f"{df.shape[1]}", help="Total columns")
    with c3:
        st.metric("Numeric", f"{len(num_cols)}", help="Numeric columns")
    with c4:
        st.metric("Missing", f"{int(df.isna().sum().sum()):,}", help="Total missing values")

    st.divider()

    # ----------------------------
    # Schema
    # ----------------------------
    st.subheader("🧱 Dataset Schema")

    schema = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(df[c].dtype) for c in df.columns],
        "non_null": df.notna().sum().values,
        "unique": df.nunique(dropna=True).values,
    })
    show_table(schema, "schema")

    st.divider()

    # ----------------------------
    # Missing values
    # ----------------------------
    st.subheader("🕳 Missing Values")

    miss = pd.DataFrame({
        "column": df.columns,
        "missing_count": df.isna().sum().values,
        "missing_%": (df.isna().mean() * 100).round(2).values,
    }).sort_values("missing_count", ascending=False)

    m1, m2 = st.columns([1.2, 1.0], vertical_alignment="top")
    with m1:
        show_table(miss, "missing_table")

    with m2:
        miss_plot = miss[miss["missing_count"] > 0].head(30)
        if miss_plot.empty:
            st.success("No missing values 🎉")
        else:
            fig = px.bar(
                miss_plot,
                x="missing_%",
                y="column",
                orientation="h",
                title="Missing % by column (top 30)",
            )
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ----------------------------
    # Numeric statistics (WITH COLUMN NAMES)
    # ----------------------------
    st.subheader("📈 Numeric Statistics")

    if not num_cols:
        st.info("No numeric columns detected.")
    else:
        n1, n2 = st.columns([1.15, 1.0], vertical_alignment="top")

        with n1:
            desc = (
                df[num_cols]
                .describe()
                .T
                .reset_index()
                .rename(columns={"index": "column"})
            )
            show_table(desc, "numeric_describe")

        with n2:
            if len(num_cols) >= 2:
                corr = df[num_cols].corr(numeric_only=True)
                fig = px.imshow(
                    corr,
                    text_auto=True,
                    title="Correlation matrix",
                    aspect="auto",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Need at least 2 numeric columns for correlation.")

        st.markdown("#### 🔎 Numeric Explorer")

        col = st.selectbox(
            "Pick a numeric column",
            num_cols,
            index=0,
            key=f"{page}_num_pick",
        )

        e1, e2 = st.columns(2, vertical_alignment="top")
        with e1:
            fig = px.histogram(df, x=col, nbins=40, title=f"Distribution of {col}")
            st.plotly_chart(fig, use_container_width=True)
        with e2:
            fig = px.box(df, y=col, points="outliers", title=f"Box plot of {col}")
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ----------------------------
    # Categorical statistics
    # ----------------------------
    if cat_cols:
        st.subheader("🏷 Categorical Statistics")

        cat = st.selectbox(
            "Explore categorical column",
            cat_cols,
            index=0,
            key=f"{page}_cat_pick",
        )

        vc = df[cat].fillna("(missing)").value_counts(dropna=False).reset_index()
        vc.columns = [cat, "count"]
        vc["%"] = (vc["count"] / len(df) * 100).round(2)

        cA, cB = st.columns([1.2, 1.0], vertical_alignment="top")
        with cA:
            show_table(vc, "cat_value_counts")
        with cB:
            fig = px.bar(vc, x=cat, y="count", title=f"Value counts: {cat}")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No categorical columns detected.")


# ----------------------------
# App entry
# ----------------------------
st.set_page_config(page_title="Data Statistics", layout="wide")

import streamlit as st
import pandas as pd
import joblib
import streamlit_shadcn_ui as ui

# -----------------------------
# Model loading (cached)
# -----------------------------
@st.cache_resource
def load_model():
    # model.ckpt is expected to be a sklearn Pipeline:
    #   preprocess (ColumnTransformer with StandardScaler + OneHotEncoder)
    #   regressor  (LinearRegression)
    return joblib.load("model.ckpt")


# -----------------------------
# Helpers
# -----------------------------
def _init_defaults():
    defaults = {
        "age": "30",
        "bmi": "25.0",
        "children": "0",
        "sex": "female",
        "smoker": "no",
        "region": "southeast",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _to_int(x, default=0):
    try:
        return int(float(x))
    except Exception:
        return default


def _to_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


# -----------------------------
# Page
# -----------------------------
def Home():
    st.set_page_config(page_title="Insurance Charges Predictor", layout="wide")
    _init_defaults()

    # Header
    col1, col2 = st.columns([4, 2], vertical_alignment="center")
    with col1:
        st.markdown("## 💳 Insurance Charges Predictor")
        st.caption("Fill in the fields and click Predict to estimate charges.")
    with col2:
        # badges expects list of tuples: (text, variant)
        ui.badges([("LinearRegression", "secondary"), ("StandardScaler", "secondary")], key="badge_model")

    st.divider()

    # "Card" shell (HTML for consistent styling across versions)
    st.markdown(
        """
        <div style="padding: 18px; border: 1px solid rgba(255,255,255,0.12);
                    border-radius: 14px; background: rgba(255,255,255,0.03); margin-bottom: 10px;">
        <h3 style="margin: 0 0 10px 0;">Input Information</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.caption("Age")
        ui.input(key="age", type="number", placeholder="e.g., 30")

        st.caption("Sex")
        ui.select(key="sex", options=["female", "male"])

    with c2:
        st.caption("BMI")
        ui.input(key="bmi", type="number", placeholder="e.g., 24.7")

        st.caption("Children")
        ui.input(key="children", type="number", placeholder="e.g., 2")

    with c3:
        st.caption("Smoker")
        ui.select(key="smoker", options=["no", "yes"])

        st.caption("Region")
        ui.select(
            key="region",
            options=["northeast", "northwest", "southeast", "southwest"],
        )

    st.divider()

    predict_clicked = ui.button("Predict", key="btn_predict")

    st.divider()

    st.markdown(
        """
        <div style="padding: 18px; border: 1px solid rgba(255,255,255,0.12);
                    border-radius: 14px; background: rgba(255,255,255,0.03); margin-bottom: 10px;">
        <h3 style="margin: 0 0 10px 0;">Prediction Result</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if predict_clicked:
        # Build input row with correct types and correct column names
        X_input = pd.DataFrame([{
            "age": _to_int(st.session_state["age"]),
            "sex": str(st.session_state["sex"]),
            "bmi": _to_float(st.session_state["bmi"]),
            "children": _to_int(st.session_state["children"]),
            "smoker": str(st.session_state["smoker"]),
            "region": str(st.session_state["region"]),
        }])

        try:
            model = load_model()

            pred = float(model.predict(X_input)[0])

            ui.metric_card(
                title="Predicted Charges",
                content=f"{pred:,.2f}",
                description="Estimated insurance cost",
                key="metric_pred",
            )

        except FileNotFoundError:
            st.error('Could not find "model.ckpt". Put it in the folder you run Streamlit from.')
        except Exception as e:
            st.error(f"Prediction failed: {e}")
    else:
        st.caption("Click **Predict** to calculate charges.")



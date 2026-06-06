import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# =========================
# LOAD MODEL
# =========================
model = joblib.load("bank_model.pkl")

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Bank Marketing Dashboard",
    page_icon="🏦",
    layout="wide"
)

# =========================
# HEADER
# =========================
st.title("🏦 AI Bank Marketing Intelligence Dashboard")
st.markdown("Predict customer subscription using Machine Learning")

st.divider()

# =========================
# SIDEBAR INPUTS (PRO UX)
# =========================
st.sidebar.header("📥 Customer Features")

age = st.sidebar.slider("Age", 18, 100, 30)

job = st.sidebar.selectbox(
    "Job",
    ["admin", "blue-collar", "entrepreneur", "housemaid", "management",
     "retired", "self-employed", "services", "student", "technician", "unemployed"]
)

marital = st.sidebar.selectbox(
    "Marital Status",
    ["single", "married", "divorced"]
)

education = st.sidebar.selectbox(
    "Education",
    ["primary", "secondary", "tertiary", "unknown"]
)

default = st.sidebar.selectbox("Has Credit Default?", ["no", "yes"])

balance = st.sidebar.number_input("Account Balance", value=1000)

housing = st.sidebar.selectbox("Housing Loan?", ["no", "yes"])

# =========================
# CREATE INPUT DATAFRAME
# =========================
input_df = pd.DataFrame([{
    "age": age,
    "job": job,
    "marital": marital,
    "education": education,
    "default": default,
    "balance": balance,
    "housing": housing
}])

# =========================
# MAIN DASHBOARD
# =========================
st.subheader("📊 Prediction Dashboard")

if st.button("🚀 Run Prediction"):

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    # =========================
    # KPI METRICS
    # =========================
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Prediction", "YES" if prediction == 1 else "NO")

    with col2:
        st.metric("Probability", f"{probability:.2f}")

    with col3:
        st.metric("Confidence", f"{probability*100:.1f}%")

    st.divider()

    # =========================
    # VISUAL INSIGHTS
    # =========================
    st.subheader("📈 Prediction Insights")

    st.progress(int(probability * 100))

    if probability > 0.7:
        st.success("🔥 High-value customer → prioritize marketing call")
    elif probability > 0.4:
        st.warning("⚠️ Medium potential → consider email campaign")
    else:
        st.error("❌ Low priority customer")

    st.divider()

    # =========================
    # INPUT DATA DISPLAY
    # =========================
    st.subheader("📋 Input Summary")
    st.dataframe(input_df)

    # =========================
    # FEATURE IMPORTANCE (IF SUPPORTED)
    # =========================
    if hasattr(model, "feature_importances_"):
        st.subheader("📊 Feature Importance")

        importances = model.feature_importances_
        features = model.feature_names_in_

        fig, ax = plt.subplots()
        ax.barh(features, importances)
        ax.set_title("Model Feature Importance")

        st.pyplot(fig)

    # =========================
    # DOWNLOAD RESULT
    # =========================
    result_df = input_df.copy()
    result_df["prediction"] = prediction
    result_df["probability"] = probability

    csv = result_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇ Download Prediction Report",
        data=csv,
        file_name="prediction_report.csv",
        mime="text/csv"
    )

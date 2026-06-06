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
# SESSION STATE (POWER BI STYLE TRACKING)
# =========================
if "history" not in st.session_state:
    st.session_state.history = []

# =========================
# HEADER
# =========================
st.title("🏦 AI Bank Marketing Intelligence Dashboard")
st.markdown("End-to-end ML system for customer subscription prediction")

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
# INPUT DATAFRAME
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

    # store history for analytics
    st.session_state.history.append(probability)

    # =========================
    # KPI CARDS (POWER BI STYLE)
    # =========================
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Prediction", "YES" if prediction == 1 else "NO")

    with col2:
        st.metric("Probability", f"{probability:.2f}")

    with col3:
        avg_prob = sum(st.session_state.history) / len(st.session_state.history)
        st.metric("Avg Probability", f"{avg_prob:.2f}")

    st.divider()

    # =========================
    # BUSINESS INSIGHT LAYER
    # =========================
    st.subheader("🧠 Business Insight")

    if probability > 0.7:
        st.success("🔥 High-value customer → prioritize sales call")
    elif probability > 0.4:
        st.warning("⚠️ Medium potential → retarget marketing campaign")
    else:
        st.error("❌ Low priority customer")

    st.divider()

    # =========================
    # PROGRESS VISUAL
    # =========================
    st.subheader("📈 Prediction Confidence")

    st.progress(int(probability * 100))
    st.write(f"Confidence Score: {probability*100:.1f}%")

    st.divider()

    # =========================
    # INPUT DISPLAY
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
        ax.set_title("Feature Importance")

        st.pyplot(fig)

    st.divider()

    # =========================
    # POWER BI STYLE ANALYTICS
    # =========================
    st.subheader("📊 Analytics Dashboard")

    if len(st.session_state.history) > 0:

        fig, ax = plt.subplots()
        ax.hist(st.session_state.history, bins=10)
        ax.set_title("Prediction Probability Distribution")
        ax.set_xlabel("Probability")
        ax.set_ylabel("Frequency")

        st.pyplot(fig)

    st.divider()

    # =========================
    # CUSTOMER SEGMENTATION
    # =========================
    st.subheader("🧠 Customer Segmentation")

    if len(st.session_state.history) > 0:

        high = len([x for x in st.session_state.history if x > 0.7])
        medium = len([x for x in st.session_state.history if 0.4 <= x <= 0.7])
        low = len([x for x in st.session_state.history if x < 0.4])

        st.write(f"🔥 High Value Customers: {high}")
        st.write(f"⚠️ Medium Value Customers: {medium}")
        st.write(f"❌ Low Value Customers: {low}")

    st.divider()

    # =========================
    # DOWNLOAD REPORT
    # =========================
    result_df = input_df.copy()
    result_df["prediction"] = prediction
    result_df["probability"] = probability

    csv = result_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇ Download Prediction Report",
        data=csv,
        file_name="ml_prediction_report.csv",
        mime="text/csv"
    )

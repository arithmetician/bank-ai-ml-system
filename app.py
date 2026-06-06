import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("bank_model.pkl")

# Page config
st.set_page_config(
    page_title="AI Bank ML Dashboard",
    page_icon="🏦",
    layout="wide"
)

# ======================
# HEADER
# ======================
st.title("🏦 AI Bank Marketing Intelligence Dashboard")
st.markdown("### Predict customer subscription using Machine Learning")

st.divider()

# ======================
# SIDEBAR INPUTS
# ======================
st.sidebar.header("📥 Customer Features")

inputs = {}

for col in model.feature_names_in_:
    inputs[col] = st.sidebar.number_input(
        f"{col}",
        value=0.0
    )

input_df = pd.DataFrame([inputs])

# ======================
# MAIN DASHBOARD
# ======================
col1, col2, col3 = st.columns(3)

if st.button("🚀 Run Prediction"):

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    # ======================
    # KPI CARDS
    # ======================
    with col1:
        st.metric("Prediction", "YES" if prediction == 1 else "NO")

    with col2:
        st.metric("Subscription Probability", f"{probability:.2f}")

    with col3:
        st.metric("Confidence Level", f"{probability*100:.1f}%")

    st.divider()

    # ======================
    # VISUAL SECTION
    # ======================
    st.subheader("📊 Prediction Insights")

    st.progress(int(probability * 100))

    if probability > 0.7:
        st.success("High likelihood of subscription")
    elif probability > 0.4:
        st.warning("Medium likelihood of subscription")
    else:
        st.error("Low likelihood of subscription")

    st.divider()

    # ======================
    # DATA VIEW
    # ======================
    st.subheader("📋 Input Data Summary")
    st.dataframe(input_df)

    # ======================
    # DOWNLOAD RESULT
    # ======================
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

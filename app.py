import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("bank_model.pkl")

st.set_page_config(page_title="AI Bank System", layout="wide")

# =====================
# HEADER
# =====================
st.title("🏦 AI Bank Marketing Intelligence System")
st.markdown("Predict customer subscription with ML-powered insights")

# =====================
# SIDEBAR INPUTS
# =====================
st.sidebar.header("Customer Input Features")

inputs = {}

for col in model.feature_names_in_:
    inputs[col] = st.sidebar.number_input(col, value=0.0)

input_df = pd.DataFrame([inputs])

# =====================
# MAIN AREA
# =====================
col1, col2 = st.columns(2)

# Prediction
if st.button("🚀 Predict"):
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    with col1:
        st.subheader("📊 Prediction Result")

        if pred == 1:
            st.success("Customer WILL Subscribe")
        else:
            st.error("Customer will NOT Subscribe")

        st.metric("Probability of Subscription", f"{prob:.2f}")

    with col2:
        st.subheader("📈 Model Insight")

        st.write("Feature values:")
        st.dataframe(input_df)

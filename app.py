import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ==========================================
# LOAD MODEL & ENCODERS
# ==========================================

model = joblib.load("bank_model.pkl")
encoders = joblib.load("encoders.pkl")

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="AI Bank Marketing Dashboard",
    page_icon="🏦",
    layout="wide"
)

# ==========================================
# SESSION STATE
# ==========================================

if "history" not in st.session_state:
    st.session_state.history = []

# ==========================================
# HEADER
# ==========================================

st.title("🏦 AI Bank Marketing Intelligence Dashboard")
st.markdown("Machine Learning System for Customer Subscription Prediction")

st.divider()

# ==========================================
# SIDEBAR INPUTS
# ==========================================

st.sidebar.header("📥 Customer Features")

age = st.sidebar.slider("Age", 18, 100, 35)

job = st.sidebar.selectbox(
    "Job",
    [
        "admin.",
        "blue-collar",
        "entrepreneur",
        "housemaid",
        "management",
        "retired",
        "self-employed",
        "services",
        "student",
        "technician",
        "unemployed",
        "unknown"
    ]
)

marital = st.sidebar.selectbox(
    "Marital Status",
    ["single", "married", "divorced"]
)

education = st.sidebar.selectbox(
    "Education",
    ["primary", "secondary", "tertiary", "unknown"]
)

default = st.sidebar.selectbox(
    "Credit Default",
    ["no", "yes"]
)

balance = st.sidebar.number_input(
    "Account Balance",
    value=1000
)

housing = st.sidebar.selectbox(
    "Housing Loan",
    ["no", "yes"]
)

loan = st.sidebar.selectbox(
    "Personal Loan",
    ["no", "yes"]
)

contact = st.sidebar.selectbox(
    "Contact Type",
    ["cellular", "telephone", "unknown"]
)

day = st.sidebar.slider(
    "Last Contact Day",
    1,
    31,
    15
)

month = st.sidebar.selectbox(
    "Month",
    [
        "jan",
        "feb",
        "mar",
        "apr",
        "may",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec"
    ]
)

duration = st.sidebar.number_input(
    "Call Duration (Seconds)",
    min_value=0,
    value=180
)

campaign = st.sidebar.number_input(
    "Campaign Contacts",
    min_value=1,
    value=1
)

pdays = st.sidebar.number_input(
    "Days Since Previous Contact",
    value=999
)

previous = st.sidebar.number_input(
    "Previous Contacts",
    min_value=0,
    value=0
)

poutcome = st.sidebar.selectbox(
    "Previous Campaign Outcome",
    ["failure", "other", "success", "unknown"]
)

# ==========================================
# ENCODE INPUTS
# ==========================================

try:

    input_df = pd.DataFrame([{
        "age": age,
        "job": encoders["job"].transform([job])[0],
        "marital": encoders["marital"].transform([marital])[0],
        "education": encoders["education"].transform([education])[0],
        "default": encoders["default"].transform([default])[0],
        "balance": balance,
        "housing": encoders["housing"].transform([housing])[0],
        "loan": encoders["loan"].transform([loan])[0],
        "contact": encoders["contact"].transform([contact])[0],
        "day": day,
        "month": encoders["month"].transform([month])[0],
        "duration": duration,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "poutcome": encoders["poutcome"].transform([poutcome])[0]
    }])

except Exception as e:
    st.error(f"Encoding Error: {e}")
    st.stop()

# ==========================================
# MAIN DASHBOARD
# ==========================================

st.subheader("📊 Prediction Dashboard")

if st.button("🚀 Run Prediction"):

    try:

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

    except Exception as e:

        st.error(f"Prediction Error: {e}")

        st.write("Expected Features")
        st.write(list(model.feature_names_in_))

        st.write("Input Data")
        st.dataframe(input_df)

        st.stop()

    st.session_state.history.append(probability)

    # ==========================================
    # KPI CARDS
    # ==========================================

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Prediction",
            "YES" if prediction == 1 else "NO"
        )

    with col2:
        st.metric(
            "Probability",
            f"{probability:.2%}"
        )

    with col3:
        avg_prob = sum(st.session_state.history) / len(st.session_state.history)

        st.metric(
            "Average Probability",
            f"{avg_prob:.2%}"
        )

    st.divider()

    # ==========================================
    # BUSINESS INSIGHT
    # ==========================================

    st.subheader("🧠 Business Insight")

    if probability >= 0.70:
        st.success(
            "🔥 High Value Customer. Prioritize immediate sales engagement."
        )

    elif probability >= 0.40:
        st.warning(
            "⚠️ Medium Potential Customer. Retarget with marketing campaigns."
        )

    else:
        st.error(
            "❌ Low Priority Customer."
        )

    st.divider()

    # ==========================================
    # CONFIDENCE SCORE
    # ==========================================

    st.subheader("📈 Prediction Confidence")

    st.progress(int(probability * 100))

    st.write(
        f"Confidence Score: {probability * 100:.1f}%"
    )

    st.divider()

    # ==========================================
    # INPUT SUMMARY
    # ==========================================

    st.subheader("📋 Input Summary")

    display_df = pd.DataFrame([{
        "Age": age,
        "Job": job,
        "Marital": marital,
        "Education": education,
        "Balance": balance,
        "Housing Loan": housing,
        "Personal Loan": loan,
        "Contact": contact,
        "Month": month,
        "Duration": duration
    }])

    st.dataframe(display_df)

    st.divider()

    # ==========================================
    # FEATURE IMPORTANCE
    # ==========================================

    if hasattr(model, "feature_importances_"):

        st.subheader("📊 Feature Importance")

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.barh(
            model.feature_names_in_,
            model.feature_importances_
        )

        ax.set_title("Feature Importance")

        st.pyplot(fig)

    st.divider()

    # ==========================================
    # ANALYTICS
    # ==========================================

    st.subheader("📊 Analytics Dashboard")

    fig, ax = plt.subplots()

    ax.hist(
        st.session_state.history,
        bins=10
    )

    ax.set_title(
        "Prediction Probability Distribution"
    )

    ax.set_xlabel(
        "Probability"
    )

    ax.set_ylabel(
        "Frequency"
    )

    st.pyplot(fig)

    st.divider()

    # ==========================================
    # CUSTOMER SEGMENTS
    # ==========================================

    st.subheader("🧠 Customer Segmentation")

    high = len(
        [x for x in st.session_state.history if x > 0.7]
    )

    medium = len(
        [x for x in st.session_state.history if 0.4 <= x <= 0.7]
    )

    low = len(
        [x for x in st.session_state.history if x < 0.4]
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("🔥 High Value", high)
    c2.metric("⚠️ Medium Value", medium)
    c3.metric("❌ Low Value", low)

    st.divider()

    # ==========================================
    # DOWNLOAD REPORT
    # ==========================================

    report_df = pd.DataFrame([{
        "prediction": prediction,
        "probability": round(probability, 4),
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "balance": balance,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "month": month,
        "duration": duration
    }])

    csv = report_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇ Download Prediction Report",
        data=csv,
        file_name="bank_prediction_report.csv",
        mime="text/csv"
    )
```

import streamlit as st
import pandas as pd
import joblib
from transformers import pipeline

model = joblib.load("bank_model.pkl")
llm = pipeline("text-generation", model="distilgpt2")

st.title("🏦 AI Bank Marketing System")

inputs = {}

for col in model.feature_names_in_:
    inputs[col] = st.number_input(col)

input_df = pd.DataFrame([inputs])

def explain(sample, prediction):
    prompt = f"""
    Banking AI assistant.

    Data: {sample.to_dict()}

    Prediction: {prediction}

    Explain simply.
    """
    return llm(prompt, max_length=120)[0]["generated_text"]

if st.button("Predict"):
    pred = model.predict(input_df)[0]

    result = "YES" if pred == 1 else "NO"
    st.success(result)

    st.write(explain(input_df, result))

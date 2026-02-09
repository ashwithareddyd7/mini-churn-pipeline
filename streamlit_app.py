import streamlit as st
import requests
import pandas as pd
import sqlite3

API_URL = "http://127.0.0.1:8000/predict"

st.title("📡 Telecom Customer Churn Predictor")
st.write("Enter customer details and evaluate churn risk in real time.")

# Load feature schema
conn = sqlite3.connect("model_store/feature_store.db")
df = pd.read_sql("SELECT * FROM features LIMIT 1", conn)
conn.close()
feature_cols = list(df.drop(columns=['churn_flag']).columns)

user_inputs = {}
st.subheader("Customer Feature Inputs")

for col in feature_cols:
    # Numeric or binary fields can default to 0
    val = st.number_input(col, value=0.0)
    user_inputs[col] = val

if st.button("Predict Churn"):
    payload = {"features": user_inputs}
    response = requests.post(API_URL, json=payload)

    if response.status_code == 200:
        prob = response.json()['churn_probability']
        st.success(f"Estimated Churn Probability: **{prob:.2f}**")
    else:
        st.error("API error. Check server logs.")
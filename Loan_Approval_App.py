import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load trained model
model = joblib.load("Loan_Approval_model.pkl")

st.title("🏦 Loan Approval Prediction App")

# -------- INPUTS --------
no_of_dependents = st.number_input(
    "Number of Dependents", min_value=0, step=1
)

education = st.selectbox(
    "Education",
    options=["Graduate", "Not Graduate"]
)

self_employed = st.selectbox(
    "Self Employed",
    options=["Yes", "No"]
)

income_annum = st.number_input(
    "Annual Income", min_value=0, step=10000
)

loan_amount = st.number_input(
    "Loan Amount", min_value=0, step=50000
)

loan_term = st.number_input(
    "Loan Term (in months)", min_value=1, step=12
)

cibil_score = st.number_input(
    "CIBIL Score", min_value=300, max_value=900
)

residential_assets_value = st.number_input(
    "Residential Assets Value", min_value=0
)

commercial_assets_value = st.number_input(
    "Commercial Assets Value", min_value=0
)

luxury_assets_value = st.number_input(
    "Luxury Assets Value", min_value=0
)

# -------- ENCODING --------
education = 1 if education == "Graduate" else 0
self_employed = 1 if self_employed == "Yes" else 0

# -------- PREDICTION --------
if st.button("Predict Loan Status"):

    input_data = np.array([[
        no_of_dependents,
        education,
        self_employed,
        income_annum,
        loan_amount,
        loan_term,
        cibil_score,
        residential_assets_value,
        commercial_assets_value,
        luxury_assets_value
    ]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")


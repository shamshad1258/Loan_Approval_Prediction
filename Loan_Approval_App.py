import streamlit as st
import numpy as np
import pickle

# Load trained model
with open("Loan_Approval_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🏦 Loan Approval Prediction")
st.write("Fill the details and click Predict")

no_of_dependents = st.number_input("Number of Dependents", 0, 10, 0)

education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])

income_annum = st.number_input("Annual Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_term = st.number_input("Loan Term (months)", min_value=1)
cibil_score = st.number_input("CIBIL Score", 300, 900)

residential_assets_value = st.number_input("Residential Assets Value", min_value=0)
commercial_assets_value = st.number_input("Commercial Assets Value", min_value=0)
luxury_assets_value = st.number_input("Luxury Assets Value", min_value=0)

# Encoding
education = 1 if education == "Graduate" else 0
self_employed = 1 if self_employed == "Yes" else 0

if st.button("Predict Loan Status"):
    X = np.array([[
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

    pred = model.predict(X)[0]

    if pred == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

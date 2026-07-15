import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model" / "random_forest_classifier.pkl"

model = joblib.load(MODEL_PATH)

st.sidebar.title("🏦 Loan Approval Predictor")

st.sidebar.markdown("---")

st.sidebar.write("### Algorithm")
st.sidebar.success("Random Forest Classifier")

st.sidebar.write("### Model Accuracy")
st.sidebar.info("84.5%")

st.sidebar.write("### Dataset")
st.sidebar.write("Loan Approval Dataset")

st.sidebar.markdown("---")

st.sidebar.caption("Developed by Ishan")

st.title("🏦 Loan Approval Prediction System")

st.write(
    "Predict whether a loan application is likely to be approved using Machine Learning."
)

st.markdown("---")

col1,col2 = st.columns(2)

with col1:

    gender = st.selectbox(
        "Gender",
        ["Male","Female"]
    )

    married = st.selectbox(
        "Married",
        ["Yes","No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["0","1","2","3+"]
    )

    education = st.selectbox(
        "Education",
        ["Graduate","Not Graduate"]
    )

    self_employed = st.selectbox(
        "Self Employed",
        ["Yes","No"]
    )

with col2:

    applicant_income = st.number_input(
        "Applicant Income",
        value=5000
    )

    coapplicant_income = st.number_input(
        "Coapplicant Income",
        value=0
    )

    loan_amount = st.number_input(
        "Loan Amount",
        value=150
    )

    loan_term = st.number_input(
        "Loan Amount Term",
        value=360
    )

    credit_history = st.selectbox(
        "Credit History",
        [1,0]
    )

    property_area = st.selectbox(
        "Property Area",
        [
            "Urban",
            "Semiurban",
            "Rural"
        ]
    )

gender = 1 if gender=="Male" else 0

married = 1 if married=="Yes" else 0

education = 1 if education=="Graduate" else 0

self_employed = 1 if self_employed=="Yes" else 0

dependents = 3 if dependents=="3+" else int(dependents)

property_area_semiurban = 0
property_area_urban = 0

if property_area=="Semiurban":
    property_area_semiurban=1

elif property_area=="Urban":
    property_area_urban=1

if st.button("🔍 Predict Loan Approval"):
    input_data = pd.DataFrame({

    "Gender":[gender],

    "Married":[married],

    "Dependents":[dependents],

    "Education":[education],

    "Self_Employed":[self_employed],

    "ApplicantIncome":[applicant_income],

    "CoapplicantIncome":[coapplicant_income],

    "LoanAmount":[loan_amount],

    "Loan_Amount_Term":[loan_term],

    "Credit_History":[credit_history],

    "Property_Area_Semiurban":[property_area_semiurban],

    "Property_Area_Urban":[property_area_urban]

})

 if st.button("🔍 Predict Loan Approval"):

    input_data = pd.DataFrame(...)

    prediction = model.predict(input_data)

    probability = model.predict_proba(input_data)
    confidence = probability.max() * 100

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
        st.balloons()
    else:
        st.error("❌ Loan Rejected")

    st.metric("Confidence", f"{confidence:.2f}%")

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Absolute path to the folder containing app.py
BASE_DIR = Path(__file__).resolve().parent

# Absolute path to the model
MODEL_PATH = BASE_DIR / "model" / "Decision_tree_classifier.pkl"

model = joblib.load(MODEL_PATH)

st.title("❤️ Heart Disease Prediction System ")

st.write("Enter the patient's details below to predict whether Heart Disease is Present or Absent.")

age = st.number_input("Age", min_value=1, max_value=100)

sex = st.selectbox(
    "Sex",
    [0,1],
    help="0 = Female, 1 = Male"
)

chest_pain = st.selectbox(
    "Chest Pain Type",
    [1,2,3,4]
)

bp = st.number_input("Resting Blood Pressure")

cholesterol = st.number_input("Serum Cholesterol")

fbs = st.selectbox(
    "Fasting Blood Sugar >120 mg/dl",
    [0,1]
)

ecg = st.selectbox(
    "Resting ECG",
    [0,1,2]
)

max_hr = st.number_input("Maximum Heart Rate")

exercise = st.selectbox(
    "Exercise Induced Angina",
    [0,1]
)

oldpeak = st.number_input(
    "Oldpeak",
    format="%.1f"
)

slope = st.selectbox(
    "Slope",
    [1,2,3]
)

vessels = st.selectbox(
    "Major Vessels",
    [0,1,2,3]
)

thal = st.selectbox(
    "Thal",
    [3,6,7]
)

if st.button("Predict"):
    input_data = pd.DataFrame({
    "Age":[age],
    "Sex":[sex],
    "Chest pain type":[chest_pain],
    "BP":[bp],
    "Cholesterol":[cholesterol],
    "FBS over 120":[fbs],
    "EKG results":[ecg],
    "Max HR":[max_hr],
    "Exercise angina":[exercise],
    "ST depression":[oldpeak],
    "Slope of ST":[slope],
    "Number of vessels fluro":[vessels],
    "Thallium":[thal]
})

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("Heart Disease Detected")
    else:
        st.success("No Heart Disease Detected")


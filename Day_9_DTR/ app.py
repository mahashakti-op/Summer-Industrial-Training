import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model" / "decision_tree_regressor.pkl"

model = joblib.load(MODEL_PATH)

st.title("💰 Medical Insurance Cost Prediction")

st.write("Enter the details below to estimate the insurance charges.")

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=25
)

sex = st.selectbox(
    "Sex",
    ["Male", "Female"]
)

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=60.0,
    value=25.0
)

children = st.number_input(
    "Children",
    min_value=0,
    max_value=10,
    value=0
)

smoker = st.selectbox(
    "Smoker",
    ["Yes", "No"]
)

region = st.selectbox(
    "Region",
    [
        "northeast",
        "northwest",
        "southeast",
        "southwest"
    ]
)

sex = 1 if sex == "Male" else 0

smoker = 1 if smoker == "Yes" else 0

region_northwest = 0
region_southeast = 0
region_southwest = 0

if region == "northwest":
    region_northwest = 1

elif region == "southeast":
    region_southeast = 1

elif region == "southwest":
    region_southwest = 1

if st.button("Predict"):

    input_data = pd.DataFrame({
        "age":[age],
        "sex":[sex],
        "bmi":[bmi],
        "children":[children],
        "smoker":[smoker],
        "region_northwest":[region_northwest],
        "region_southeast":[region_southeast],
        "region_southwest":[region_southwest]
    })

    prediction = model.predict(input_data)

    st.success(
    f"Estimated Insurance Charges: ${prediction[0]:,.2f}"
)
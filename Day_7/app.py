import streamlit as st
import joblib
import pandas as pd

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("linear_regression_model.pkl")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("📌 About")

st.sidebar.info("""
This application predicts the **House Price**
using a trained **Linear Regression Model**.

**Algorithm Used**
- Linear Regression

**Developer**
- Your Name
""")

# -----------------------------
# Main Title
# -----------------------------
st.title("🏠 House Price Prediction")

st.write(
    "Enter the house details below and click **Predict Price**."
)

st.divider()

# -----------------------------
# Input Fields
# -----------------------------
income = st.number_input(
    "💰 Average Area Income",
    min_value=0.0,
    value=50000.0
)

age = st.number_input(
    "🏡 Average Area House Age",
    min_value=0.0,
    value=5.0
)

rooms = st.number_input(
    "🛋 Average Number of Rooms",
    min_value=0.0,
    value=6.0
)

bedrooms = st.number_input(
    "🛏 Average Number of Bedrooms",
    min_value=0.0,
    value=3.0
)

population = st.number_input(
    "👨‍👩‍👧 Area Population",
    min_value=0.0,
    value=30000.0
)

st.divider()

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("🚀 Predict Price", use_container_width=True):

    input_data = pd.DataFrame(
        [[income, age, rooms, bedrooms, population]],
        columns=[
            "Avg. Area Income",
            "Avg. Area House Age",
            "Avg. Area Number of Rooms",
            "Avg. Area Number of Bedrooms",
            "Area Population"
        ]
    )

    prediction = model.predict(input_data)[0]

    st.success("Prediction Successful!")

    st.metric(
        label="🏡 Estimated House Price",
        value=f"${prediction:,.2f}"
    )

st.divider()

st.caption("Made with ❤️ using Streamlit")

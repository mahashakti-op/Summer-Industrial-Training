import streamlit as st
import pandas as pd
import joblib

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🛍️",
    layout="centered"
)

# ----------------------------
# Load Model
# ----------------------------
model_data = joblib.load("Day_11/kmeans.pkl")

kmeans = model_data["model"]
scaler = model_data["scaler"]
encoder = model_data["label_encoder"]
feature_columns = model_data["feature_columns"]

# ----------------------------
# Title
# ----------------------------
st.title("🛍️ Customer Segmentation using K-Means")
st.markdown("Enter customer details to identify the customer segment.")

st.divider()

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.header("About")
st.sidebar.info(
    """
    This application uses the **K-Means Clustering Algorithm**
    to group customers based on:

    - Gender
    - Age
    - Annual Income
    - Spending Score
    """
)

# ----------------------------
# User Input
# ----------------------------
gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

age = st.slider(
    "Age",
    18,
    70,
    30
)

income = st.slider(
    "Annual Income (k$)",
    10,
    150,
    50
)

score = st.slider(
    "Spending Score (1-100)",
    1,
    100,
    50
)

# ----------------------------
# Prediction
# ----------------------------
if st.button("Find Customer Segment"):

    gender_encoded = encoder.transform([gender])[0]

    sample = pd.DataFrame(
        [[gender_encoded, age, income, score]],
        columns=feature_columns
    )

    sample_scaled = scaler.transform(sample)

    cluster = kmeans.predict(sample_scaled)[0]

    cluster_names = {
        0: "💎 Premium Customers",
        1: "💰 High Income, Low Spending",
        2: "🛒 Budget Customers",
        3: "🎯 Young High Spenders",
        4: "👨‍👩‍👧 Average Customers"
    }
    

    st.success(f"Predicted Cluster: {cluster}")

    st.info(cluster_names[cluster])

st.divider()

st.caption("Developed using Streamlit <3")

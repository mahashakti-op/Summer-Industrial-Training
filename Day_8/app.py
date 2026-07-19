import streamlit as st
import pickle
import pandas as pd
from pathlib import Path

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="centered"
)

# -------------------------------
# Load Model
# -------------------------------
BASE_DIR = Path(__file__).parent

with open(BASE_DIR / "flower_model.pkl", "rb") as f:
    model = pickle.load(f)

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("🌸 About Project")

st.sidebar.info("""
### Machine Learning Model

**Algorithm Used**
- Logistic Regression

**Dataset**
- Iris Dataset

**Classification**
- 🌸 Iris Setosa
- 🌺 Iris Virginica

**Developer**
- Ishan Airan
""")

# -------------------------------
# Title
# -------------------------------
st.title("🌸 Iris Flower Classification")

st.write(
    "Enter the flower measurements below and click **Predict Species**."
)

st.divider()

# -------------------------------
# Two Column Layout
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input(
        "Sepal Length (cm)",
        min_value=0.0,
        value=5.1
    )

    sepal_width = st.number_input(
        "Sepal Width (cm)",
        min_value=0.0,
        value=3.5
    )

with col2:
    petal_length = st.number_input(
        "Petal Length (cm)",
        min_value=0.0,
        value=1.4
    )

    petal_width = st.number_input(
        "Petal Width (cm)",
        min_value=0.0,
        value=0.2
    )

st.divider()

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔍 Predict Species", use_container_width=True):

    input_data = pd.DataFrame(
        [[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]],
        columns=[
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width"
        ]
    )

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0]
    confidence = max(probability) * 100

    st.success("Prediction Completed Successfully! 🎉")

    st.divider()

    if prediction == 0:

        st.markdown("## 🌸 Predicted Flower")
        st.success("### Iris Setosa")

    else:

        st.markdown("## 🌺 Predicted Flower")
        st.success("### Iris Virginica")

    st.metric(
        "🎯 Prediction Confidence",
        f"{confidence:.2f}%"
    )

    st.divider()

    st.subheader("Prediction Probability")

    st.write("🌸 Iris Setosa")
    st.progress(float(probability[0]))

    st.write("🌺 Iris Virginica")
    st.progress(float(probability[1]))

st.divider()

st.caption("Made with ❤️ using Streamlit")

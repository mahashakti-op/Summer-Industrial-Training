import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(

    page_title="Used Car Price Predictor",

    page_icon="🚗",

    layout="wide"

)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR/"model"/"random_forest_regressor.pkl"

model = joblib.load(MODEL_PATH)

st.markdown("""

<style>

.main{

padding-top:20px;

}

h1{

text-align:center;

color:#1E88E5;

}

</style>

""",unsafe_allow_html=True)

st.sidebar.title("🚗 Used Car Price Predictor")

st.sidebar.markdown("---")

st.sidebar.success("Random Forest Regressor")

st.sidebar.info("R² Score : 82%")

st.sidebar.write("Dataset")

st.sidebar.write("Used Car Price Dataset")

st.sidebar.markdown("---")

st.sidebar.caption("Developed by Ishan")

st.title("🚗 Used Car Price Prediction")

st.write(
"Predict the estimated selling price of a used car using Machine Learning."
)

st.markdown("---")

col1,col2=st.columns(2)

with col1:

    make_year=st.number_input(
        "Manufacturing Year",
        2000,
        2025,
        2020
    )

    fuel_type=st.selectbox(

        "Fuel Type",

        ["Petrol","Diesel","CNG","Electric"]

    )

    brand = st.selectbox(
        "Brand",
        [
            "Chevrolet",
            "Ford",
            "Honda",
            "Hyundai",
            "Kia",
            "Nissan",
            "Tesla",
            "Toyota",
            "Volkswagen"
        ]
    )
    
    transmission=st.selectbox(

        "Transmission",

        [
            "Manual",
            "Automatic"
        ]

    )

    color=st.selectbox(

        "Color",

        [
            "Black",
            "Blue",
            "Gray",
            "Red",
            "Silver",
            "White"
        ]

    )

with col2:

    mileage = st.number_input(
        "Mileage (km/l)",
        value=18.0
    )

    engine = st.number_input(
        "Engine (CC)",
        value=1200
    )

    owner = st.number_input(
        "Owner Count",
        1,
        5,
        1
    )

    accidents = st.number_input(
        "Accidents Reported",
        min_value=0,
        max_value=20,
        value=0
    )

    service = st.selectbox(
        "Service History",
        ["Full", "Partial"]
    )

    insurance = st.selectbox(
        "Insurance Valid",
        ["Yes", "No"]
    )

# ==========================
# Encoding
# ==========================

fuel_type_electric = 0
fuel_type_petrol = 0

if fuel_type == "Electric":
    fuel_type_electric = 1
elif fuel_type == "Petrol":
    fuel_type_petrol = 1


brand_chevrolet = 0
brand_ford = 0
brand_honda = 0
brand_hyundai = 0
brand_kia = 0
brand_nissan = 0
brand_tesla = 0
brand_toyota = 0
brand_volkswagen = 0

if brand == "Chevrolet":
    brand_chevrolet = 1
elif brand == "Ford":
    brand_ford = 1
elif brand == "Honda":
    brand_honda = 1
elif brand == "Hyundai":
    brand_hyundai = 1
elif brand == "Kia":
    brand_kia = 1
elif brand == "Nissan":
    brand_nissan = 1
elif brand == "Tesla":
    brand_tesla = 1
elif brand == "Toyota":
    brand_toyota = 1
elif brand == "Volkswagen":
    brand_volkswagen = 1


transmission_manual = 1 if transmission == "Manual" else 0


color_blue = 0
color_gray = 0
color_red = 0
color_silver = 0
color_white = 0

if color == "Blue":
    color_blue = 1
elif color == "Gray":
    color_gray = 1
elif color == "Red":
    color_red = 1
elif color == "Silver":
    color_silver = 1
elif color == "White":
    color_white = 1


service_history_partial = 1 if service == "Partial" else 0

insurance_valid_yes = 1 if insurance == "Yes" else 0


# ==========================
# Prediction
# ==========================

if st.button("💰 Predict Price"):

    input_data = pd.DataFrame({

        "make_year": [make_year],
        "mileage_kmpl": [mileage],
        "engine_cc": [engine],
        "owner_count": [owner],
        "accidents_reported": [accidents],

        "fuel_type_Electric": [fuel_type_electric],
        "fuel_type_Petrol": [fuel_type_petrol],

        "brand_Chevrolet": [brand_chevrolet],
        "brand_Ford": [brand_ford],
        "brand_Honda": [brand_honda],
        "brand_Hyundai": [brand_hyundai],
        "brand_Kia": [brand_kia],
        "brand_Nissan": [brand_nissan],
        "brand_Tesla": [brand_tesla],
        "brand_Toyota": [brand_toyota],
        "brand_Volkswagen": [brand_volkswagen],

        "transmission_Manual": [transmission_manual],

        "color_Blue": [color_blue],
        "color_Gray": [color_gray],
        "color_Red": [color_red],
        "color_Silver": [color_silver],
        "color_White": [color_white],

        "service_history_Partial": [service_history_partial],

        "insurance_valid_Yes": [insurance_valid_yes]

    })

    prediction = model.predict(input_data)

    st.markdown("---")

    st.success("Prediction Completed Successfully ✅")

    st.metric(
        "Estimated Price",
        f"${prediction[0]:,.0f}"
    )
    
    st.balloons()

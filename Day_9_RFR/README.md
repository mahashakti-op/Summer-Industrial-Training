# 🚗 Used Car Price Prediction using Random Forest Regressor

## 📌 Project Overview

This project predicts the estimated selling price of a used car using a **Random Forest Regressor**. It performs data preprocessing, feature engineering, hyperparameter tuning using **RandomizedSearchCV**, evaluates the model using regression metrics, and deploys the prediction system through **Streamlit**.

Here's the link : 
---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib

---

## 📂 Dataset

**Used Car Price Prediction Dataset**

### Features

- Manufacturing Year
- Mileage (km/l)
- Engine Capacity (CC)
- Owner Count
- Accidents Reported
- Fuel Type
- Brand
- Transmission
- Color
- Service History
- Insurance Valid

### Target

- Price (USD)

---

## ⚙️ Data Preprocessing

- Handled Missing Values
- One-Hot Encoding using `pd.get_dummies()`
- Removed Redundant Categories using `drop_first=True`
- Train-Test Split (80:20)

---

## 🤖 Machine Learning Algorithm

**Random Forest Regressor**

---

## 🚀 Hyperparameter Tuning

This project uses **RandomizedSearchCV** with **5-Fold Cross Validation** to find the best model parameters.

### Tuned Parameters

- n_estimators
- max_depth
- min_samples_split
- min_samples_leaf
- max_features

---

## 📊 Model Evaluation

The model was evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

---

## 📈 Feature Importance

Random Forest feature importance was used to identify which vehicle attributes have the greatest impact on the predicted selling price.

---

## 🌐 Streamlit Application

The web application allows users to enter vehicle details and instantly predicts the estimated resale price of the car.

---

## ▶️ How to Run

Clone the repository

```bash
git clone <repository-link>
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the Streamlit application

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```text
RFR/
│
├── dataset/
│      used_car_price_dataset.csv
│
├── model/
│      random_forest_regressor.pkl
│
├── train_model.py
├── app.py
├── requirements.txt
├── README.md
└── notebook.ipynb
```

---

## 🎯 Project Highlights

- Random Forest Regressor
- Missing Value Handling
- One-Hot Encoding
- Hyperparameter Tuning
- RandomizedSearchCV
- 5-Fold Cross Validation
- Regression Metrics
- Feature Importance
- Interactive Streamlit GUI
- Live Deployment

---

## 👨‍💻 Developed By

**Ishan**

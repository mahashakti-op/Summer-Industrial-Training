# 🏠 House Price Prediction using Linear Regression

A Machine Learning project that predicts house prices based on various housing features using the **Linear Regression** algorithm. The project also includes an interactive **Streamlit web application** for real-time predictions.

Here's the link : https://summer-industrial-training-dvecxcpncf2kyxnowsqxnn.streamlit.app/
---

## 📌 Project Overview

The objective of this project is to build a regression model that can estimate the price of a house using different numerical features such as:

- Average Area Income
- Average Area House Age
- Average Area Number of Rooms
- Average Area Number of Bedrooms
- Area Population

The model is trained using the **USA Housing Dataset** and deployed with **Streamlit** for easy user interaction.

---

## 📂 Dataset

**Dataset:** USA Housing Dataset

### Features

| Feature | Description |
|---------|-------------|
| Avg. Area Income | Average income of residents in the area |
| Avg. Area House Age | Average age of houses in the area |
| Avg. Area Number of Rooms | Average number of rooms |
| Avg. Area Number of Bedrooms | Average number of bedrooms |
| Area Population | Population of the area |

### Target

- Price

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit

---

## 🤖 Machine Learning Algorithm

### Linear Regression

Linear Regression is a supervised machine learning algorithm used for predicting continuous numerical values.

In this project, it learns the relationship between housing features and house prices to make future predictions.

---

## 📊 Model Workflow

1. Import Libraries
2. Load Dataset
3. Data Exploration
4. Data Preprocessing
5. Feature Selection
6. Train-Test Split
7. Train Linear Regression Model
8. Evaluate Model
9. Save Model using Joblib
10. Deploy using Streamlit

---

## 📈 Evaluation Metrics

The model was evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

---

## 🌐 Streamlit Application

The web application allows users to:

- Enter house details
- Predict house price instantly
- View predictions through an interactive interface

---

## 📁 Project Structure

```text
House_Price_Prediction/
│
├── app.py
├── model/
│   └── linear_regression_model.pkl
├── USA_Housing.csv
├── Linear_Regression.ipynb
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

Clone the repository

```bash
git clone <repository-link>
```

Move into the project directory

```bash
cd House_Price_Prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## 📷 Application Preview

> Add screenshots of your Streamlit application here.

---

## 🎯 Future Improvements

- Better UI Design
- Model Comparison with other Regression Algorithms
- Feature Engineering
- Hyperparameter Optimization
- Cloud Deployment

---

## 👨‍💻 Developer

**Ishan Airan**

Computer Science Engineering Student

---

## ⭐ If you found this project helpful, consider giving it a star!

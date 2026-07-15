# 🏦 Loan Approval Prediction using Random Forest Classifier

## 📌 Project Overview

This project predicts whether a loan application is likely to be **Approved** or **Rejected** using a Random Forest Classifier. It includes data preprocessing, missing value handling, feature encoding, hyperparameter tuning using RandomizedSearchCV, model evaluation, and deployment with Streamlit.

Here's the link : https://summer-industrial-training-tjr5au3syqyacamelshxmv.streamlit.app/
---

## 🛠 Technologies Used

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

**Loan Approval Prediction Dataset**

### Features

- Gender
- Married
- Dependents
- Education
- Self Employed
- Applicant Income
- Coapplicant Income
- Loan Amount
- Loan Amount Term
- Credit History
- Property Area

### Target

- Loan Status (Approved / Rejected)

---

## ⚙ Data Preprocessing

- Removed Loan_ID column
- Handled Missing Values
- Mode Imputation for Categorical Features
- Median Imputation for Numerical Features
- Label Encoding
- One-Hot Encoding

---

## 🤖 Machine Learning Algorithm

**Random Forest Classifier**

---

## 🚀 Hyperparameter Tuning

This project uses **RandomizedSearchCV** with **5-Fold Cross Validation** to automatically find the best model parameters.

### Tuned Parameters

- n_estimators
- max_depth
- min_samples_split
- min_samples_leaf
- max_features

---

## 📊 Model Evaluation

- Accuracy Score
- Confusion Matrix
- Classification Report
- Feature Importance

---

## 🌐 Streamlit Application

The web application allows users to enter applicant details and instantly predicts whether the loan is likely to be approved.

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

Run the application

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```text
RFC/
│
├── dataset/
│      Training Dataset.csv
│
├── model/
│      random_forest_classifier.pkl
│
├── train_model.py
├── app.py
├── requirements.txt
├── README.md
└── notebook.ipynb
```

---

## 🎯 Project Highlights

- Random Forest Classifier
- Missing Value Handling
- Data Encoding
- Hyperparameter Tuning
- RandomizedSearchCV
- 5-Fold Cross Validation
- Feature Importance
- Interactive Streamlit GUI
- Live Deployment

---

## 👨‍💻 Developed By

**Ishan**

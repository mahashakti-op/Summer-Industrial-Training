# 🛍️ Customer Segmentation using K-Means Clustering

A Machine Learning project that segments customers into different groups based on their purchasing behavior using the **K-Means Clustering Algorithm**. The project also includes an interactive **Streamlit Web Application** that allows users to identify the customer segment by entering customer details.

Here's the link : https://summer-industrial-training-z87hqn6ocrq6hammuspuqp.streamlit.app/
---

## 📌 Project Overview

Customer segmentation is an unsupervised machine learning technique used to group customers with similar characteristics. Businesses use customer segmentation to better understand customer behavior and create targeted marketing strategies.

In this project, customers are grouped based on:

- Gender
- Age
- Annual Income
- Spending Score

The trained model is deployed using **Streamlit**, allowing users to predict the customer segment interactively.

---

## 📂 Dataset

**Dataset Name:** Mall Customers Dataset

### Features

| Feature | Description |
|----------|-------------|
| CustomerID | Unique Customer ID |
| Genre | Customer Gender |
| Age | Customer Age |
| Annual Income (k$) | Annual Income in Thousand Dollars |
| Spending Score (1-100) | Spending Score assigned by the mall |

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Joblib

---

## 🤖 Machine Learning Workflow

- Import Libraries
- Load Dataset
- Exploratory Data Analysis (EDA)
- Data Preprocessing
- Label Encoding
- Feature Scaling
- Elbow Method
- K-Means Clustering
- Cluster Visualization
- Model Saving
- Streamlit Deployment

---

## 📊 Exploratory Data Analysis

The following analyses were performed:

- Dataset Information
- Missing Value Check
- Duplicate Value Check
- Statistical Summary
- Gender Distribution
- Age Distribution
- Income Distribution
- Spending Score Distribution
- Correlation Heatmap

---

## ⚙️ Model Building

The K-Means Clustering algorithm was trained after scaling the numerical features using **StandardScaler**.

The **Elbow Method** was used to determine the optimal number of clusters.

---

## 🚀 Streamlit Web Application

The web application allows users to enter:

- Gender
- Age
- Annual Income
- Spending Score

The trained model predicts the customer segment instantly.

---

## 📁 Project Structure

```
Day_11_KMeans/
│
├── KMeans.ipynb
├── app.py
├── kmeans.pkl
├── Mall_Customers.csv
├── requirements.txt
└── README.md
```

---

## ▶️ Installation

Clone the repository:

```bash
git clone <repository_link>
```

Move into the project folder:

```bash
cd Day_11_KMeans
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

## 📷 Application Preview

The application allows users to:

- Enter customer details
- Predict customer segment
- Display the predicted cluster instantly

---

## 📈 Future Improvements

- Better customer segment naming
- Cluster profiling dashboard
- PCA visualization
- Interactive cluster plots
- Download prediction report

---

## 🎯 Learning Outcomes

Through this project, I learned:

- Unsupervised Machine Learning
- Customer Segmentation
- K-Means Clustering
- Elbow Method
- Feature Scaling
- Cluster Visualization
- Streamlit Deployment
- Model Serialization using Joblib

---

## 👨‍💻 Developed By

**Harsh**

Machine Learning & AI Enthusiast 🚀

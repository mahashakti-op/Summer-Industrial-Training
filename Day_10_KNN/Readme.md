# 🎬 Movie Recommendation System using K-Nearest Neighbors (KNN)

A Machine Learning project that recommends movies similar to a selected movie using the **K-Nearest Neighbors (KNN)** algorithm. The project is deployed as an interactive **Streamlit Web Application**, allowing users to select a movie and instantly receive similar movie recommendations.

Here's the link : https://summer-industrial-training-fuautnftxrrlgv8kpebddg.streamlit.app/
---

## 📌 Project Overview

This project uses the K-Nearest Neighbors algorithm to build a **content-based movie recommendation system**.

Instead of predicting ratings, the model compares movie features such as genre, director, lead actor, budget, revenue, remake status, and franchise information to recommend movies with similar characteristics.

---

## 📂 Dataset

The dataset contains information about **1698 movies**.

### Features Used

- Movie Name
- Genre
- Release Period
- Whether Remake
- Whether Franchise
- New Actor
- New Director
- New Music Director
- Lead Star
- Director
- Music Director
- Number of Screens
- Revenue (INR)
- Budget (INR)

---

## 🤖 Machine Learning Algorithm

### K-Nearest Neighbors (KNN)

KNN is a supervised machine learning algorithm that can also be used for similarity-based recommendation systems.

The model identifies the nearest movies by calculating the distance between feature vectors and recommends the most similar movies.

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit

---

## 📊 Project Workflow

1. Import Libraries
2. Load Dataset
3. Data Cleaning
4. Handle Missing Values
5. Label Encoding
6. Feature Selection
7. Feature Scaling using StandardScaler
8. Train KNN Model
9. Save Model using Joblib
10. Build Streamlit Web Application

---

## 📁 Project Structure

```text
Movie_Recommendation_System/
│
├── app.py
├── KNN.ipynb
├── knn.pkl
├── Data for repository.csv
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/your-username/Movie-Recommendation-System.git
```

Move into the project folder

```bash
cd Movie-Recommendation-System
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

## 🎯 Features

- 🎬 Movie Recommendation
- ⚡ Fast KNN Search
- 📊 Content-Based Recommendation
- 🎨 Interactive Streamlit Interface
- 📱 User-Friendly Design

---

## 🌐 Application Preview

> Add screenshots of the Streamlit application here.

---

## 🔮 Future Improvements

- Movie Posters using TMDB API
- Search Bar with Auto Complete
- Cosine Similarity Recommendation
- Hybrid Recommendation System
- Deployment on Streamlit Cloud

---

## 👨‍💻 Developer

**Ishan Airan**

Computer Science Engineering Student

---

## ⭐ If you found this project useful, consider giving it a star!

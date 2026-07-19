import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
BASE_DIR = Path(__file__).parent

model_data = joblib.load(BASE_DIR / "knn.pkl")

model = model_data["model"]
scaler = model_data["scaler"]
movie_names = model_data["movie_names"]
feature_columns = model_data["feature_columns"]
encoders = model_data["encoders"]

df = pd.read_csv(BASE_DIR / "Data for repository.csv")

# -----------------------------
# Title
# -----------------------------
st.title("🎬 Movie Recommendation System")

st.markdown(
    """
Find movies similar to your favourite movie using the **K-Nearest Neighbors (KNN)** Machine Learning Algorithm.
"""
)

st.divider()

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("📖 About Project")

st.sidebar.info(
"""
### Machine Learning

**Algorithm**
- K-Nearest Neighbors (KNN)

**Dataset**
- 1698 Movies

**Recommendation Type**
- Content Based Recommendation

**Developer**
- Ishan Airan
"""
)

# -----------------------------
# Movie Selection
# -----------------------------
movie = st.selectbox(
    "🎥 Select a Movie",
    sorted(movie_names)
)

# -----------------------------
# Recommendation
# -----------------------------
if st.button("🚀 Recommend Movies", use_container_width=True):

    movie_index = movie_names[movie_names == movie].index[0]

    feature_row = df.loc[movie_index, feature_columns].copy()

    # Encode categorical values
    for col, encoder in encoders.items():
        if col in feature_row.index:
            feature_row[col] = encoder.transform([str(feature_row[col])])[0]

    feature_row = pd.DataFrame([feature_row])

    feature_scaled = scaler.transform(feature_row)

    distances, indices = model.kneighbors(feature_scaled)

    st.success("Top 5 Similar Movies")

    st.divider()

    for i, idx in enumerate(indices[0][1:], start=1):

        movie_data = df.iloc[idx]

        with st.container():

            st.subheader(f"🎬 {i}. {movie_data['Movie_Name']}")

            col1, col2, col3 = st.columns(3)

            col1.metric("Genre", movie_data["Genre"])

            col2.metric("Director", movie_data["Director"])

            col3.metric("Lead Star", movie_data["Lead_Star"])

            st.write(f"**Budget:** ₹ {movie_data['Budget(INR)']:,}")

            st.write(f"**Revenue:** ₹ {movie_data['Revenue(INR)']:,}")

            st.write(f"**Number of Screens:** {movie_data['Number_of_Screens']}")

            st.divider()

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")

st.caption("Made with ❤️ using Streamlit")
import streamlit as st
import pandas as pd 
import numpy as np
import pickle
st.title("Flower species prediction")
with open('flower_model.pkl','rb') as file:
    model=pickle.load(file)
st.write("Enter the flower features to predict the species")
sepal_length = st.number_input("Sepal length", min_value=0.0, max_value=10.0, step=0.1, key="sepal_length")
sepal_width=st.number_input("sepal width", min_value=0.0,max_value=10.0,step=0.1,key='sepal_width')
petal_length=st.number_input("petal length", min_value=0.0,max_value=10.0,step=0.1,key='petal_length')
petal_width=st.number_input("petal width", min_value=0.0,max_value=10.0,step=0.1,key='petal_width')
if st.button("Predict the class"):
    features=np.array([[sepal_length,sepal_width,petal_length,petal_width]])
    prediction=model.predict(features)
    if prediction[0]==0:
        st.write(f"The predicted flower species is: Setosa")
    elif prediction[0]==1:
        st.write(f"The predicted flower species is: Versicolor")
    
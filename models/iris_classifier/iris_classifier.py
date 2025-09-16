import streamlit as st
import numpy as np
import joblib

# Load trained models
def Iris_Classifier():
     
    model_L = joblib.load("models/iris_classifier/iris_model_logistic.h5")  
    model_R = joblib.load("models/iris_classifier/iris_model_random_forest.h5")  

    st.write("Enter the flower measurements to predict its species.")

    # Input sliders (keep as float for precision)
    sepal_length = int(st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1))
    sepal_width  = int(st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5))
    petal_length = int(st.slider("Petal Length (cm)", 1.0, 7.0, 1.4))
    petal_width  = int(st.slider("Petal Width (cm)", 0.1, 2.5, 0.2))

    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Choose model
    model_choice = st.radio("Select a model:", ("Logistic Regression", "Random Forest"))

    # Predict button
    if st.button("Predict"):
        if model_choice == "Logistic Regression":
            prediction = model_L.predict(features)[0]
        else:
            prediction = model_R.predict(features)[0]

        st.success(f"🌼 Predicted species: **{prediction}**")

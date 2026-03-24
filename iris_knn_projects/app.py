import joblib 
import numpy as np 
import streamlit as st 
# Load saved model and scaler 
model = joblib.load("models/knn_model.pkl") 
scaler = joblib.load("models/scaler.pkl") 
species_names = ["Setosa", "Versicolor", "Virginica"] 
st.set_page_config(page_title="Iris Flower Classifier", page_icon="🌸", layout="centered") 
st.title("Iris Flower Classification using KNN") 
st.write("Enter flower measurements to predict the species.") 
# Input fields 
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.1, 
step=0.1) 
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5, 
step=0.1) 
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.4, 
step=0.1) 
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2, 
step=0.1) 
if st.button("Predict"): 
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]]) 
    input_scaled = scaler.transform(input_data) 
    prediction = model.predict(input_scaled)[0] 
    prediction_proba = model.predict_proba(input_scaled)[0] 
    st.success(f"Predicted Species: **{species_names[prediction]}**") 
    st.subheader("Prediction Probabilities") 
    for i, prob in enumerate(prediction_proba):
        st.write(f"{species_names[i]}: {prob:.2%}")

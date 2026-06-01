
import streamlit as st
import pandas as pd
import pickle

st.set_page_config(
    page_title="Insurance Approval Prediction",
    layout="centered"
)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

st.title("Insurance Approval Prediction")

age = st.number_input("Age", min_value=18, max_value=100, value=30)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=60.0,
    value=25.0
)

children = st.number_input(
    "Children",
    min_value=0,
    max_value=10,
    value=0
)

smoker = st.selectbox(
    "Smoker",
    ["Yes", "No"]
)

region = st.selectbox(
    "Region",
    ["Northwest", "Northeast", "Southwest", "Southeast"]
)

income = st.number_input(
    "Income",
    min_value=0,
    value=50000
)

medical_history = st.selectbox(
    "Medical History",
    ["None", "Diabetes", "Hypertension", "Heart Disease"]
)

exercise = st.selectbox(
    "Exercise",
    ["Never", "Rarely", "Regularly"]
)

if st.button("Predict"):

    input_data = pd.DataFrame({
        "age": [age],
        "gender": [gender],
        "bmi": [bmi],
        "children": [children],
        "smoker": [smoker],
        "region": [region],
        "income": [income],
        "medical_history": [medical_history],
        "exercise": [exercise]
    })

    processed_data = preprocessor.transform(input_data)

    prediction = model.predict(processed_data)[0]

    if prediction == 1:
        st.success("Insurance Approved")
    else:
        st.error("Insurance Not Approved")

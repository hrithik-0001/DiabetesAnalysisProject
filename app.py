# import all essential library
import numpy as np
import pandas as pd
import joblib
import streamlit as st

st.title("Diabetes prediction  Application")
st.header("Machine Learning Project")

model = joblib.load('Logisticmodel.sav')

# Pregnancies	
# Glucose	
# BloodPressure	SkinThickness	Insulin	BMI	DiabetesPedigreeFunction	Age	Outcome

pregnancies = st.number_input("Pregnancies",min_value=0,max_value=17)
Glucose = st.number_input("Glucose",min_value=0,max_value=200)
BloodPressure = st.number_input("BloodPressure",min_value=0,max_value=125)
SkinThickness = st.number_input("SkinThickness",min_value=0,max_value=100)
Insulin = st.number_input("Insulin",min_value=0,max_value=846)
BMI = st.number_input("BMI",min_value=0,max_value=67)
DiabetesPedigreeFunction = st.number_input("DiabetesPedigreeFunction",min_value=0.078,max_value=2.42)
Age = st.number_input("Age",min_value=21,max_value=81)


if st.button("Predict"):
    new_array=np.array([[pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
    prediction = model.predict(new_array)

    if prediction == 0:
        st.success("Negative")

    else:
        st.success("Positive")



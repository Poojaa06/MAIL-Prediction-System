import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Load data
heart_data = pd.read_csv(r"C:\Users\DELL 3520\OneDrive\Desktop\streamlit\dataset\heart_disease_data.csv", encoding = 'latin1')

x = heart_data.drop('target',axis=1)
y = heart_data['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 2)

model = LogisticRegression()
model.fit(x_train, y_train)
x_train_pred = model.predict(x_train)
x_train_accuracy = accuracy_score(x_train_pred, y_train)

x_test_pred = model.predict(x_test)
x_test_accuracy = accuracy_score(x_test_pred, y_test)

#streamlit interface
st.title("Heart disease report")
input_data = st.text_area("Enter your test report result")
if st.button("Predict"):
    try:
        input_data = [float(i) for i in input_data.split(',')]
        if len(input_data) != x.shape[1]:
            st.warning(f"⚠ Expected {x.shape[1]} input values, but got {len(input_data)}.")
        else:
            reshaped_array = pd.DataFrame([input_data], columns=x.columns)
            prediction = model.predict(reshaped_array)
            probability = model.predict_proba(reshaped_array)[0][1]

            if prediction[0] == 1:
                st.warning("⚠ You are suffering from heart disease")
            else:
                st.success("✅ Your report is normal")

            st.write(f"🧠 Prediction Probability of Heart Disease: {probability:.2f}")
    except ValueError:
        st.error("🚫 Invalid input. Please enter numeric values only, separated by commas.")






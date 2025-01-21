import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
def load_data():
    file_path = 'insurance1.csv'
    return pd.read_csv(file_path)

data = load_data()

# Train model
X = data[['age', 'sex', 'bmi', 'children', 'smoker']]
y = data['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit app
st.title("Insurance Charges Prediction")

# Input fields
age = st.number_input("Age", min_value=0, max_value=120, value=25)
sex = st.selectbox("Sex", options=["Female", "Male"])
bmi = st.number_input("BMI", min_value=0.0, max_value=60.0, value=25.0, step=0.1)
children = st.number_input("Children", min_value=0, max_value=10, value=0)
smoker = st.selectbox("Smoker", options=["No", "Yes"])

# Map input to numeric values
sex_numeric = 1 if sex == "Male" else 0
smoker_numeric = 1 if smoker == "Yes" else 0

# Predict charges
if st.button("Predict"):
    input_data = np.array([[age, sex_numeric, bmi, children, smoker_numeric]])
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Insurance Charges: ${prediction:,.2f}")
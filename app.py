
import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# Load dataset
df = pd.read_csv("dataset/maternal_health.csv")

# Encode labels
encoder = LabelEncoder()
df['RiskLevel'] = encoder.fit_transform(df['RiskLevel'])

# Features and target
X = df.drop("RiskLevel", axis=1)
y = df["RiskLevel"]

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# Streamlit UI
st.title("Maternal Health Risk Triage System")

st.write("Enter patient health details below:")

age = st.number_input("Age", min_value=1, max_value=100)
systolic = st.number_input("Systolic Blood Pressure")
diastolic = st.number_input("Diastolic Blood Pressure")
bs = st.number_input("Blood Sugar Level")
temp = st.number_input("Body Temperature")
heart_rate = st.number_input("Heart Rate")

if st.button("Predict Risk"):

    custom_data = pd.DataFrame([{
        'Age': age,
        'SystolicBP': systolic,
        'DiastolicBP': diastolic,
        'BS': bs,
        'BodyTemp': temp,
        'HeartRate': heart_rate
    }])

    prediction = model.predict(custom_data)

    risk = encoder.inverse_transform(prediction)

    st.success(f"Predicted Risk Level: {risk[0]}")

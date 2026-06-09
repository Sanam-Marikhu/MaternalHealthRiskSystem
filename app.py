import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

st.set_page_config(
    page_title="Maternal Health Risk Triage System",
    page_icon="🤰",
    layout="centered"
)

# Load dataset
df = pd.read_csv("dataset/maternal_health.csv")

model = joblib.load("model.pkl")
encoder = joblib.load("encoder.pkl")
# Streamlit UI
st.title("Maternal Health Risk Triage System")

st.warning(
    "⚠️ This application is for educational purposes only and "
    "should not be used as a substitute for professional medical advice."
)

st.markdown("""
This application predicts maternal health risk levels using Machine Learning.

### Risk Categories
- 🟢 Low Risk
- 🟡 Medium Risk
- 🔴 High Risk
""")

st.divider()

st.write("Enter patient health details below:")

age = st.number_input("Age", min_value=18, max_value=50,value=25)
systolic = st.number_input("Systolic Blood Pressure", min_value=50, max_value=130,value=80)
diastolic = st.number_input("Diastolic Blood Pressure",min_value=50,max_value=130,value=80 )
bs = st.number_input("Blood Sugar Level",min_value=3.0, max_value=30.0,value=6.0)
temp = st.number_input("Body Temperature",min_value=95.0,max_value=105.0,value=98.0)
heart_rate = st.number_input("Heart Rate",min_value=40,max_value=180,value=80)

if st.button("Predict Risk"):
    
    if diastolic >= systolic:
        st.error(
            "Diastolic Blood Pressure must be lower than Systolic Blood Pressure."
        )
        st.stop()

    Patient_data = pd.DataFrame([{
        'Age': age,
        'SystolicBP': systolic,
        'DiastolicBP': diastolic,
        'BS': bs,
        'BodyTemp': temp,
        'HeartRate': heart_rate
    }])

    prediction = model.predict(Patient_data)
    probabilities = model.predict_proba(Patient_data)
    confidence = max(probabilities[0]) * 100
    risk = encoder.inverse_transform(prediction)[0]
    st.divider()

    st.subheader("Prediction Result")

    st.info(f"Prediction Confidence: {confidence:.2f}%")

    if risk == "high risk":
        st.error("🔴 High Risk Pregnancy")
        st.write(
            "Immediate medical consultation and continuous monitoring are recommended."
        )

    elif risk == "mid risk":
        st.warning("🟡 Medium Risk Pregnancy")
        st.write(
            "Regular monitoring and follow-up with healthcare professionals are advised."
        )

    else:
        st.success("🟢 Low Risk Pregnancy")
        st.write(
            "Continue routine prenatal checkups and maintain a healthy lifestyle."
        )

    st.subheader("Patient Details")
    st.dataframe(Patient_data)
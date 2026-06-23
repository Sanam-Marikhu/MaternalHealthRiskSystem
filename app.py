import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import numpy as np
import shap
from history_manager import save_prediction

st.set_page_config(
    page_title="Maternal Health Risk Triage System",
    page_icon="🤰",
    layout="centered"
)
# SESSION STATE INIT
if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False

if "result_data" not in st.session_state:
    st.session_state.result_data = {}

# SIDEBAR
page = st.sidebar.selectbox(
    "Navigation",
    ["Prediction", "Analytics"]
)

# Load Model
model = joblib.load("model.pkl")
encoder = joblib.load("encoder.pkl")
explainer = shap.TreeExplainer(model)

# ================================
# PREDICTION PAGE
# ================================
if page == "Prediction":

    st.title("🤰 Maternal Health Risk Prediction System")
    st.markdown("AI-powered clinical decision support system")

    st.warning("⚠ This is NOT a medical diagnosis tool.")

    st.divider()

    st.header("Patient Information")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 10, 60, 25)
        systolic = st.number_input("Systolic BP", 50, 200, 120)
        diastolic = st.number_input("Diastolic BP", 30, 150, 80)
        bs = st.number_input("Blood Sugar", 0.0, 30.0, 7.0)
        bmi = st.number_input("BMI", 10.0, 50.0, 22.0)

    with col2:
        body_temp = st.number_input("Body Temperature", 90.0, 110.0, 98.0)
        prev_comp = st.number_input("Previous Complications", 0, 1, 0)
        pre_dm = st.number_input("Preexisting Diabetes", 0, 1, 0)
        gest_dm = st.number_input("Gestational Diabetes", 0, 1, 0)
        mental = st.number_input("Mental Health Issues", 0, 1, 0)
        heart_rate = st.number_input("Heart Rate", 30, 180, 80)

    st.divider()

    # =========================
    # PREDICT BUTTON
    # =========================
    if st.button("🔍 Predict Risk"):

        input_data = pd.DataFrame([{
            "Age": age,
            "Systolic_BP": systolic,
            "Diastolic": diastolic,
            "BS": bs,
            "Body_Temp": body_temp,
            "BMI": bmi,
            "Previous_Complications": prev_comp,
            "Preexisting_Diabetes": pre_dm,
            "Gestational_Diabetes": gest_dm,
            "Mental_Health": mental,
            "Heart_Rate": heart_rate
        }])

        pred = model.predict(input_data)
        proba = model.predict_proba(input_data)

        risk = encoder.inverse_transform(pred)[0]
        confidence = np.max(proba) * 100

        # Explainability
        reasons = []
        
        if prev_comp ==1:
            reasons.append("Previous complication exists")
        if bs > 8:
            reasons.append("High blood sugar")
        if bmi > 30:
            reasons.append("High BMI")
        if systolic > 140:
            reasons.append("High BP")
        if pre_dm == 1:
            reasons.append("Preexisting diabetes")
        if gest_dm == 1:
            reasons.append("Gestational diabetes")
        if mental == 1:
            reasons.append("Mental health issue")

        if not reasons:
            reasons = ["No major risk factors"]

        save_prediction(age, risk, confidence)

        # STORE IN SESSION
        st.session_state.prediction_done = True
        st.session_state.result_data = {
            "age": age,
            "risk": risk,
            "confidence": confidence,
            "reasons": reasons,
            "patient_data": {
                "Age": age,
                "Systolic BP": systolic,
                "Diastolic BP": diastolic,
                "Blood Sugar": bs,
                "BMI": bmi,
                "Body Temperature": body_temp,
                "Heart Rate": heart_rate
            }
        }

        st.success("Prediction Completed!")

    # =========================
    # SHOW RESULTS
    # =========================
    if st.session_state.prediction_done:

        data = st.session_state.result_data

        st.subheader("🧠 Risk Result")

        if data["risk"] == "High":
            st.error("🔴 HIGH RISK")
        elif data["risk"] == "Medium":
            st.warning("🟡 MEDIUM RISK")
        else:
            st.success("🟢 LOW RISK")

        st.subheader("📊 Confidence")
        st.progress(int(data["confidence"]))
        st.write(f"{data['confidence']:.2f}%")

        st.subheader("🧾 Why this prediction?")
        for r in data["reasons"]:
            st.write("•", r)

        st.subheader("📈 Feature Importance")

        feat_df = pd.DataFrame({
            "Feature": model.feature_names_in_,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=True)

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.barh(feat_df["Feature"], feat_df["Importance"])
        ax.set_title("Model Feature Importance")
        ax.set_xlabel("Importance Score")

        st.pyplot(fig)
import streamlit as st
import pandas as pd

def analytics_page():
    st.title("📊 Hospital Analytics Dashboard")

    history = pd.read_csv("prediction_history.csv")

    # =========================
    # KPI CALCULATIONS
    # =========================
    total_patients = len(history)
    high_risk = len(history[history["Risk_Level"] == "High"])
    low_risk = len(history[history["Risk_Level"] == "Low"])

    high_risk_percent = (high_risk / total_patients) * 100
    low_risk_percent = (low_risk / total_patients) * 100

    avg_confidence = history["Confidence"].mean()

    # =========================
    # KPI CARDS
    # =========================
    col1, col2, col3 = st.columns(3)

    col1.metric("👥 Total Patients", total_patients)
    col2.metric("🔴 High Risk %", f"{high_risk_percent:.1f}%")
    col3.metric("🟢 Low Risk %", f"{low_risk_percent:.1f}%")

    st.divider()
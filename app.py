import streamlit as st
import numpy as np
import joblib
import pandas as pd

st.set_page_config(page_title="Cardio Disease Prediction", layout="centered")

import os
import joblib
import streamlit as st

@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(BASE_DIR, "model")

    model_path = os.path.join(MODEL_DIR, "ensemble.pkl")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    selector_path = os.path.join(MODEL_DIR, "selector.pkl")

    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.stop()

    if not os.path.exists(scaler_path):
        st.error(f"Scaler file not found: {scaler_path}")
        st.stop()

    if not os.path.exists(selector_path):
        st.error(f"Selector file not found: {selector_path}")
        st.stop()

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    selector = joblib.load(selector_path)

    return model, scaler, selector


model, scaler, selector = load_model()

st.title("❤️ Prediksi Penyakit Jantung")
st.write("Masukkan data pasien untuk prediksi risiko penyakit jantung")

# Input
age = st.number_input("Age (days)", min_value=1)
height = st.number_input("Height (cm)", min_value=100)
weight = st.number_input("Weight (kg)", min_value=30)
ap_hi = st.number_input("Systolic BP", min_value=80)
ap_lo = st.number_input("Diastolic BP", min_value=50)
cholesterol = st.selectbox("Cholesterol", [1, 2, 3])
gluc = st.selectbox("Glucose", [1, 2, 3])
smoke = st.selectbox("Smoke", [0, 1])
alco = st.selectbox("Alcohol", [0, 1])
active = st.selectbox("Physical Activity", [0, 1])

if st.button("🔍 Predict"):
    BMI = weight / ((height/100)**2)
    pressure_diff = ap_hi - ap_lo

    data = np.array([[
        age, height, weight, ap_hi, ap_lo,
        cholesterol, gluc, smoke, alco, active,
        BMI, pressure_diff
    ]])

    data_scaled = scaler.transform(data)
    data_selected = selector.transform(data_scaled)

    proba = model.predict_proba(data_selected)[0][1]
    pred = model.predict(data_selected)[0]

    st.subheader("📊 Hasil Prediksi")
    st.write(f"Probabilitas Penyakit Jantung: **{proba:.2%}**")

    if pred == 1:
        st.error("⚠️ Berisiko Penyakit Jantung")
    else:
        st.success("✅ Risiko Rendah")


import streamlit as st
import joblib
import numpy as np

# load model
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")  # hapus jika tidak pakai

st.title("Prediksi Risiko Penyakit Jantung")

# input user
age = st.number_input("Umur", min_value=1, max_value=100)
chol = st.number_input("Kolesterol", min_value=100, max_value=400)

if st.button("Prediksi"):
    data = np.array([[age, chol]])
    data = scaler.transform(data)  # hapus jika tidak pakai scaler

    pred = model.predict(data)

    if pred[0] == 1:
        st.error("Berisiko Penyakit Jantung")
    else:
        st.success("Tidak Berisiko Penyakit Jantung")

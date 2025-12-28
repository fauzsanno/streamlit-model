import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectFromModel
from imblearn.combine import SMOTEENN
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Cardio Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
/* Background */
.stApp {
    background: linear-gradient(135deg, #f8fbff, #eef2f7);
    font-family: 'Segoe UI', sans-serif;
}

/* Main title */
h1 {
    text-align: center;
    color: #c0392b;
    font-weight: 700;
}

/* Section headers */
h3 {
    color: #2c3e50;
    margin-top: 30px;
}

/* Card container */
.card {
    background-color: white;
    padding: 25px;
    border-radius: 16px;
    box-shadow: 0px 8px 20px rgba(0,0,0,0.08);
    margin-bottom: 25px;
}

/* Button */
.stButton>button {
    width: 100%;
    border-radius: 12px;
    background: linear-gradient(90deg, #e74c3c, #c0392b);
    color: white;
    font-size: 16px;
    font-weight: bold;
    padding: 12px;
    border: none;
    transition: 0.3s ease;
}

.stButton>button:hover {
    background: linear-gradient(90deg, #c0392b, #96281b);
    transform: scale(1.02);
}

/* Success & error box */
.stAlert {
    border-radius: 12px;
}

/* Metric text */
.result-box {
    font-size: 20px;
    font-weight: 600;
    text-align: center;
    margin-top: 15px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================
st.title("❤️ Prediksi Penyakit Jantung")

# =========================
# TRAIN MODEL (CACHED)
# =========================
@st.cache_resource
def train_model():
    df = pd.read_csv("cardio_train.csv", sep=";")

    # Drop ID
    df = df.drop(columns=["id"])

    # Cleaning
    df = df[(df['ap_hi'] < 250) & (df['ap_lo'] < 200)]
    df = df[(df['height'] > 100) & (df['weight'] < 200)]

    # Feature engineering
    df['BMI'] = df['weight'] / ((df['height'] / 100) ** 2)
    df['pressure_diff'] = df['ap_hi'] - df['ap_lo']

    X = df.drop(columns=['cardio'])
    y = df['cardio']

    feature_names = X.columns.tolist()

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    smoteenn = SMOTEENN(random_state=42)
    X_res, y_res = smoteenn.fit_resample(X_scaled, y)

    selector = SelectFromModel(
        XGBClassifier(random_state=42, eval_metric="logloss")
    )
    X_selected = selector.fit_transform(X_res, y_res)

    X_train, _, y_train, _ = train_test_split(
        X_selected, y_res,
        test_size=0.2,
        random_state=42,
        stratify=y_res
    )

    xgb = XGBClassifier(
        learning_rate=0.05,
        max_depth=6,
        n_estimators=300,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )

    lgb = LGBMClassifier(
        learning_rate=0.05,
        num_leaves=31,
        n_estimators=300,
        random_state=42
    )

    model = VotingClassifier(
        estimators=[("xgb", xgb), ("lgb", lgb)],
        voting="soft"
    )

    model.fit(X_train, y_train)

    return model, scaler, selector, feature_names


with st.spinner("Training model (hanya 1x)..."):
    model, scaler, selector, feature_names = train_model()

st.success("Model siap digunakan ✅")

# =========================
# INPUT CARD
# =========================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📝 Input Data Pasien")

age = st.number_input("Age (days)", min_value=1)
gender = st.selectbox("Gender", [1, 2], format_func=lambda x: "Male" if x == 2 else "Female")
height = st.number_input("Height (cm)", min_value=100)
weight = st.number_input("Weight (kg)", min_value=30)
ap_hi = st.number_input("Systolic BP", min_value=80)
ap_lo = st.number_input("Diastolic BP", min_value=50)
cholesterol = st.selectbox("Cholesterol", [1, 2, 3])
gluc = st.selectbox("Glucose", [1, 2, 3])
smoke = st.selectbox("Smoke", [0, 1])
alco = st.selectbox("Alcohol", [0, 1])
active = st.selectbox("Physical Activity", [0, 1])

st.markdown('</div>', unsafe_allow_html=True)

# =========================
# PREDICTION
# =========================
if st.button("🔍 Predict"):
    BMI = weight / ((height / 100) ** 2)
    pressure_diff = ap_hi - ap_lo

    input_dict = {
        "age": age,
        "gender": gender,
        "height": height,
        "weight": weight,
        "ap_hi": ap_hi,
        "ap_lo": ap_lo,
        "cholesterol": cholesterol,
        "gluc": gluc,
        "smoke": smoke,
        "alco": alco,
        "active": active,
        "BMI": BMI,
        "pressure_diff": pressure_diff
    }

    input_df = pd.DataFrame([input_dict])
    input_df = input_df[feature_names]

    input_scaled = scaler.transform(input_df)
    input_selected = selector.transform(input_scaled)

    proba = model.predict_proba(input_selected)[0][1]
    pred = model.predict(input_selected)[0]

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Hasil Prediksi")
    st.markdown(
        f"<div class='result-box'>Probabilitas Penyakit Jantung: {proba:.2%}</div>",
        unsafe_allow_html=True
    )

    if pred == 1:
        st.error("⚠️ Berisiko Penyakit Jantung")
    else:
        st.success("✅ Risiko Rendah")

    st.markdown('</div>', unsafe_allow_html=True)

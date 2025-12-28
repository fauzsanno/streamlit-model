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

st.set_page_config(page_title="Cardio Disease Prediction", layout="centered")
st.title("❤️ Prediksi Penyakit Jantung")

# =========================
# TRAIN MODEL (CACHED)
# =========================
@st.cache_resource
def train_model():
    df = pd.read_csv("cardio_train.csv", sep=";")

     # ⛔ DROP ID DI SINI
    df = df.drop(columns=["id"])

    # Cleaning
    df = df[(df['ap_hi'] < 250) & (df['ap_lo'] < 200)]
    df = df[(df['height'] > 100) & (df['weight'] < 200)]

    # Feature engineering
    df['BMI'] = df['weight'] / ((df['height'] / 100) ** 2)
    df['pressure_diff'] = df['ap_hi'] - df['ap_lo']

    # =========================
    # FEATURES & TARGET
    # =========================
    X = df.drop(columns=['cardio'])
    y = df['cardio']

    # SIMPAN URUTAN FITUR
    feature_names = X.columns.tolist()

    # Scaling
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # Handle imbalance
    smoteenn = SMOTEENN(random_state=42)
    X_res, y_res = smoteenn.fit_resample(X_scaled, y)

    # Feature selection
    selector = SelectFromModel(
        XGBClassifier(
            random_state=42,
            eval_metric="logloss"
        )
    )
    X_selected = selector.fit_transform(X_res, y_res)

    # Train split
    X_train, _, y_train, _ = train_test_split(
        X_selected,
        y_res,
        test_size=0.2,
        random_state=42,
        stratify=y_res
    )

    # Models
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

    # Train ensemble
    model.fit(X_train, y_train)

    return model, scaler, selector, feature_names


with st.spinner("Training model (hanya 1x)..."):
    model, scaler, selector, feature_names = train_model()

st.success("Model siap digunakan ✅")

# =========================
# INPUT USER
# =========================
age = st.number_input("Age (days)", min_value=1)
gender = st.selectbox(
    "Gender",
    options=[1, 2],
    format_func=lambda x: "Male" if x == 2 else "Female"
)

height = st.number_input("Height (cm)", min_value=100)
weight = st.number_input("Weight (kg)", min_value=30)
ap_hi = st.number_input("Systolic BP", min_value=80)
ap_lo = st.number_input("Diastolic BP", min_value=50)
cholesterol = st.selectbox("Cholesterol", [1, 2, 3])
gluc = st.selectbox("Glucose", [1, 2, 3])
smoke = st.selectbox("Smoke", [0, 1])
alco = st.selectbox("Alcohol", [0, 1])
active = st.selectbox("Physical Activity", [0, 1])

# =========================
# PREDICTION
# =========================
if st.button("🔍 Predict"):
    BMI = weight / ((height / 100) ** 2)
    pressure_diff = ap_hi - ap_lo

    # ⬇️ INPUT HARUS DATAFRAME
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

    # ⬇️ SAMAKAN URUTAN FITUR
    input_df = input_df[feature_names]

    # Transform
    input_scaled = scaler.transform(input_df)
    input_selected = selector.transform(input_scaled)

    # Predict
    proba = model.predict_proba(input_selected)[0][1]
    pred = model.predict(input_selected)[0]

    st.subheader("📊 Hasil Prediksi")
    st.write(f"Probabilitas Penyakit Jantung: **{proba:.2%}**")

    if pred == 1:
        st.error("⚠️ Berisiko Penyakit Jantung")
    else:
        st.success("✅ Risiko Rendah")

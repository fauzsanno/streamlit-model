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
# PREMIUM CSS (LUXURY UI)
# =========================
st.markdown("""
<style>
/* ===== GLOBAL BACKGROUND ===== */
body {
    background-color: #757575;
}
.main {
    background-color: #757575;
}

/* ===== CARD ===== */
.card {
    background: #ffffff;
    padding: 2rem;
    border-radius: 20px;
    box-shadow: 0 12px 40px rgba(0,0,0,0.25);
    margin-bottom: 1.8rem;
}

/* ===== TITLE ===== */
.title {
    text-align: center;
    font-size: 2.3rem;
    font-weight: 800;
    color: #1f2937;
}
.subtitle {
    text-align: center;
    color: #6b7280;
    margin-bottom: 2rem;
}

/* ===== INPUT FIELDS ===== */
input, select, textarea {
    background-color: #bdbdbd !important;
    color: #212121 !important;
    border-radius: 12px !important;
    border: none !important;
    padding: 0.6rem !important;
}

/* Number input arrows */
input::-webkit-inner-spin-button {
    opacity: 1;
}

/* ===== STREAMLIT INPUT FIX ===== */
div[data-baseweb="input"] > div {
    background-color: #bdbdbd !important;
    border-radius: 12px !important;
}

div[data-baseweb="select"] > div {
    background-color: #bdbdbd !important;
    border-radius: 12px !important;
}

/* ===== BUTTON ===== */
.stButton>button {
    background: linear-gradient(135deg, #ef4444, #b91c1c);
    color: white;
    font-weight: 700;
    border-radius: 14px;
    height: 3.2rem;
    width: 100%;
    border: none;
    box-shadow: 0 8px 25px rgba(239,68,68,0.4);
}
.stButton>button:hover {
    background: linear-gradient(135deg, #b91c1c, #7f1d1d);
}

/* ===== RESULT BOX ===== */
.result-box {
    padding: 1.6rem;
    border-radius: 16px;
    margin-top: 1.5rem;
    font-size: 1.15rem;
}

/* ===== SUCCESS & DANGER ===== */
.success {
    background: #e8f5e9;
    color: #1b5e20;
    border-left: 6px solid #2e7d32;
}
.danger {
    background: #ffebee;
    color: #b71c1c;
    border-left: 6px solid #d32f2f;
}

/* ===== FOOTER ===== */
.footer {
    text-align: center;
    color: #e0e0e0;
    margin-top: 2.5rem;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="app-title">❤️ Cardio Disease Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Luxury Medical AI Decision Support System</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# =========================
# TRAIN MODEL (UNCHANGED)
# =========================
@st.cache_resource
def train_model():
    df = pd.read_csv("cardio_train.csv", sep=";")
    df = df.drop(columns=["id"])

    df = df[(df['ap_hi'] < 250) & (df['ap_lo'] < 200)]
    df = df[(df['height'] > 100) & (df['weight'] < 200)]

    df['BMI'] = df['weight'] / ((df['height'] / 100) ** 2)
    df['pressure_diff'] = df['ap_hi'] - df['ap_lo']

    X = df.drop(columns=['cardio'])
    y = df['cardio']

    feature_names = X.columns.tolist()

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    smoteenn = SMOTEENN(random_state=42)
    X_res, y_res = smoteenn.fit_resample(X_scaled, y)

    selector = SelectFromModel(XGBClassifier(random_state=42, eval_metric="logloss"))
    X_selected = selector.fit_transform(X_res, y_res)

    X_train, _, y_train, _ = train_test_split(
        X_selected, y_res, test_size=0.2, random_state=42, stratify=y_res
    )

    model = VotingClassifier(
        estimators=[
            ("xgb", XGBClassifier(
                learning_rate=0.05,
                max_depth=6,
                n_estimators=300,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42
            )),
            ("lgb", LGBMClassifier(
                learning_rate=0.05,
                num_leaves=31,
                n_estimators=300,
                random_state=42
            ))
        ],
        voting="soft"
    )

    model.fit(X_train, y_train)
    return model, scaler, selector, feature_names


with st.spinner("Initializing AI Model..."):
    model, scaler, selector, feature_names = train_model()

# =========================
# INPUT FORM
# =========================
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("🩺 Patient Medical Information")

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
if st.button("🔍 Predict Risk"):
    BMI = weight / ((height / 100) ** 2)
    pressure_diff = ap_hi - ap_lo

    input_df = pd.DataFrame([{
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
    }])[feature_names]

    input_scaled = scaler.transform(input_df)
    input_selected = selector.transform(input_scaled)

    proba = model.predict_proba(input_selected)[0][1]
    pred = model.predict(input_selected)[0]

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📊 Prediction Result")

    if pred == 1:
        st.markdown(
            f"<div class='result-danger'>⚠️ High Risk of Heart Disease<br>Probability: {proba:.2%}</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='result-success'>✅ Low Risk of Heart Disease<br>Probability: {proba:.2%}</div>",
            unsafe_allow_html=True
        )
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="footer">© 2025 Cardio AI • Premium Medical Decision System</div>', unsafe_allow_html=True)

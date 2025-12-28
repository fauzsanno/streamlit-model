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

/* BACKGROUND */
body {
    background: linear-gradient(135deg, #424242, #eef2ff);
}
.main {
    background: #424242;
}

/* GLASS CARD */
.glass-card {
    background: rgba(255, 255, 255, 0.85);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border-radius: 22px;
    padding: 2.2rem;
    margin-bottom: 2rem;
    box-shadow:
        0 20px 40px rgba(0,0,0,0.08),
        inset 0 1px 0 rgba(255,255,255,0.4);
}

/* HEADER */
.app-title {
    text-align: center;
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(90deg, #dc2626, #ef4444);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.app-subtitle {
    text-align: center;
    color: #6b7280;
    margin-top: 0.3rem;
    font-size: 1.05rem;
}

/* BUTTON */
.stButton > button {
    background: linear-gradient(135deg, #dc2626, #b91c1c);
    color: white;
    font-size: 1.1rem;
    font-weight: 700;
    border-radius: 14px;
    padding: 0.7rem;
    width: 100%;
    border: none;
    box-shadow: 0 10px 25px rgba(220,38,38,0.35);
    transition: all 0.3s ease;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 15px 30px rgba(220,38,38,0.45);
}

/* RESULT */
.result-success {
    background: linear-gradient(135deg, #ecfdf5, #d1fae5);
    color: #065f46;
    padding: 1.6rem;
    border-radius: 18px;
    font-size: 1.15rem;
    font-weight: 600;
    box-shadow: 0 12px 25px rgba(16,185,129,0.25);
}
.result-danger {
    background: linear-gradient(135deg, #fef2f2, #fee2e2);
    color: #7f1d1d;
    padding: 1.6rem;
    border-radius: 18px;
    font-size: 1.15rem;
    font-weight: 600;
    box-shadow: 0 12px 25px rgba(239,68,68,0.25);
}

/* FOOTER */
.footer {
    text-align: center;
    color: #9ca3af;
    font-size: 0.85rem;
    margin-top: 3rem;
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

age = st.number_input("Umur ( Jumlah hari)", min_value=1)
gender = st.selectbox("Jenis Kelamin", [1, 2], format_func=lambda x: "Male" if x == 2 else "Female")
height = st.number_input("Tinggi Badan (cm)", min_value=100)
weight = st.number_input("Berat Badan (kg)", min_value=30)
ap_hi = st.number_input(" Tekanan Darah(Systolic)", min_value=80)
ap_lo = st.number_input("Tekanan Dara (Diastolic)", min_value=50)
cholesterol = st.selectbox("Kolesterol", [1, 2, 3])
gluc = st.selectbox("Gula", [1, 2, 3])
smoke = st.selectbox("Merkok", [0, 1])
alco = st.selectbox("Alkohol", [0, 1])
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

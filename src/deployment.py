import streamlit as st
import pandas as pd
import joblib
import numpy as np

# =========================
# 🎯 PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Startup Success Predictor (Pre-Launch)",
    page_icon="🚀",
    layout="centered"
)

st.title("🚀 Startup Success Prediction App")
st.markdown("""
### Predict Startup Success *Before Launch*  
Use early-stage (pre-launch) signals to estimate your startup’s potential for success.  
Fill out the form below to get your prediction.
""")

# =========================
# 🧠 LOAD MODEL & SCALER
# =========================
@st.cache_resource
def load_model():
    model = joblib.load("models/startup_success_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
    return model, scaler, feature_names

model, scaler, feature_names = load_model()

# =========================
# 🧾 INPUT SECTIONS
# =========================
st.header("📍 Basic Startup Information")
name = st.text_input("Startup Name", "My Innovative Startup")
category_code = st.selectbox("Category", [
    "software", "web", "mobile", "enterprise",
    "advertising", "gamesvideo", "ecommerce",
    "biotech", "consulting", "other"
])
founded_at = st.number_input(
    "Years since founding", min_value=0, max_value=10, value=0,
    help="Enter 0 if the startup is pre-launch or not yet incorporated."
)

# =========================
# 💰 FUNDING INFORMATION
# =========================
st.header("💰 Early Funding Signals (Pre-Launch Stage)")
funding_rounds = st.number_input(
    "Number of Funding Rounds", min_value=0, max_value=2, value=0,
    help="Pre-launch startups usually have 0–1 funding rounds."
)
funding_total_usd = st.number_input(
    "Total Funding (USD)", min_value=0, value=1000000, step=1000,
    help="Total funding received so far (0 if none)."
)
has_vc = st.selectbox("Has Venture Capital?", [0, 1])
has_angel = st.selectbox("Has Angel Investors?", [0, 1])

# =========================
# 💼 TEAM & MARKET INFO
# =========================
st.header("💼 Team & Market Indicators")
relationships = st.number_input(
    "Team/Network Connections", min_value=0, max_value=50, value=5,
    help="Approximate number of industry or investor relationships."
)
milestones = st.number_input(
    "Milestones Achieved", min_value=0, value=10,
    help="Key product or partnership milestones reached."
)
avg_participants = st.slider(
    "Avg Participants per Funding Round", 0.0, 10.0, 2.0,
    help="Average number of investors participating per round."
)

# =========================
# ⚙️ DATA PREPARATION
# =========================
input_dict = {
    'funding_rounds': [funding_rounds],
    'funding_total_usd': [funding_total_usd],
    'relationships': [relationships],
    'milestones': [milestones],
    'avg_participants': [avg_participants],
    'has_vc': [has_vc],
    'has_angel': [has_angel],
    'is_software': [1 if category_code == "software" else 0],
    'is_web': [1 if category_code == "web" else 0],
    'is_mobile': [1 if category_code == "mobile" else 0],
    'is_enterprise': [1 if category_code == "enterprise" else 0],
    'is_advertising': [1 if category_code == "advertising" else 0],
    'is_gamesvideo': [1 if category_code == "gamesvideo" else 0],
    'is_ecommerce': [1 if category_code == "ecommerce" else 0],
    'is_biotech': [1 if category_code == "biotech" else 0],
    'is_consulting': [1 if category_code == "consulting" else 0],
    'is_othercategory': [1 if category_code == "other" else 0],
}

input_df = pd.DataFrame(input_dict)

# Align columns with training features
for col in feature_names:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[feature_names]

# =========================
# 🔮 PREDICTION
# =========================
if st.button("🚀 Predict Startup Success"):
    # Scale input features
    input_scaled = scaler.transform(input_df)

    # Make prediction
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    # Display results
    st.success(f"🎯 Predicted Success: {'✅ Successful' if pred == 1 else '❌ Unsuccessful'}")
    st.info(f"📈 Probability of Success: {prob*100:.2f}%")



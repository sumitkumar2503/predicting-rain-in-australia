import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ── Page config ─────────────────────────────────────────────
st.set_page_config(
    page_title="Rain Predictor",
    page_icon="🌧",
    layout="wide"
)

# ── Custom CSS (Cards) ─────────────────────────────────────
st.markdown("""
<style>
.card {
    background-color: #1e1e1e;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    height: 100%;
}
</style>
""", unsafe_allow_html=True)

# ── Load model ─────────────────────────────────────────────
@st.cache_resource
def load_bundle():
    return joblib.load("rain_model.pkl")

bundle = load_bundle()
pre_fit = bundle["preprocessor"]
model = bundle["model"]
threshold = bundle["threshold"]

# ── Features ───────────────────────────────────────────────
NUM_FEATS = [
    "MinTemp","MaxTemp","Rainfall","WindGustSpeed",
    "WindSpeed9am","WindSpeed3pm","Humidity9am","Humidity3pm",
    "Pressure9am","Pressure3pm","Cloud9am","Cloud3pm",
    "Temp9am","Temp3pm","RainToday","Month",
    "TempRange","PressureDrop","HumidityRise","CloudAvg","HumidityXCloud"
]
CAT_FEATS = ["WindGustDir","WindDir9am","WindDir3pm","Season"]
ALL_FEATS = NUM_FEATS + CAT_FEATS

WIND_DIRS = ["N","NNE","NE","ENE","E","ESE","SE","SSE",
             "S","SSW","SW","WSW","W","WNW","NW","NNW"]

# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.title("Settings")
    month = st.slider("Month",1,12,6)
    rain_today = st.radio("Rain today?",["No","Yes"])
    rain_today_enc = 1 if rain_today=="Yes" else 0

def get_season(m):
    if m in [12,1,2]: return "Summer"
    elif m in [3,4,5]: return "Autumn"
    elif m in [6,7,8]: return "Winter"
    else: return "Spring"

season = get_season(month)

# ── Title ──────────────────────────────────────────────────
st.title("🌧 Rain Prediction System")
st.divider()

# ── 2x2 GRID ───────────────────────────────────────────────
c1, c2 = st.columns(2)
c3, c4 = st.columns(2)

# Card 1
with c1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🌡 Temperature & Pressure")
    min_temp = st.slider("Min Temp", -10.0, 35.0, 12.0)
    max_temp = st.slider("Max Temp", 0.0, 50.0, 23.0)
    temp_9am = st.slider("Temp 9am", -10.0, 40.0, 16.0)
    temp_3pm = st.slider("Temp 3pm", -10.0, 45.0, 21.0)
    pressure_9am = st.slider("Pressure 9am", 980.0, 1040.0, 1017.0)
    pressure_3pm = st.slider("Pressure 3pm", 980.0, 1040.0, 1015.0)
    st.markdown('</div>', unsafe_allow_html=True)

# Card 2
with c2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("☁️ Cloud & Rain")
    cloud_9am = st.slider("Cloud 9am", 0, 8, 4)
    cloud_3pm = st.slider("Cloud 3pm", 0, 8, 4)
    rainfall = st.slider("Rainfall", 0.0, 100.0, 1.0)
    sunshine = st.slider("Sunshine", 0.0, 14.0, 7.0)
    st.markdown('</div>', unsafe_allow_html=True)

# Card 3
with c3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("💨 Wind")
    wind_gust_dir = st.selectbox("Gust Dir", WIND_DIRS)
    wind_dir_9am = st.selectbox("Dir 9am", WIND_DIRS)
    wind_dir_3pm = st.selectbox("Dir 3pm", WIND_DIRS)
    wind_gust_speed = st.slider("Gust Speed", 0, 130, 39)
    wind_speed_9am = st.slider("Speed 9am", 0, 80, 15)
    wind_speed_3pm = st.slider("Speed 3pm", 0, 80, 19)
    st.markdown('</div>', unsafe_allow_html=True)

# Card 4
with c4:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("💧 Humidity")
    humidity_9am = st.slider("Humidity 9am", 0, 100, 68)
    humidity_3pm = st.slider("Humidity 3pm", 0, 100, 52)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Engineered Features (CENTERED) ─────────────────────────
st.divider()
col1, col2, col3 = st.columns([1,2,1])

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("⚙️ Engineered Features")

    temp_range = max_temp - min_temp
    pressure_drop = pressure_9am - pressure_3pm
    humidity_rise = humidity_3pm - humidity_9am
    cloud_avg = (cloud_9am + cloud_3pm)/2
    humidity_xcloud = humidity_3pm * cloud_3pm

    st.metric("Temp Range", round(temp_range,2))
    st.metric("Pressure Drop", round(pressure_drop,2))
    st.metric("Humidity Rise", round(humidity_rise,2))
    st.metric("Cloud Avg", round(cloud_avg,2))
    st.metric("Humidity x Cloud", round(humidity_xcloud,2))

    st.markdown('</div>', unsafe_allow_html=True)

# ── Predict ────────────────────────────────────────────────
st.divider()

if st.button("🔮 Predict"):
    input_data = {
        "MinTemp": min_temp,
        "MaxTemp": max_temp,
        "Rainfall": rainfall,
        "WindGustSpeed": wind_gust_speed,
        "WindSpeed9am": wind_speed_9am,
        "WindSpeed3pm": wind_speed_3pm,
        "Humidity9am": humidity_9am,
        "Humidity3pm": humidity_3pm,
        "Pressure9am": pressure_9am,
        "Pressure3pm": pressure_3pm,
        "Cloud9am": float(cloud_9am),
        "Cloud3pm": float(cloud_3pm),
        "Temp9am": temp_9am,
        "Temp3pm": temp_3pm,
        "RainToday": float(rain_today_enc),
        "Month": float(month),
        "TempRange": temp_range,
        "PressureDrop": pressure_drop,
        "HumidityRise": humidity_rise,
        "CloudAvg": cloud_avg,
        "HumidityXCloud": humidity_xcloud,
        "WindGustDir": wind_gust_dir,
        "WindDir9am": wind_dir_9am,
        "WindDir3pm": wind_dir_3pm,
        "Season": season,
    }

    df = pd.DataFrame([input_data])[ALL_FEATS]
    X = pre_fit.transform(df)
    proba = model.predict_proba(X)[0][1]
    pred = int(proba >= threshold)

    if pred:
        st.error(f"🌧 Rain Expected ({proba:.1%})")
    else:
        st.success(f"☀️ No Rain ({proba:.1%})")

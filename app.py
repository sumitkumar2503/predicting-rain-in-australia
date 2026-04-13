import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Rain in Australia Predictor",
    page_icon="🌧",
    layout="wide"
)

# ── Load model bundle ─────────────────────────────────────────────────────────
# Bundle structure (from Cell 98 in notebook):
#   bundle['preprocessor'] → fitted ColumnTransformer (pre_fit)
#   bundle['model']        → CalibratedClassifierCV (hgb_calibrated)
#   bundle['threshold']    → float (best_thresh from PR curve)
@st.cache_resource
def load_bundle():
    return joblib.load("rain_model.pkl")

bundle     = load_bundle()
pre_fit    = bundle["preprocessor"]
model      = bundle["model"]
threshold  = bundle["threshold"]

# ── Exact feature lists from Cell 48 of the notebook ─────────────────────────
NUM_FEATS = [
    "MinTemp", "MaxTemp", "Rainfall", "WindGustSpeed",
    "WindSpeed9am", "WindSpeed3pm", "Humidity9am", "Humidity3pm",
    "Pressure9am", "Pressure3pm", "Cloud9am", "Cloud3pm",
    "Temp9am", "Temp3pm", "RainToday", "Month",
    "TempRange", "PressureDrop", "HumidityRise", "CloudAvg", "HumidityXCloud"
]
CAT_FEATS = ["WindGustDir", "WindDir9am", "WindDir3pm", "Season"]
ALL_FEATS  = NUM_FEATS + CAT_FEATS   # exact column order the preprocessor expects

WIND_DIRS = [
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"
]

LOCATIONS = [
    "Albury", "BadgerysCreek", "Cobar", "CoffsHarbour", "Moree",
    "Newcastle", "NorahHead", "NorfolkIsland", "Penrith", "Richmond",
    "Sydney", "SydneyAirport", "WaggaWagga", "Williamtown", "Wollongong",
    "Canberra", "Tuggeranong", "MountGinini", "Ballarat", "Bendigo",
    "Sale", "MelbourneAirport", "Melbourne", "Mildura", "Nhil",
    "Portland", "Watsonia", "Dartmoor", "Brisbane", "Cairns",
    "GoldCoast", "Townsville", "Adelaide", "MountGambier", "Nuriootpa",
    "Woomera", "Albany", "Witchcliffe", "PearceRAAF", "PerthAirport",
    "Perth", "SalmonGums", "Walpole", "Hobart", "Launceston",
    "AliceSprings", "Darwin", "Katherine", "Uluru"
]

MONTH_NAMES = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December"
}

def get_season(m):
    # From Cell 44 of the notebook (label_season function)
    if m in [12, 1, 2]:   return "Summer"
    elif m in [3, 4, 5]:  return "Autumn"
    elif m in [6, 7, 8]:  return "Winter"
    else:                  return "Spring"

# ── Sidebar — location, date context, model info ─────────────────────────────
with st.sidebar:
    st.title("🌏 Location & Date")
    st.divider()

    # Location is NOT a model feature (dropped before X is defined in notebook)
    # but shown for UX context
    location = st.selectbox("Weather Station", LOCATIONS, index=LOCATIONS.index("Sydney"))

    month = st.slider("Month", min_value=1, max_value=12, value=6)
    season = get_season(month)

    season_icons = {"Summer": "☀️", "Autumn": "🍂", "Winter": "❄️", "Spring": "🌸"}
    st.caption(f"{MONTH_NAMES[month]}  ·  {season_icons[season]} {season}")

    st.divider()
    rain_today_str = st.radio("Did it rain today?", ["No", "Yes"], horizontal=True)
    rain_today_enc = 1 if rain_today_str == "Yes" else 0

    st.divider()
    st.markdown("**Model info**")
    st.caption(
        f"Calibrated HistGradientBoostingClassifier  \n"
        f"ROC-AUC ≈ 0.887  ·  Recall ≈ 0.785  \n"
        f"Decision threshold: **{threshold:.3f}**  \n"
        f"(tuned on precision-recall curve)"
    )

# ── Main header ───────────────────────────────────────────────────────────────
st.title("🌧 Rain in Australia — Next-Day Predictor")
st.caption(
    f"**{location}**  ·  {MONTH_NAMES[month]} ({season})  ·  "
    f"Rain today: {rain_today_str}  ·  Threshold: {threshold:.3f}"
)
st.divider()

# ── Four feature-group cards (based on Section 3.2 of the notebook) ───────────
card_thermo, card_cloud = st.columns(2)
card_wind,   card_rain  = st.columns(2)

# ── Card 1 — Temperature & Pressure (Thermodynamic) ─────────────────────────
with card_thermo:
    st.markdown("#### 🌡 Temperature & Pressure")
    st.caption("Thermodynamic group — directly reflect air-mass properties.")
    min_temp     = st.slider("Min Temp (°C)",       -10.0, 35.0,  12.0, 0.5)
    max_temp     = st.slider("Max Temp (°C)",          0.0, 50.0,  23.0, 0.5)
    temp_9am     = st.slider("Temp at 9am (°C)",    -10.0, 40.0,  16.0, 0.5)
    temp_3pm     = st.slider("Temp at 3pm (°C)",    -10.0, 45.0,  21.0, 0.5)
    pressure_9am = st.slider("Pressure 9am (hPa)", 980.0, 1040.0, 1017.0, 0.5)
    pressure_3pm = st.slider("Pressure 3pm (hPa)", 980.0, 1040.0, 1015.0, 0.5)
    humidity_9am = st.slider("Humidity 9am (%)",       0, 100, 68)
    humidity_3pm = st.slider("Humidity 3pm (%)",       0, 100, 52)

# ── Card 2 — Cloud & Radiation ────────────────────────────────────────────────
with card_cloud:
    st.markdown("#### ☁️ Cloud & Radiation")
    st.caption("High missingness (38–48%) — median-imputed inside the pipeline.")
    cloud_9am   = st.slider("Cloud cover 9am (oktas)", 0, 8, 4)
    cloud_3pm   = st.slider("Cloud cover 3pm (oktas)", 0, 8, 4)
    evaporation = st.slider("Evaporation (mm)",        0.0, 30.0, 5.0, 0.5)
    sunshine    = st.slider("Sunshine (hrs)",           0.0, 14.0, 7.0, 0.5)

    st.markdown("##### 🌧 Rainfall")
    rainfall = st.slider("Rainfall today (mm)", 0.0, 100.0, 1.0, 0.5)

# ── Card 3 — Wind ─────────────────────────────────────────────────────────────
with card_wind:
    st.markdown("#### 💨 Wind")
    st.caption("Synoptic movement — frontal systems bring directional shifts before rain.")
    wind_gust_dir   = st.selectbox("Wind Gust Direction", WIND_DIRS, index=0)
    wind_dir_9am    = st.selectbox("Wind Direction 9am",  WIND_DIRS, index=0)
    wind_dir_3pm    = st.selectbox("Wind Direction 3pm",  WIND_DIRS, index=4)
    wind_gust_speed = st.slider("Wind Gust Speed (km/h)",  0, 130, 39)
    wind_speed_9am  = st.slider("Wind Speed 9am (km/h)",   0,  80, 15)
    wind_speed_3pm  = st.slider("Wind Speed 3pm (km/h)",   0,  80, 19)

# ── Card 4 — Engineered features (auto-computed, Section 3.3) ─────────────────
with card_rain:
    st.markdown("#### ⚙️ Engineered Features")
    st.caption("Computed from your inputs (Cell 44 of notebook) — passed directly to the model.")

    # From Cell 44 of the notebook
    temp_range     = round(max_temp - min_temp, 2)
    pressure_drop  = round(pressure_9am - pressure_3pm, 2)
    humidity_rise  = round(humidity_3pm - humidity_9am, 2)
    cloud_avg      = round((cloud_9am + cloud_3pm) / 2, 2)
    humidity_xcloud = round(humidity_3pm * cloud_3pm, 2)

    derived_df = pd.DataFrame({
        "Feature":        ["TempRange", "PressureDrop", "HumidityRise", "CloudAvg", "HumidityXCloud", "Month", "Season"],
        "Value":          [temp_range, pressure_drop, humidity_rise, cloud_avg, humidity_xcloud, month, season],
        "Signal":         [
            "Narrow → overcast" if temp_range < 8 else "Wide → stable/dry",
            "Falling → frontal" if pressure_drop > 2 else "Stable",
            "Rising → moisture loading" if humidity_rise > 5 else "Stable",
            f"{cloud_avg:.1f} oktas average",
            "Strong signal" if humidity_xcloud > 250 else "Weak signal",
            MONTH_NAMES[month],
            f"{season_icons[season]} {season}",
        ]
    })
    st.dataframe(derived_df, hide_index=True, use_container_width=True)

# ── Predict ───────────────────────────────────────────────────────────────────
st.divider()

_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    predict = st.button("🔮  Predict Rain Tomorrow", type="primary", use_container_width=True)

if predict:
    # Build DataFrame in the EXACT column order the preprocessor was fitted on
    # (NUM_FEATS then CAT_FEATS — Cell 48/49 of the notebook)
    input_data = {
        # numeric features (NUM_FEATS order)
        "MinTemp":         min_temp,
        "MaxTemp":         max_temp,
        "Rainfall":        rainfall,
        "WindGustSpeed":   wind_gust_speed,
        "WindSpeed9am":    wind_speed_9am,
        "WindSpeed3pm":    wind_speed_3pm,
        "Humidity9am":     humidity_9am,
        "Humidity3pm":     humidity_3pm,
        "Pressure9am":     pressure_9am,
        "Pressure3pm":     pressure_3pm,
        "Cloud9am":        float(cloud_9am),
        "Cloud3pm":        float(cloud_3pm),
        "Temp9am":         temp_9am,
        "Temp3pm":         temp_3pm,
        "RainToday":       float(rain_today_enc),
        "Month":           float(month),
        "TempRange":       temp_range,
        "PressureDrop":    pressure_drop,
        "HumidityRise":    humidity_rise,
        "CloudAvg":        cloud_avg,
        "HumidityXCloud":  humidity_xcloud,
        # categorical features (CAT_FEATS order)
        "WindGustDir":     wind_gust_dir,
        "WindDir9am":      wind_dir_9am,
        "WindDir3pm":      wind_dir_3pm,
        "Season":          season,
    }

    input_df = pd.DataFrame([input_data])[ALL_FEATS]   # enforce exact column order

    # Predict using the same two-step process as Cell 99:
    # 1. preprocessor.transform()  →  2. model.predict_proba()
    X_transformed = pre_fit.transform(input_df)
    proba         = model.predict_proba(X_transformed)[0][1]
    prediction    = int(proba >= threshold)

    # ── Result display ────────────────────────────────────────────────────────
    st.divider()
    left, right = st.columns(2)

    with left:
        if prediction == 1:
            st.error("### 🌧 Rain predicted tomorrow")
        else:
            st.success("### ☀️ No rain tomorrow")

        st.metric("Rain probability", f"{proba:.1%}")
        st.progress(float(proba))
        st.caption(f"Threshold used: {threshold:.3f}  ·  Predicted class: {'Rain' if prediction else 'No Rain'}")

    with right:
        # Probability interpretation guide
        if   proba < 0.30: interp, colour = "Low chance of rain",              "🟢"
        elif proba < 0.45: interp, colour = "Slight chance of rain",           "🟡"
        elif proba < 0.60: interp, colour = "Borderline / mixed signal",       "🟠"
        elif proba < 0.75: interp, colour = "Moderate chance of rain",         "🟠"
        else:              interp, colour = "Strong chance of rain",           "🔴"

        st.markdown(f"**Interpretation:** {colour} {interp}")
        st.markdown("---")
        st.markdown("**Probability guide**")
        guide = pd.DataFrame({
            "Range":   ["< 30%", "30–45%", "45–60%", "60–75%", "> 75%"],
            "Meaning": ["Low", "Slight", "Borderline", "Moderate", "Strong"]
        })
        st.dataframe(guide, hide_index=True, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "COM763 Advanced Machine Learning  ·  Rain in Australia Dataset  ·  "
    "Bureau of Meteorology  ·  Deployed on Streamlit Community Cloud"
)

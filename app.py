import streamlit as st
import pandas as pd
import joblib

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Australia Rain Predictor",
    page_icon="🌧",
    layout="centered"
)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("rain_model_pipeline_v2.pkl")

model = load_model()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("Australia Next-Day Rain Predictor")
st.markdown("Enter today's weather observations to predict whether it will rain tomorrow.")
st.divider()

# ── Input form ────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Temperature & Rainfall")
    min_temp      = st.slider("Min Temp (°C)",        -10.0, 35.0,  12.0, 0.5)
    max_temp      = st.slider("Max Temp (°C)",          0.0, 50.0,  23.0, 0.5)
    rainfall      = st.slider("Rainfall (mm)",          0.0, 100.0,  1.0, 0.5)
    temp_9am      = st.slider("Temp at 9am (°C)",     -10.0, 40.0,  16.0, 0.5)
    temp_3pm      = st.slider("Temp at 3pm (°C)",     -10.0, 45.0,  21.0, 0.5)

    st.subheader("Humidity & Cloud")
    humidity_9am  = st.slider("Humidity 9am (%)",       0, 100, 68)
    humidity_3pm  = st.slider("Humidity 3pm (%)",       0, 100, 52)
    cloud_9am     = st.slider("Cloud cover 9am (oktas)", 0, 8, 4)
    cloud_3pm     = st.slider("Cloud cover 3pm (oktas)", 0, 8, 4)

with col2:
    st.subheader("Wind")
    wind_gust_dir  = st.selectbox("Wind Gust Direction",
        ['N','NNE','NE','ENE','E','ESE','SE','SSE',
         'S','SSW','SW','WSW','W','WNW','NW','NNW'])
    wind_dir_9am   = st.selectbox("Wind Direction 9am",
        ['N','NNE','NE','ENE','E','ESE','SE','SSE',
         'S','SSW','SW','WSW','W','WNW','NW','NNW'])
    wind_dir_3pm   = st.selectbox("Wind Direction 3pm",
        ['N','NNE','NE','ENE','E','ESE','SE','SSE',
         'S','SSW','SW','WSW','W','WNW','NW','NNW'])
    wind_gust_speed = st.slider("Wind Gust Speed (km/h)",  0, 130, 39)
    wind_speed_9am  = st.slider("Wind Speed 9am (km/h)",   0,  80, 15)
    wind_speed_3pm  = st.slider("Wind Speed 3pm (km/h)",   0,  80, 19)

    st.subheader("Pressure & Other")
    pressure_9am  = st.slider("Pressure 9am (hPa)", 980.0, 1040.0, 1017.0, 0.5)
    pressure_3pm  = st.slider("Pressure 3pm (hPa)", 980.0, 1040.0, 1015.0, 0.5)
    rain_today    = st.selectbox("Did it rain today?", ['No', 'Yes'])
    month         = st.slider("Month", 1, 12, 6)
    season        = st.selectbox("Season", ['Summer', 'Autumn', 'Winter', 'Spring'])

st.divider()

# ── Derived features ──────────────────────────────────────────────────────────
temp_range    = max_temp - min_temp
pressure_drop = pressure_9am - pressure_3pm
humidity_rise = humidity_3pm - humidity_9am
rain_today_enc = 1 if rain_today == 'Yes' else 0

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("Predict Rain Tomorrow", type="primary", use_container_width=True):
    input_df = pd.DataFrame([{
        'MinTemp': min_temp, 'MaxTemp': max_temp, 'Rainfall': rainfall,
        'WindGustSpeed': wind_gust_speed, 'WindSpeed9am': wind_speed_9am,
        'WindSpeed3pm': wind_speed_3pm, 'Humidity9am': humidity_9am,
        'Humidity3pm': humidity_3pm, 'Pressure9am': pressure_9am,
        'Pressure3pm': pressure_3pm, 'Cloud9am': cloud_9am,
        'Cloud3pm': cloud_3pm, 'Temp9am': temp_9am, 'Temp3pm': temp_3pm,
        'RainToday': rain_today_enc, 'Month': month,
        'TempRange': temp_range, 'PressureDrop': pressure_drop,
        'HumidityRise': humidity_rise,
        'WindGustDir': wind_gust_dir, 'WindDir9am': wind_dir_9am,
        'WindDir3pm': wind_dir_3pm, 'Season': season
    }])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.divider()
    if prediction == 1:
        st.error(f"Rain predicted tomorrow")
        st.metric("Rain probability", f"{probability:.1%}")
    else:
        st.success(f"No rain predicted tomorrow")
        st.metric("Rain probability", f"{probability:.1%}")

    st.progress(float(probability))
    st.caption(f"Model confidence: {probability:.1%} chance of rain tomorrow")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("COM763 Advanced Machine Learning · Rain in Australia Dataset · Bureau of Meteorology")

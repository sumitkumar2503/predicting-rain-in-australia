import streamlit as st
import pandas as pd
import joblib

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Rain in Australia Predictor",
    page_icon="🌧",
    layout="wide"
)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("rain_model_pipeline.pkl")

model = load_model()

# ── Sidebar — Location & temporal context ────────────────────────────────────
with st.sidebar:
    st.title("🌏 Location & Date")
    st.markdown("Set the weather station location and observation month.")
    st.divider()

    location = st.selectbox("Weather Station", [
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
    ])

    month = st.slider("Month of observation", min_value=1, max_value=12, value=6,
                      format="%d")

    month_names = {1:"January",2:"February",3:"March",4:"April",5:"May",
                   6:"June",7:"July",8:"August",9:"September",10:"October",
                   11:"November",12:"December"}
    st.caption(f"Selected: **{month_names[month]}**")

    def get_season(m):
        if m in [12, 1, 2]: return "Summer"
        elif m in [3, 4, 5]: return "Autumn"
        elif m in [6, 7, 8]: return "Winter"
        else:                return "Spring"

    season = get_season(month)
    season_icons = {"Summer": "☀️", "Autumn": "🍂", "Winter": "❄️", "Spring": "🌸"}
    st.info(f"{season_icons[season]} Australian season: **{season}**")

    st.divider()
    rain_today = st.selectbox("Did it rain today?", ["No", "Yes"])
    st.caption("RainToday flag — affects the model directly.")

    st.divider()
    st.markdown("**About this model**")
    st.markdown(
        "Calibrated HistGradientBoostingClassifier trained on the "
        "Rain in Australia dataset (Bureau of Meteorology, 2007–2017). "
        "ROC-AUC ≈ 0.887 on held-out test data."
    )

# ── Main header ───────────────────────────────────────────────────────────────
st.title("🌧 Rain in Australia — Next-Day Prediction")
st.markdown(
    f"Entering observations for **{location}** · **{month_names[month]}** "
    f"({season})  \nFill in today's weather readings across the four feature "
    "groups below, then click **Predict**."
)
st.divider()

# ── Feature group cards ───────────────────────────────────────────────────────
# Based on Section 3.2 of the notebook: Thermodynamic / Cloud & Radiation /
# Wind / Rainfall & Rain flag

col_thermo, col_cloud = st.columns(2)
col_wind, col_rain = st.columns(2)

# ── Card 1: Temperature & Pressure (Thermodynamic) ───────────────────────────
with col_thermo:
    st.markdown("#### 🌡 Temperature & Pressure")
    st.markdown("*Thermodynamic features — reflect air mass properties.*")

    min_temp     = st.slider("Min Temp (°C)",       -10.0, 35.0,  12.0, 0.5)
    max_temp     = st.slider("Max Temp (°C)",          0.0, 50.0,  23.0, 0.5)
    temp_9am     = st.slider("Temp at 9am (°C)",    -10.0, 40.0,  16.0, 0.5)
    temp_3pm     = st.slider("Temp at 3pm (°C)",    -10.0, 45.0,  21.0, 0.5)
    pressure_9am = st.slider("Pressure 9am (hPa)", 980.0, 1040.0, 1017.0, 0.5)
    pressure_3pm = st.slider("Pressure 3pm (hPa)", 980.0, 1040.0, 1015.0, 0.5)
    humidity_9am = st.slider("Humidity 9am (%)",      0, 100, 68)
    humidity_3pm = st.slider("Humidity 3pm (%)",      0, 100, 52)

# ── Card 2: Cloud & Radiation ─────────────────────────────────────────────────
with col_cloud:
    st.markdown("#### ☁️ Cloud & Radiation")
    st.markdown("*High missingness (38–48%) — imputed inside pipeline.*")

    cloud_9am   = st.slider("Cloud cover 9am (oktas)", 0, 8, 4)
    cloud_3pm   = st.slider("Cloud cover 3pm (oktas)", 0, 8, 4)
    evaporation = st.slider("Evaporation (mm)",        0.0, 30.0, 5.0, 0.5)
    sunshine    = st.slider("Sunshine (hrs)",           0.0, 14.0, 7.0, 0.5)

    st.markdown("#### 🌧 Rainfall")
    st.markdown("*Today's rainfall measurement.*")
    rainfall    = st.slider("Rainfall (mm)",            0.0, 100.0, 1.0, 0.5)

# ── Card 3: Wind ──────────────────────────────────────────────────────────────
with col_wind:
    st.markdown("#### 💨 Wind")
    st.markdown("*Direction and speed — frontal systems bring pre-rain shifts.*")

    DIRS = ["N","NNE","NE","ENE","E","ESE","SE","SSE",
            "S","SSW","SW","WSW","W","WNW","NW","NNW"]

    wind_gust_dir   = st.selectbox("Wind Gust Direction", DIRS, index=0)
    wind_dir_9am    = st.selectbox("Wind Direction 9am",  DIRS, index=0)
    wind_dir_3pm    = st.selectbox("Wind Direction 3pm",  DIRS, index=4)
    wind_gust_speed = st.slider("Wind Gust Speed (km/h)",  0, 130, 39)
    wind_speed_9am  = st.slider("Wind Speed 9am (km/h)",   0,  80, 15)
    wind_speed_3pm  = st.slider("Wind Speed 3pm (km/h)",   0,  80, 19)

# ── Card 4: Engineered features preview ───────────────────────────────────────
with col_rain:
    st.markdown("#### ⚙️ Derived Features")
    st.markdown("*Auto-computed from your inputs — used directly by the model.*")

    temp_range    = round(max_temp - min_temp, 2)
    pressure_drop = round(pressure_9am - pressure_3pm, 2)
    humidity_rise = round(humidity_3pm - humidity_9am, 2)
    cloud_avg     = round((cloud_9am + cloud_3pm) / 2, 2)
    humidity_cloud = round(humidity_3pm * cloud_3pm, 2)
    rain_today_enc = 1 if rain_today == "Yes" else 0

    derived_df = pd.DataFrame({
        "Feature":     ["TempRange", "PressureDrop", "HumidityRise",
                        "CloudAvg", "HumidityXCloud", "Month", "Season"],
        "Value":       [temp_range, pressure_drop, humidity_rise,
                        cloud_avg, humidity_cloud, month, season],
        "Interpretation": [
            "Narrow → cloudy day" if temp_range < 8 else "Wide → stable/dry",
            "Rising → frontal" if pressure_drop > 2 else "Stable pressure",
            "Rising → moisture" if humidity_rise > 5 else "Stable humidity",
            f"{cloud_avg:.1f} oktas avg",
            "High interaction signal" if humidity_cloud > 250 else "Low signal",
            month_names[month],
            season
        ]
    })
    st.dataframe(derived_df, hide_index=True, use_container_width=True)

# ── Predict button ────────────────────────────────────────────────────────────
st.divider()
predict_col, _ = st.columns([1, 2])
with predict_col:
    predict_clicked = st.button(
        "🔮 Predict Rain Tomorrow",
        type="primary",
        use_container_width=True
    )

if predict_clicked:
    input_df = pd.DataFrame([{
        "MinTemp":        min_temp,
        "MaxTemp":        max_temp,
        "Rainfall":       rainfall,
        "Evaporation":    evaporation,
        "Sunshine":       sunshine,
        "WindGustSpeed":  wind_gust_speed,
        "WindSpeed9am":   wind_speed_9am,
        "WindSpeed3pm":   wind_speed_3pm,
        "Humidity9am":    humidity_9am,
        "Humidity3pm":    humidity_3pm,
        "Pressure9am":    pressure_9am,
        "Pressure3pm":    pressure_3pm,
        "Cloud9am":       cloud_9am,
        "Cloud3pm":       cloud_3pm,
        "Temp9am":        temp_9am,
        "Temp3pm":        temp_3pm,
        "RainToday":      rain_today_enc,
        "Month":          month,
        "TempRange":      temp_range,
        "PressureDrop":   pressure_drop,
        "HumidityRise":   humidity_rise,
        "WindGustDir":    wind_gust_dir,
        "WindDir9am":     wind_dir_9am,
        "WindDir3pm":     wind_dir_3pm,
        "Season":         season,
        # CloudAvg and HumidityXCloud — include if your pipeline expects them
        "CloudAvg":               cloud_avg,
        "HumidityCloudInteraction": humidity_cloud,
    }])

    prediction  = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.divider()
    res_left, res_right = st.columns([1, 1])

    with res_left:
        if prediction == 1:
            st.error("### 🌧 Rain predicted tomorrow")
        else:
            st.success("### ☀️ No rain predicted tomorrow")

        st.metric(
            label="Estimated rain probability",
            value=f"{probability:.1%}"
        )
        st.progress(float(probability))

    with res_right:
        st.markdown("**Probability interpretation**")
        interp_df = pd.DataFrame({
            "Range":   ["0–30%", "30–45%", "45–60%", "60–75%", "75–100%"],
            "Meaning": [
                "Low chance", "Slight chance",
                "Borderline / mixed signal",
                "Moderate chance", "Strong chance"
            ]
        })
        st.dataframe(interp_df, hide_index=True, use_container_width=True)

        if   probability < 0.30: interp = "Low chance of rain"
        elif probability < 0.45: interp = "Slight chance of rain"
        elif probability < 0.60: interp = "Borderline / mixed weather signal"
        elif probability < 0.75: interp = "Moderate chance of rain"
        else:                    interp = "Strong chance of rain"

        st.info(f"**Interpretation:** {interp}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "COM763 Advanced Machine Learning · Rain in Australia Dataset · "
    "Bureau of Meteorology · Deployed on Streamlit Community Cloud"
)

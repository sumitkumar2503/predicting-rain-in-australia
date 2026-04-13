import streamlit as st
import pandas as pd
import joblib

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Rain in Australia Predictor",
    page_icon="🌧",
    layout="wide"
)

# ── Minimal CSS — only structural things Streamlit can't do natively ──────────
st.markdown("""
<style>
/* Slim the default top padding */
.block-container { padding-top: 1.4rem; }

/* Card shell used around each input group */
.card {
    border: 1px solid rgba(128,128,128,0.18);
    border-radius: 14px;
    padding: 16px 18px 4px;
    margin-bottom: 4px;
}
.card-header {
    display: flex;
    align-items: center;
    gap: 9px;
    margin-bottom: 12px;
    padding-bottom: 10px;
    border-bottom: 1px solid rgba(128,128,128,0.12);
}
.card-icon {
    width: 30px; height: 30px; ######
    border-radius: 7px;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; flex-shrink: 0;
}
.card-title {
    font-size: 13px; font-weight: 700; #####
    letter-spacing: 0.07em; text-transform: uppercase; opacity: 0.5; margin: 0;
}
.icon-blue   { background: #E6F1FB; }
.icon-teal   { background: #E1F5EE; }
.icon-amber  { background: #FAEEDA; }
.icon-purple { background: #EEEDFE; }

/* Result panel border + layout */
.result-panel {
    border-radius: 14px;
    padding: 20px 24px;
    border: 1px solid rgba(128,128,128,0.18);
}
.result-verdict { font-size: 19px; font-weight: 700; letter-spacing: -0.02em; }
.result-interp  { font-size: 13px; opacity: 0.5; margin-top: 3px; }
.result-pct     { font-size: 36px; font-weight: 700; letter-spacing: -0.04em; }
.prob-bar-track { background: rgba(128,128,128,0.12); border-radius: 6px; height: 7px; overflow: hidden; margin: 16px 0 14px; }
.prob-bar-fill  { height: 100%; border-radius: 6px; }
.tag {
    display: inline-block; font-size: 11px; padding: 3px 9px;
    border-radius: 6px; background: rgba(128,128,128,0.09); opacity: 0.7; margin: 2px 3px 2px 0;
}
</style>
""", unsafe_allow_html=True)

# ── Load model bundle ─────────────────────────────────────────────────────────
@st.cache_resource
def load_bundle():
    return joblib.load("rain_model.pkl")

bundle    = load_bundle()
pre_fit   = bundle["preprocessor"]
model     = bundle["model"]
threshold = bundle["threshold"]

# ── Constants ─────────────────────────────────────────────────────────────────
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

LOCATIONS = [
    "Albury","BadgerysCreek","Cobar","CoffsHarbour","Moree","Newcastle",
    "NorahHead","NorfolkIsland","Penrith","Richmond","Sydney","SydneyAirport",
    "WaggaWagga","Williamtown","Wollongong","Canberra","Tuggeranong",
    "MountGinini","Ballarat","Bendigo","Sale","MelbourneAirport","Melbourne",
    "Mildura","Nhil","Portland","Watsonia","Dartmoor","Brisbane","Cairns",
    "GoldCoast","Townsville","Adelaide","MountGambier","Nuriootpa","Woomera",
    "Albany","Witchcliffe","PearceRAAF","PerthAirport","Perth","SalmonGums",
    "Walpole","Hobart","Launceston","AliceSprings","Darwin","Katherine","Uluru"
]

MONTH_NAMES = {
    1:"January",2:"February",3:"March",4:"April",5:"May",6:"June",
    7:"July",8:"August",9:"September",10:"October",11:"November",12:"December"
}
SEASON_ICONS = {"Summer":"☀️","Autumn":"🍂","Winter":"❄️","Spring":"🌸"}

def get_season(m):
    if m in [12,1,2]:  return "Summer"
    elif m in [3,4,5]: return "Autumn"
    elif m in [6,7,8]: return "Winter"
    else:              return "Spring"

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Location & Date") ####
    st.divider()
    location       = st.selectbox("Weather station", LOCATIONS,
                                  index=LOCATIONS.index("Sydney"))
    month          = st.slider("Month", 1, 12, 6)
    season         = get_season(month)
    st.caption(f"{MONTH_NAMES[month]}  ·  {SEASON_ICONS[season]} {season}")
    st.divider()
    rain_today_str = st.radio("Rain today?", ["No","Yes"], horizontal=True)
    rain_today_enc = 1 if rain_today_str == "Yes" else 0
    st.divider()
    st.markdown("**Model**")
    st.caption(
        f"Calibrated HistGradientBoosting  \n"
        f"ROC-AUC ≈ 0.887 · Recall ≈ 0.785  \n"
        f"Threshold: **{threshold:.3f}**"
    )

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown(f"## 🌧 Rain in Australia — Next-Day Predictor")
st.caption(
    f"**{location}** · {MONTH_NAMES[month]} · {SEASON_ICONS[season]} {season} · " ####
    f"Rain today: {rain_today_str}"
)
st.write("")

# ══════════════════════════════════════════════════════════════════════════════
# ROW 1 — Temperature | Humidity & Pressure
# ══════════════════════════════════════════════════════════════════════════════
col_a, col_b = st.columns(2, gap="medium")

with col_a:
    st.markdown("""
    <div class="card">
      <div class="card-header">
        <div class="card-icon icon-blue">🌡</div>
        <span class="card-title">Temperature</span>
      </div>
    </div>""", unsafe_allow_html=True)
    min_temp = st.slider("Min temp (°C)",      -10.0, 35.0,  12.0, 0.5)
    max_temp = st.slider("Max temp (°C)",        0.0, 50.0,  23.0, 0.5)
    temp_9am = st.slider("Temp at 9 am (°C)", -10.0, 40.0,  16.0, 0.5)
    temp_3pm = st.slider("Temp at 3 pm (°C)", -10.0, 45.0,  21.0, 0.5)

with col_b:
    st.markdown("""
    <div class="card">
      <div class="card-header">
        <div class="card-icon icon-teal">💧</div>
        <span class="card-title">Humidity &amp; Pressure</span>
      </div>
    </div>""", unsafe_allow_html=True)
    humidity_9am = st.slider("Humidity 9 am (%)",       0, 100, 68)
    humidity_3pm = st.slider("Humidity 3 pm (%)",       0, 100, 52)
    pressure_9am = st.slider("Pressure 9 am (hPa)", 980.0, 1040.0, 1017.0, 0.5)
    pressure_3pm = st.slider("Pressure 3 pm (hPa)", 980.0, 1040.0, 1015.0, 0.5)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 2 — Wind | Cloud & Rainfall
# ══════════════════════════════════════════════════════════════════════════════
col_c, col_d = st.columns(2, gap="medium")

with col_c:
    st.markdown("""
    <div class="card">
      <div class="card-header">
        <div class="card-icon icon-amber">💨</div>
        <span class="card-title">Wind</span>
      </div>
    </div>""", unsafe_allow_html=True)
    wind_gust_dir   = st.selectbox("Gust direction",      WIND_DIRS, index=0)
    wind_dir_9am    = st.selectbox("Wind direction 9 am", WIND_DIRS, index=0)
    wind_dir_3pm    = st.selectbox("Wind direction 3 pm", WIND_DIRS, index=4)
    wind_gust_speed = st.slider("Gust speed (km/h)",      0, 130, 39)
    wind_speed_9am  = st.slider("Wind speed 9 am (km/h)", 0,  80, 15)
    wind_speed_3pm  = st.slider("Wind speed 3 pm (km/h)", 0,  80, 19)

with col_d:
    st.markdown("""
    <div class="card">
      <div class="card-header">
        <div class="card-icon icon-purple">☁️</div>
        <span class="card-title">Cloud &amp; Rainfall</span>
      </div>
    </div>""", unsafe_allow_html=True)
    cloud_9am   = st.slider("Cloud cover 9 am (oktas)", 0, 8, 4)
    cloud_3pm   = st.slider("Cloud cover 3 pm (oktas)", 0, 8, 4)
    evaporation = st.slider("Evaporation (mm)",          0.0, 30.0, 5.0, 0.5)
    sunshine    = st.slider("Sunshine (hrs)",             0.0, 14.0, 7.0, 0.5)
    rainfall    = st.slider("Rainfall today (mm)",        0.0, 100.0, 1.0, 0.5)

# ── Compute engineered features (always, so they're ready for the table) ─────
temp_range      = round(max_temp - min_temp, 1)
pressure_drop   = round(pressure_9am - pressure_3pm, 1)
humidity_rise   = round(humidity_3pm - humidity_9am, 1)
cloud_avg       = round((cloud_9am + cloud_3pm) / 2, 1)
humidity_xcloud = round(humidity_3pm * cloud_3pm, 1)

pd_sign = "+" if pressure_drop >= 0 else ""
hr_sign = "+" if humidity_rise >= 0 else ""

# ── Predict button ────────────────────────────────────────────────────────────
st.write("")
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    predict = st.button("🔮  Predict Rain Tomorrow", type="primary", use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# RESULT — shown only after clicking Predict
# Left col: engineered features table  |  Right col: prediction result
# ══════════════════════════════════════════════════════════════════════════════
if predict:
    # ── Run prediction ────────────────────────────────────────────────────────
    input_data = {
        "MinTemp": min_temp, "MaxTemp": max_temp, "Rainfall": rainfall,
        "WindGustSpeed": wind_gust_speed, "WindSpeed9am": wind_speed_9am,
        "WindSpeed3pm": wind_speed_3pm, "Humidity9am": humidity_9am,
        "Humidity3pm": humidity_3pm, "Pressure9am": pressure_9am,
        "Pressure3pm": pressure_3pm, "Cloud9am": float(cloud_9am),
        "Cloud3pm": float(cloud_3pm), "Temp9am": temp_9am, "Temp3pm": temp_3pm,
        "RainToday": float(rain_today_enc), "Month": float(month),
        "TempRange": temp_range, "PressureDrop": pressure_drop,
        "HumidityRise": humidity_rise, "CloudAvg": cloud_avg,
        "HumidityXCloud": humidity_xcloud,
        "WindGustDir": wind_gust_dir, "WindDir9am": wind_dir_9am,
        "WindDir3pm": wind_dir_3pm, "Season": season,
    }

    input_df      = pd.DataFrame([input_data])[ALL_FEATS]
    X_transformed = pre_fit.transform(input_df)
    proba         = model.predict_proba(X_transformed)[0][1]
    prediction    = int(proba >= threshold)

    pct          = round(proba * 100, 1)
    accent_color = "#185FA5" if prediction else "#0F6E56"
    verdict      = "Rain predicted tomorrow" if prediction else "No rain tomorrow"
    verdict_icon = "🌧" if prediction else "☀️"

    if   proba < 0.30: interp = "Low chance of rain"
    elif proba < 0.45: interp = "Slight chance of rain"
    elif proba < 0.60: interp = "Borderline — mixed signal"
    elif proba < 0.75: interp = "Moderate chance of rain"
    else:              interp = "Strong chance of rain"

    st.write("")
    left_col, right_col = st.columns(2, gap="large")

    # ── Left: engineered features as a native Streamlit table ────────────────
    with left_col:
        st.markdown("##### 🔬 Engineered Features")
        st.caption("Derived from your inputs — passed directly to the model")

        eng_df = pd.DataFrame({
            "Feature":     ["Temp Range", "Pressure Drop", "Humidity Rise", "Cloud Avg", "Humidity × Cloud", "Season"],
            "Value":       [
                f"{temp_range} °C",
                f"{pd_sign}{pressure_drop} hPa",
                f"{hr_sign}{humidity_rise} %",
                f"{cloud_avg} oktas",
                f"{humidity_xcloud}",
                f"{SEASON_ICONS[season]} {season}",
            ],
            "Signal":      [
                "Wide → stable air" if temp_range >= 10 else "Narrow → overcast",
                "Falling → frontal risk" if pressure_drop >= 2 else "Stable",
                "Rising → moisture loading" if humidity_rise >= 5 else "Stable",
                "Heavy cover" if cloud_avg >= 5 else "Partial cover",
                "Strong combined signal" if humidity_xcloud >= 250 else "Weak combined signal",
                MONTH_NAMES[month],
            ],
        })
        st.dataframe(eng_df, use_container_width=True, hide_index=True)

    # ── Right: prediction result panel ───────────────────────────────────────
    with right_col:
        st.markdown("##### 📊 Prediction Result")
        st.caption(f"Station: {location} · {MONTH_NAMES[month]} · Rain today: {rain_today_str}")

        st.markdown(f"""
<div class="result-panel" style="border-color: {accent_color}55;">
  <div style="display:flex; align-items:flex-start; justify-content:space-between; flex-wrap:wrap; gap:10px;">
    <div>
      <div class="result-verdict">{verdict_icon} {verdict}</div>
      <div class="result-interp">{interp}</div>
    </div>
    <div class="result-pct" style="color:{accent_color};">{pct}%</div>
  </div>

  <div class="prob-bar-track">
    <div class="prob-bar-fill" style="background:{accent_color}; width:{pct}%;"></div>
  </div>

  <div>
    <span class="tag">Threshold {threshold:.3f}</span>
    <span class="tag">{SEASON_ICONS[season]} {season}</span>
    <span class="tag">Rain today: {rain_today_str}</span>
  </div>
</div>
""", unsafe_allow_html=True)

        # Native metric for at-a-glance probability
        st.metric(label="Rain probability", value=f"{pct}%",
                  delta=f"{pct - threshold*100:+.1f}% vs threshold",
                  delta_color="inverse")

# ── Footer ────────────────────────────────────────────────────────────────────
st.write("")
st.divider()
st.caption(
    "Rain in Australia Dataset · Bureau of Meteorology · Streamlit Community Cloud"
)

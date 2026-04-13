import streamlit as st
import pandas as pd
import joblib

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Rain in Australia Predictor",
    page_icon="🌧",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Card shell ── */
.card {
    border: 1px solid rgba(128,128,128,0.2);
    border-radius: 14px;
    padding: 18px 20px 8px;
    margin-bottom: 16px;
}

/* ── Card header ── */
.card-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 14px;
    padding-bottom: 12px;
    border-bottom: 1px solid rgba(128,128,128,0.13);
}

.card-icon {
    width: 30px;
    height: 30px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 15px;
    flex-shrink: 0;
}

.card-title {
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    opacity: 0.55;
    margin: 0;
}

.icon-blue   { background: #E6F1FB; }
.icon-teal   { background: #E1F5EE; }
.icon-amber  { background: #FAEEDA; }
.icon-purple { background: #EEEDFE; }

/* ── Insight panel ── */
.insight-wrap {
    display: flex;
    justify-content: center;
    margin: 8px 0 4px;
}

.insight-panel {
    width: 100%;
    max-width: 800px;
    border: 1px solid rgba(128,128,128,0.18);
    border-radius: 14px;
    padding: 20px 24px 22px;
}

.insight-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    opacity: 0.45;
    margin-bottom: 14px;
}

.chips {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
}

.chip {
    border-radius: 10px;
    padding: 12px 14px;
}

.chip-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    opacity: 0.5;
    margin-bottom: 4px;
}

.chip-val {
    font-size: 18px;
    font-weight: 700;
    letter-spacing: -0.02em;
}

.chip-hint {
    font-size: 11px;
    opacity: 0.55;
    margin-top: 2px;
}

.c-blue   { background:#E6F1FB; }  .c-blue   .chip-val { color:#0C447C; }
.c-teal   { background:#E1F5EE; }  .c-teal   .chip-val { color:#085041; }
.c-amber  { background:#FAEEDA; }  .c-amber  .chip-val { color:#633806; }
.c-purple { background:#EEEDFE; }  .c-purple .chip-val { color:#3C3489; }
.c-pink   { background:#FBEAF0; }  .c-pink   .chip-val { color:#72243E; }
.c-gray   { background:#F1EFE8; }  .c-gray   .chip-val { color:#444441; }

/* ── Result panel ── */
.result-wrap {
    display: flex;
    justify-content: center;
    margin-top: 10px;
}

.result-panel {
    width: 100%;
    max-width: 680px;
    border-radius: 14px;
    padding: 22px 28px;
    border: 1px solid rgba(128,128,128,0.18);
}

.result-verdict {
    font-size: 20px;
    font-weight: 700;
    letter-spacing: -0.02em;
}

.result-interp {
    font-size: 13px;
    opacity: 0.5;
    margin-top: 4px;
}

.result-pct {
    font-size: 38px;
    font-weight: 700;
    letter-spacing: -0.04em;
}

.prob-bar-track {
    background: rgba(128,128,128,0.12);
    border-radius: 6px;
    height: 7px;
    overflow: hidden;
    margin: 18px 0 16px;
}

.prob-bar-fill {
    height: 100%;
    border-radius: 6px;
}

.tag {
    display: inline-block;
    font-size: 12px;
    padding: 4px 10px;
    border-radius: 6px;
    background: rgba(128,128,128,0.09);
    opacity: 0.75;
    margin: 2px 4px 2px 0;
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

# ── Constants (exact feature lists from Cell 48 of notebook) ──────────────────
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
    st.markdown("### Context")
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
st.markdown(f"## 🌧 Rain Predictor — {location}")
st.caption(
    f"{MONTH_NAMES[month]} · {SEASON_ICONS[season]} {season} · "
    f"Rain today: {rain_today_str} · Threshold: {threshold:.3f}"
)
st.write("")

# ══════════════════════════════════════════════════════════════════════════════
# ROW 1 — Cards A & B
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
# ROW 2 — Cards C & D
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

# ══════════════════════════════════════════════════════════════════════════════
# ENGINEERED FEATURES — centred compact insight panel
# ══════════════════════════════════════════════════════════════════════════════
temp_range      = round(max_temp - min_temp, 1)
pressure_drop   = round(pressure_9am - pressure_3pm, 1)
humidity_rise   = round(humidity_3pm - humidity_9am, 1)
cloud_avg       = round((cloud_9am + cloud_3pm) / 2, 1)
humidity_xcloud = round(humidity_3pm * cloud_3pm, 1)

pd_sign   = "+" if pressure_drop  >= 0 else ""
hr_sign   = "+" if humidity_rise  >= 0 else ""

st.write("")

# Centre via 3-column trick: empty | panel | empty
_, centre_col, _ = st.columns([0.5, 9, 0.5])
with centre_col:
    st.markdown(f"""
<div class="insight-panel">
  <p class="insight-label">Engineered features — derived from your inputs</p>
  <div class="chips">

    <div class="chip c-blue">
      <div class="chip-label">Temp range</div>
      <div class="chip-val">{temp_range} °C</div>
      <div class="chip-hint">{"Wide — stable dry air" if temp_range >= 10 else "Narrow — overcast day"}</div>
    </div>

    <div class="chip c-teal">
      <div class="chip-label">Pressure drop</div>
      <div class="chip-val">{pd_sign}{pressure_drop} hPa</div>
      <div class="chip-hint">{"Falling — frontal risk" if pressure_drop >= 2 else "Stable pressure"}</div>
    </div>

    <div class="chip c-purple">
      <div class="chip-label">Humidity rise</div>
      <div class="chip-val">{hr_sign}{humidity_rise} %</div>
      <div class="chip-hint">{"Rising — moisture loading" if humidity_rise >= 5 else "Stable humidity"}</div>
    </div>

    <div class="chip c-amber">
      <div class="chip-label">Cloud avg</div>
      <div class="chip-val">{cloud_avg} oktas</div>
      <div class="chip-hint">{"Heavy cover" if cloud_avg >= 5 else "Partial cover"}</div>
    </div>

    <div class="chip c-pink">
      <div class="chip-label">Humidity × cloud</div>
      <div class="chip-val">{humidity_xcloud}</div>
      <div class="chip-hint">{"Strong combined signal" if humidity_xcloud >= 250 else "Weak combined signal"}</div>
    </div>

    <div class="chip c-gray">
      <div class="chip-label">Season</div>
      <div class="chip-val">{season}</div>
      <div class="chip-hint">{MONTH_NAMES[month]}</div>
    </div>

  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PREDICT BUTTON
# ══════════════════════════════════════════════════════════════════════════════
st.write("")
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    predict = st.button("Predict Rain Tomorrow", type="primary", use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# RESULT — centred panel
# ══════════════════════════════════════════════════════════════════════════════
if predict:
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
    elif proba < 0.60: interp = "Borderline / mixed signal"
    elif proba < 0.75: interp = "Moderate chance of rain"
    else:              interp = "Strong chance of rain"

    _, res_col, _ = st.columns([0.5, 9, 0.5])
    with res_col:
        st.markdown(f"""
<div class="result-panel" style="border-color: {accent_color}44;">
  <div style="display:flex; align-items:flex-start; justify-content:space-between; flex-wrap:wrap; gap:12px;">
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
    <span class="tag">Threshold: {threshold:.3f}</span>
    <span class="tag">{SEASON_ICONS[season]} {season} · {MONTH_NAMES[month]}</span>
    <span class="tag">Rain today: {rain_today_str}</span>
    <span class="tag">Station: {location}</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.write("")
st.divider()
st.caption(
    "COM763 Advanced Machine Learning  ·  Rain in Australia Dataset  ·  "
    "Bureau of Meteorology  ·  Streamlit Community Cloud"
)

import streamlit as st
import pickle
import numpy as np

# --- Load Model and Scaler ---
try:
    ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
    standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))
except FileNotFoundError:
    st.error("❌ Error: Model or scaler files not found.")
    st.stop()

# --- Page Configuration ---
st.set_page_config(page_title="🔥 Forest Fire Prediction", layout="centered", page_icon="🔥")

# --- App Title & Description ---
st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>🔥 Forest Fire Prediction App</h1>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; font-size: 18px; color: #333; padding: 0 20px;'>
    🌲 Predict the <b>Forest Fire Danger Code (DC)</b> based on environmental conditions.  
    Fill in the values below and click <b>Predict</b> to get results!
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("## 📥 Input Parameters")

# --- Input Fields ---
col1, col2 = st.columns(2)

with col1:
    temperature = st.number_input("🌡️ Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0, step=0.1)
    ws = st.number_input("💨 Wind Speed (km/h)", min_value=0.0, max_value=50.0, value=15.0, step=0.1)
    ffmc = st.number_input("🔥 FFMC (Fine Fuel Moisture Code)", min_value=0.0, max_value=100.0, value=85.0, step=0.1)
    isi = st.number_input("🚀 ISI (Initial Spread Index)", min_value=0.0, max_value=50.0, value=5.0, step=0.1)

with col2:
    rh = st.number_input("💧 Relative Humidity (%)", min_value=0.0, max_value=100.0, value=60.0, step=0.1)
    rain = st.number_input("🌧️ Rain (mm)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    dmc = st.number_input("🌿 DMC (Duff Moisture Code)", min_value=0.0, max_value=200.0, value=10.0, step=0.1)

# --- Select Inputs ---
st.markdown("### 🔍 Additional Info")
col3, col4 = st.columns(2)

classes_mapping = {"No Fire 🔵": 0, "Fire 🔴": 1}
selected_class = col3.selectbox("🔥 Fire Class (Historical)", list(classes_mapping.keys()))
classes = classes_mapping[selected_class]

region_mapping = {"🌍 Region A": 0, "🌎 Region B": 1}
selected_region = col4.selectbox("📍 Region", list(region_mapping.keys()))
region = region_mapping[selected_region]

# --- Prediction ---
st.markdown("### 📊 Prediction")
if st.button("🚨 Predict Forest Fire Danger Code (DC)"):
    try:
        data = np.array([[temperature, rh, ws, rain, ffmc, dmc, isi, classes, region]])
        scaled_data = standard_scaler.transform(data)
        prediction_result = ridge_model.predict(scaled_data)
        st.success(f"✅ **Predicted DC:** {round(prediction_result[0], 2)}")
        st.info("ℹ️ DC indicates deep organic layer dryness — higher = more risk.")
    except Exception as e:
        st.error(f"⚠️ Error: {e}")
        st.warning("Please ensure correct inputs and model availability.")

# --- Footer ---
st.markdown("---")
st.markdown("<div style='text-align: center;'>Made with ❤️ by <b>Hariraj</b></div>", unsafe_allow_html=True)

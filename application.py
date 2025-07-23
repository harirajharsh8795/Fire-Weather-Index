import streamlit as st
import pickle
import numpy as np

# --- Load Model and Scaler ---
# Make sure 'models/ridge.pkl' and 'models/scaler.pkl' are in the
# same directory as your Streamlit script, or provide the correct path.
try:
    ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
    standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))
except FileNotFoundError:
    st.error("Error: Model or scaler files not found. Please ensure 'models/ridge.pkl' and 'models/scaler.pkl' exist in the correct directory.")
    st.stop() # Stop the app if files are not found

# --- Streamlit App Layout ---
st.set_page_config(page_title="Forest Fire Prediction", layout="centered")

st.title("🔥 Forest Fire Prediction App")
st.markdown("""
    Enter the environmental parameters below to predict the Forest Fire Danger Index (DC).
    This model predicts the Danger Code (DC) based on various weather and forest conditions.
""")

st.header("Input Parameters")

# --- Input Fields ---
# Using st.number_input for numerical inputs.
# You can add min_value, max_value, and step for better user experience.
temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0, step=0.1)
rh = st.number_input("Relative Humidity (%)", min_value=0.0, max_value=100.0, value=60.0, step=0.1)
ws = st.number_input("Wind Speed (km/h)", min_value=0.0, max_value=50.0, value=15.0, step=0.1)
rain = st.number_input("Rain (mm)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
ffmc = st.number_input("Fine Fuel Moisture Code (FFMC)", min_value=0.0, max_value=100.0, value=85.0, step=0.1)
dmc = st.number_input("Duff Moisture Code (DMC)", min_value=0.0, max_value=200.0, value=10.0, step=0.1)
isi = st.number_input("Initial Spread Index (ISI)", min_value=0.0, max_value=50.0, value=5.0, step=0.1)

# For 'Classes' and 'Region', assuming they are categorical and encoded as 0 or 1
classes_mapping = {"No Fire": 0, "Fire": 1}
selected_class = st.selectbox("Fire Class (Historical)", list(classes_mapping.keys()))
classes = classes_mapping[selected_class]

region_mapping = {"Region A": 0, "Region B": 1} # Adjust these based on your actual regions
selected_region = st.selectbox("Region", list(region_mapping.keys()))
region = region_mapping[selected_region]

# --- Prediction Button ---
if st.button("Predict Forest Fire Danger Code (DC)"):
    try:
        # Prepare the input data for scaling and prediction
        # Ensure the order of features matches the training data of your model
        data = np.array([[temperature, rh, ws, rain, ffmc, dmc, isi, classes, region]])

        # Scale the input data using the loaded scaler
        scaled_data = standard_scaler.transform(data)

        # Make prediction using the loaded Ridge model
        prediction_result = ridge_model.predict(scaled_data)

        # Display the result
        st.success(f"Predicted Forest Fire Danger Code (DC): **{round(prediction_result[0], 2)}**")
        st.info("The Danger Code (DC) is an indicator of the drying of deep organic layers.")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.warning("Please check your input values and ensure the model and scaler are loaded correctly.")

st.markdown("---")
st.markdown("Developed with ❤️ using Streamlit")

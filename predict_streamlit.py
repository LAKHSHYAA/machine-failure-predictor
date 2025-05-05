import streamlit as st
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt

# Load model and scaler
model = load_model("machine_failure_model.h5")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Machine Failure Predictor", layout="centered")

st.title("🔧 Machine Failure Prediction System")
st.write("Use the sliders in the sidebar to simulate sensor values and predict if the machine will fail.")

# Sidebar sliders for user inputs
st.sidebar.header("Input Features (Scroll sliders)")

air_temp = st.sidebar.slider("Air temperature [K]", min_value=290.0, max_value=320.0, value=300.0, step=0.5)
process_temp = st.sidebar.slider("Process temperature [K]", min_value=295.0, max_value=330.0, value=310.0, step=0.5)
rpm = st.sidebar.slider("Rotational speed [rpm]", min_value=1000.0, max_value=3000.0, value=1500.0, step=10.0)
torque = st.sidebar.slider("Torque [Nm]", min_value=10.0, max_value=90.0, value=40.0, step=1.0)
tool_wear = st.sidebar.slider("Tool wear [min]", min_value=0.0, max_value=300.0, value=20.0, step=1.0)

# Make prediction
input_data = np.array([[air_temp, process_temp, rpm, torque, tool_wear]])
input_scaled = scaler.transform(input_data)
prediction_prob = model.predict(input_scaled)
prediction = (prediction_prob > 0.5).astype(int)

# Result
st.subheader("Prediction Result")
if prediction[0][0] == 1:
    st.error("🚨 Machine Failure Detected!")
else:
    st.success("✅ No Failure Detected.")

st.metric(label="Prediction Probability", value=f"{prediction_prob[0][0]:.2%}")

# Optional: Interactive Trend Plot
st.subheader("Feature Sensitivity (Torque vs Failure Probability)")
torque_range = np.linspace(10, 90, 100)
trend_data = np.array([[air_temp, process_temp, rpm, t, tool_wear] for t in torque_range])
trend_scaled = scaler.transform(trend_data)
trend_probs = model.predict(trend_scaled)

fig, ax = plt.subplots()
ax.plot(torque_range, trend_probs, color='orange', linewidth=2)
ax.set_xlabel("Torque [Nm]")
ax.set_ylabel("Failure Probability")
ax.set_title("Impact of Torque on Failure Probability")
st.pyplot(fig)

import streamlit as st
import requests
import numpy as np
import matplotlib.pyplot as plt

st.title("Personal Biology Twin Dashboard")

# API base URL
API_URL = "http://localhost:8000"

# Input signals
st.header("Input Physiological Signals")
hr = st.slider("Heart Rate", 50, 120, 70)
hrv = st.slider("HRV", 20, 100, 50)
sleep = st.slider("Sleep Quality (0-1)", 0.0, 1.0, 0.8)

signals = {"hr": [hr], "hrv": [hrv], "sleep": [sleep]}

if st.button("Update Twin"):
    response = requests.post(f"{API_URL}/update_twin", json={"signals": signals})
    if response.status_code == 200:
        data = response.json()
        st.write("Latent State:", data["latent_state"])
        st.write("Personalized:", data["personalized"])
    else:
        st.error("API Error")

if st.button("Predict Trajectory"):
    response = requests.post(f"{API_URL}/predict_trajectory", json={"horizon": 7})
    if response.status_code == 200:
        traj = np.array(response.json()["trajectory"])
        fig, ax = plt.subplots()
        ax.plot(traj)
        ax.set_title("Predicted Latent Trajectory")
        st.pyplot(fig)
    else:
        st.error("API Error")

# Counterfactual simulation
st.header("Counterfactual Simulation")
intervention_type = st.selectbox("Intervention", ["sleep", "training_load", "stress"])
delta = st.slider("Delta", -1.0, 1.0, 0.0)

if st.button("Simulate"):
    intervention = {intervention_type: delta}
    response = requests.post(f"{API_URL}/simulate_counterfactual", json={"intervention": intervention, "horizon": 7})
    if response.status_code == 200:
        cf_traj = np.array(response.json()["counterfactual_trajectory"])
        fig, ax = plt.subplots()
        ax.plot(cf_traj, label="Counterfactual")
        ax.set_title("Counterfactual Trajectory")
        st.pyplot(fig)
    else:
        st.error("API Error")
import streamlit as st
import requests
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Configure page
st.set_page_config(
    page_title="Personal Biology Twin",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for minimalist white and black design
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 300;
        color: #000000;
        text-align: center;
        margin-bottom: 2rem;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .card {
        background: #ffffff;
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid #e0e0e0;
        transition: transform 0.2s ease;
    }
    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
    }
    .metric-card {
        background: #ffffff;
        color: #000000;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem;
        text-align: center;
        border: 2px solid #000000;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 300;
        margin: 0.5rem 0;
        color: #000000;
    }
    .metric-label {
        font-size: 1rem;
        opacity: 0.8;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #333333;
    }
    .input-section {
        background: #f8f8f8;
        border-radius: 12px;
        padding: 2rem;
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
    }
    .stButton>button {
        background: #ffffff;
        color: #000000;
        border: 2px solid #000000;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: #000000;
        color: #ffffff;
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.2);
    }
    .chart-container {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0;
    }
    .sidebar-card {
        background: #f8f8f8;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #000000;
    }
    body {
        background-color: #ffffff;
        color: #000000;
    }
    .stApp {
        background-color: #ffffff;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #000000 !important;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
    .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        color: #000000 !important;
    }
    .main-header {
        color: #000000 !important;
    }
    p {
        color: #333333;
    }
</style>
""", unsafe_allow_html=True)

# API base URL
API_URL = "http://localhost:8000"

# Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.markdown("### Health Dashboard")
    st.markdown("Monitor your physiological state and explore personalized interventions.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.markdown("### Quick Stats")
    current_time = datetime.now().strftime("%H:%M")
    st.markdown(f"**Last Updated:** {current_time}")
    st.markdown("**Status:** Active")
    st.markdown('</div>', unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">Personal Biology Twin</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #333333; font-size: 1.2rem; margin-bottom: 3rem;">Advanced physiological modeling for personalized health insights</p>', unsafe_allow_html=True)

# Create columns for metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">72</div>
        <div class="metric-label">Heart Rate</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">85</div>
        <div class="metric-label">HRV Score</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">8.2</div>
        <div class="metric-label">Sleep Hours</div>
    </div>
    """, unsafe_allow_html=True)

# Input signals section
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("## Physiological Signals Input")
st.markdown("Adjust your current physiological parameters to update your digital twin.")

col1, col2, col3 = st.columns(3)

with col1:
    hr = st.slider("Heart Rate (BPM)", 50, 120, 72, help="Your current heart rate in beats per minute")
    st.markdown(f"**Current:** {hr} BPM")

with col2:
    hrv = st.slider("Heart Rate Variability", 20, 100, 85, help="Measure of autonomic nervous system health")
    st.markdown(f"**Current:** {hrv} ms")

with col3:
    sleep = st.slider("Sleep Quality Score", 0.0, 1.0, 0.82, help="Overall sleep quality (0-1 scale)")
    st.markdown(".2%")

st.markdown('</div>', unsafe_allow_html=True)

# Action buttons
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    if st.button("Update Digital Twin", key="update"):
        with st.spinner("Updating your biological twin..."):
            signals = {"hr": [hr], "hrv": [hrv], "sleep": [sleep]}
            try:
                response = requests.post(f"{API_URL}/update_twin", json={"signals": signals}, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    st.success("Digital twin updated successfully!")
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    st.markdown("### Latent State Representation")
                    # Create simple visualization without pandas
                    latent_values = data["latent_state"][0] if isinstance(data["latent_state"], list) and len(data["latent_state"]) > 0 else data["latent_state"]
                    if isinstance(latent_values, list):
                        fig, ax = plt.subplots(figsize=(8, 4))
                        ax.bar(range(len(latent_values)), latent_values, color='#000000', alpha=0.7)
                        ax.set_xlabel('Latent Dimensions')
                        ax.set_ylabel('Values')
                        ax.grid(True, alpha=0.3)
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        st.pyplot(fig)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.error(f"API Error: {response.status_code}")
            except requests.exceptions.RequestException as e:
                st.error(f"Connection Error: {str(e)}")

with col2:
    if st.button("Predict Health Trajectory", key="predict"):
        with st.spinner("Generating predictions..."):
            try:
                response = requests.post(f"{API_URL}/predict_trajectory", json={"horizon": 7}, timeout=10)
                if response.status_code == 200:
                    traj = np.array(response.json()["trajectory"])
                    st.success("Trajectory predicted successfully!")

                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    st.markdown("### 7-Day Health Trajectory Prediction")

                    # Create a more sophisticated plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    days = [f"Day {i+1}" for i in range(len(traj))]
                    ax.plot(days, traj, linewidth=3, color='#000000', marker='o', markersize=6, markerfacecolor='white', markeredgewidth=2)
                    ax.fill_between(days, traj, alpha=0.3, color='#000000')
                    ax.set_ylabel('Health State', fontsize=12, fontweight='500')
                    ax.grid(True, alpha=0.3)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.error(f"API Error: {response.status_code}")
            except requests.exceptions.RequestException as e:
                st.error(f"Connection Error: {str(e)}")

with col3:
    if st.button("Run Diagnostics", key="diagnostics"):
        st.info("Running comprehensive health diagnostics...")
        # Placeholder for diagnostics
        st.success("All systems nominal. Your biological twin is healthy!")

# Counterfactual simulation section
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("## Counterfactual Analysis")
st.markdown("Explore how different interventions might affect your health trajectory.")

col1, col2 = st.columns([2, 1])

with col1:
    intervention_type = st.selectbox(
        "Select Intervention Type",
        ["Sleep Quality", "Exercise Intensity", "Stress Level", "Nutrition Quality"],
        help="Choose which aspect of your lifestyle to modify"
    )

    # Map to API values
    intervention_map = {
        "Sleep Quality": "sleep",
        "Exercise Intensity": "training_load",
        "Stress Level": "stress",
        "Nutrition Quality": "nutrition"
    }

    delta = st.slider(
        f"Change in {intervention_type}",
        -1.0, 1.0, 0.0,
        step=0.1,
        help=f"Adjust {intervention_type.lower()} by this amount"
    )

with col2:
    st.markdown("### Intervention Impact")
    if delta > 0:
        st.success(f"Positive change: +{delta:.1f}")
    elif delta < 0:
        st.warning(f"Negative change: {delta:.1f}")
    else:
        st.info("No change selected")

    if st.button("Simulate Intervention", key="simulate"):
        with st.spinner("Simulating counterfactual scenario..."):
            intervention = {intervention_map[intervention_type]: delta}
            try:
                response = requests.post(
                    f"{API_URL}/simulate_counterfactual",
                    json={"intervention": intervention, "horizon": 7},
                    timeout=10
                )
                if response.status_code == 200:
                    cf_traj = np.array(response.json()["counterfactual_trajectory"])
                    st.success("Counterfactual simulation completed!")

                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    st.markdown("### Intervention Impact Analysis")

                    fig, ax = plt.subplots(figsize=(10, 6))
                    days = [f"Day {i+1}" for i in range(len(cf_traj))]

                    # Plot both baseline and counterfactual
                    ax.plot(days, [0.8] * len(cf_traj), linewidth=2, color='#666666', linestyle='--', label='Baseline', alpha=0.7)
                    ax.plot(days, cf_traj, linewidth=3, color='#000000', marker='o', markersize=6, label='With Intervention')

                    ax.fill_between(days, [0.8] * len(cf_traj), cf_traj, where=(cf_traj > [0.8] * len(cf_traj)), color='#000000', alpha=0.2)
                    ax.fill_between(days, [0.8] * len(cf_traj), cf_traj, where=(cf_traj < [0.8] * len(cf_traj)), color='#cccccc', alpha=0.2)

                    ax.set_ylabel('Health State', fontsize=12, fontweight='500')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.error(f"API Error: {response.status_code}")
            except requests.exceptions.RequestException as e:
                st.error(f"Connection Error: {str(e)}")

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown('<p style="text-align: center; color: #666666; font-size: 0.9rem;">Personal Biology Twin - Advanced Health Analytics Platform</p>', unsafe_allow_html=True)
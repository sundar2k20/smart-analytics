import mlflow

import streamlit as st
import requests
import json
from datetime import datetime



# log the question and answer to mlflow
mlflow.set_tracking_uri("http://localhost:5000/")

mlflow.set_experiment("rag-experiment")

# API Endpoint
DEFAULT_ENDPOINT = "http://localhost:8000/predict"

st.set_page_config(
    page_title="Device Telemetry Diagnostics",
    layout="wide"
)

st.title("🔌 Device Telemetry Diagnostics")

# Endpoint Configuration
endpoint = st.sidebar.text_input(
    "API Endpoint",
    DEFAULT_ENDPOINT
)

# Device Information
st.header("Device Telemetry")

col1, col2 = st.columns(2)

with col1:
    device_id = st.text_input(
        "Device ID",
        value="BRK001"
    )

    voltage = st.number_input(
        "Voltage",
        value=415.2
    )

    current = st.number_input(
        "Current",
        value=98.5
    )

    power = st.number_input(
        "Power",
        value=40.9
    )

with col2:
    frequency = st.number_input(
        "Frequency",
        value=50.0
    )

    power_factor = st.number_input(
        "Power Factor",
        value=0.96,
        min_value=0.0,
        max_value=1.0
    )

    temperature = st.number_input(
        "Temperature",
        value=42.3
    )

    humidity = st.number_input(
        "Humidity",
        value=65.1
    )

timestamp = st.text_input(
    "Timestamp",
    value=datetime.utcnow().isoformat()
)

payload = {
    "device_id": device_id,
    "timestamp": timestamp,
    "voltage": voltage,
    "current": current,
    "power": power,
    "frequency": frequency,
    "power_factor": power_factor,
    "temperature": temperature,
    "humidity": humidity
}

st.subheader("Request Payload")

st.json(payload)

if st.button("Run Diagnostics", type="primary"):

    try:

        response = requests.post(
            endpoint,
            json=payload,
            timeout=300
        )

        st.subheader("Response Status")
        st.success(f"HTTP {response.status_code}")

        response_json = response.json()

        st.subheader("API Response")
        st.json(response_json)

        # Log to MLflow
        with mlflow.start_run():
            mlflow.log_param("device_id", device_id)
            mlflow.log_param("timestamp", timestamp)
            mlflow.log_param("voltage", voltage)
            mlflow.log_param("current", current)
            mlflow.log_param("power", power)
            mlflow.log_param("frequency", frequency)
            mlflow.log_param("power_factor", power_factor)
            mlflow.log_param("temperature", temperature)
            mlflow.log_param("humidity", humidity)

            mlflow.log_metric("health_score", response_json.get("health_score", 0))
            mlflow.log_metric("anomaly_score", response_json.get("anomaly_score", 0))

        st.subheader("API Response")
        st.json(response_json)

        

    except Exception as ex:
        st.error(f"Error calling API: {str(ex)}")
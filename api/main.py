from contextlib import contextmanager
from datetime import datetime
from typing import Optional

from pika import data
import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi import FastAPI, HTTPException, Query, Body
from pydantic import BaseModel, Field
from pymodbus import payload

import json
import logging
import os
import warnings

# Silence noisy third-party warnings BEFORE importing the libraries that emit
# them (sklearn, tensorflow, mlflow, urllib3, etc.). Setting the env vars
# first ensures TF/oneDNN logs are suppressed at import time.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")          # TensorFlow C++ logs
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")         # oneDNN reorder spam
os.environ.setdefault("PYTHONWARNINGS", "ignore")
warnings.filterwarnings("ignore")

# Quiet the most common chatty loggers used by our dependencies.
for _name in (
    "mlflow",
    "tensorflow",
    "urllib3",
    "pika",
    "sklearn",
    "absl",
):
    logging.getLogger(_name).setLevel(logging.ERROR)

import mlflow
import pika
import sys
import threading
from pathlib import Path

import mlflow
import pandas as pd

# Point MLflow at the tracking + registry server used during training so that
# `models:/...` URIs resolve against the same registry the training script
# wrote to. Without this, the API would fall back to the local file store
# (./mlruns) and raise "Registered Model ... not found".
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000/")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_registry_uri(MLFLOW_TRACKING_URI)

# Ensure the project root is on sys.path so sibling packages (database, ml, rca)
# are importable when running this script directly: `python consumer/telemetry_consumer.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.evaluate_rules import evaluate_rules, get_device_profile

# import save_telemetry from database.postgresql_writer
from database.postgresql_writer import save_telemetry, update_recommendation


from database.postgresql_writer import get_last_60_records
from database.postgresql_writer import update_ml_results

from ml.anomaly_detector import detect_anomaly
# from ml.lstm_detector import detect_degradation

from rca.root_cause_engine import determine_root_cause

from rag.query_service import generate_rag_diagnostics, generate_rag_qanda

from functools import lru_cache

mlflow.set_experiment("analytics-experiment")

class Telemetry(BaseModel):
    device_id: str = Field(..., min_length=1)
    timestamp: datetime
    voltage: float = Field(..., gt=0)
    current: float = Field(..., ge=0)
    power: float = Field(..., ge=0)
    frequency: float = Field(..., gt=0)
    power_factor: float = Field(..., ge=0, le=1)
    temperature: float
    humidity: float = Field(..., ge=0, le=100)

@lru_cache(maxsize=1000)
def get_device_profile_cached(device_id):
    return get_device_profile(device_id)


@lru_cache(maxsize=128)
def _load_model_cached(model_uri):
    """Cache mlflow.sklearn.load_model by URI.

    The first call for a given URI hits the MLflow tracking server and
    downloads the model; subsequent calls return the in-memory object.
    Cache key is the full URI, so different aliases / versions / device
    models all get their own slot.

    Call _load_model_cached.cache_clear() to force a reload (e.g. after
    promoting a new champion).
    """
    return mlflow.sklearn.load_model(model_uri)


def load_device_model(device_id, model_type="isolation_forest", alias="champion"):
    """Return the cached sklearn model for a device, loading it on first use."""
    model_uri = f"models:/{device_id}_{model_type}@{alias}"
    return _load_model_cached(model_uri)

def calculate_health_score(
        anomaly_score,
        reconstruction_error):

    score = 100

    score -= abs(anomaly_score) * 50
    score -= reconstruction_error * 500

    return max(0, min(100, score))


# ---------------- APP ----------------
app = FastAPI(
    title="Telemetry API",
    description="REST API for Modbus telemetry stored in PostgreSQL.",
    version="1.0.0"
)

# create a post method to receive telemetry data from devices and call the process_telemetry function to process the telemetry data and save to database. The process_telemetry function will be implemented in the next step.
@app.post("/ask-agent")
def receive_telemetry(data: str = Body(..., embed=True)):
    """Endpoint to receive telemetry data from devices."""

    rag_result = generate_rag_qanda(
            text=data
        )
    print("RAG Q&A:")
    print(rag_result)

    # log the question and answer to mlflow. Be defensive about the dict keys
    # returned by generate_rag_qanda (it has used both Q/A and question/answer
    # in different versions) so a missing key never leaves a run unclosed.
    mlflow.set_experiment("rag-experiment")
    try:
        with mlflow.start_run(run_name="rag_qanda"):
            question = rag_result.get("question") or rag_result.get("Q")
            answer = rag_result.get("answer") or rag_result.get("A")

            df = pd.DataFrame([{
                "input_text": data,
                "question": question,
                "answer": answer,
            }])
            mlflow.log_table(
                data=df,
                artifact_file="rag_qanda_result.json"
            )
    except Exception as exc:
        # Logging failures must not break the endpoint response.
        print(f"[ask-agent] MLflow logging failed: {exc}")
        # Make sure no run is left dangling for the next request.
        if mlflow.active_run() is not None:
            mlflow.end_run(status="FAILED")

    return rag_result

@app.post("/predict")
def predict(data: Telemetry):
    try:

        print(f"Received telemetry for device {data.device_id} at {data.timestamp.isoformat()}")
        # Pydantic model -> dict for downstream helpers that expect a mapping
        data_dict = data.model_dump()
        # Make timestamp JSON-serialisable
        if isinstance(data_dict.get("timestamp"), datetime):
            data_dict["timestamp"] = data_dict["timestamp"].isoformat()


        with mlflow.start_run(run_name=f"predict_{data.device_id}"):
          

            profile = get_device_profile_cached(data.device_id)

            # call evaluate_rules(data) to get any immediate issues based on static thresholds
            issues = evaluate_rules(data_dict, profile)

            mlflow.log_metric("issue_count", len(issues))


            print(
                f"Evaluated rules and found "
                f"{len(issues)} issues"
                )

            print(issues)

            # ----------------------------------
            # 2. Isolation Forest
            # ----------------------------------

            print(
                "Running anomaly detection..."
                )

            # load the isolation forest model for the device_id and run detect_anomaly(data) to get anomaly and anomaly_score

            # load the model from mlflow (cached per device after first call)

            print(f"Loading model {data.device_id}_isolation_forest from mlflow..."
                  )
            model = load_device_model(data.device_id)

            print(f"Loaded model {data.device_id}_isolation_forest from mlflow (cached)")


            anomaly_result = detect_anomaly(data_dict, model=model)

            anomaly = anomaly_result["anomaly"]
            anomaly_score = anomaly_result["score"]

            print(
                f"Anomaly={anomaly} "
                f"Score={anomaly_score:.4f}"
                )

            mlflow.log_metric("anomaly_score", anomaly_score)

            rag_result = generate_rag_diagnostics(
                telemetry=data_dict,
                profile=profile,
                issues=issues,
                anomaly_score=anomaly_score
            )
            print("RAG Diagnostics:")
            print(rag_result)

            mlflow.log_dict(rag_result, "rag_diagnostics.json")

            # # ----------------------------------
            # # 6. Update ML Results
            # # ----------------------------------

            health_score = calculate_health_score(anomaly_score, 0)

         
            print(
                f"Health={health_score:.2f} "
                f"Anomaly={anomaly} "
                f"RCA={None}"
            )

            return {               
                "device_id": data.device_id,
                "issues": issues,
                "anomaly": anomaly,
                "anomaly_score": anomaly_score,
                "health_score": health_score,
                "rag_diagnostics": rag_result,
            }
    except HTTPException:
        raise
    except Exception as exc:
        import traceback
        traceback.print_exc()
        # Make sure a failed request never leaves an MLflow run open.
        if mlflow.active_run() is not None:
            mlflow.end_run(status="FAILED")
        raise HTTPException(
            status_code=500,
            detail=f"{type(exc).__name__}: {exc}"
        )

@app.get("/")
def health():
    """Simple liveness + DB connectivity check."""
    try:
       
        return {"status": "ok"}
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Database unreachable: {exc}"
        )

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

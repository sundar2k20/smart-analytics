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

from rag.query_service import generate_rag_diagnostics

from functools import lru_cache

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



def process_telemetry(data):


    mlflow.set_tracking_uri("http://127.0.0.1:5000/")
    mlflow.set_experiment("analytics-experiment")

    print(
        f"Processing device "
        f"{data['device_id']}"
    )

    with mlflow.start_run(run_name=f"telemetry_{data['device_id']}"):
   

        # create a dataframe from data json for logging telemetry metrics to mlflow
        df = pd.DataFrame([data])
        mlflow.log_table(
        data=df,
        artifact_file="telemetry_table.json"
    )

        # ----------------------------------
        # 1. Save telemetry
        # ----------------------------------
        print(
            "Saving telemetry to database..."
            )
        telemetry_id = save_telemetry(data)

        profile = get_device_profile_cached(data["device_id"])

        # call evaluate_rules(data) to get any immediate issues based on static thresholds
        issues = evaluate_rules(data, profile)

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
        model = load_device_model(data["device_id"])

        print(f"Loaded model {data['device_id']}_isolation_forest from mlflow (cached)")


        anomaly_result = detect_anomaly(data, model=model)

        anomaly = anomaly_result["anomaly"]
        anomaly_score = anomaly_result["score"]

        print(
            f"Anomaly={anomaly} "
            f"Score={anomaly_score:.4f}"
            )

        mlflow.log_metric("anomaly_score", anomaly_score)

        rag_result = generate_rag_diagnostics(
            telemetry=data,
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

        update_ml_results(
            telemetry_id=telemetry_id,
            anomaly=anomaly,
            anomaly_score=anomaly_score,
            health_score=calculate_health_score(anomaly_score, 0),
            root_cause=None
        )

        print(
            f"Health={calculate_health_score(anomaly_score, 0):.2f} "
            f"Anomaly={anomaly} "
            f"RCA={None}"
        )

    # recommendation = get_recommendation(
    # telemetry=data,
    # root_cause=None,
    # anomaly=anomaly,
    # health_score=calculate_health_score(anomaly_score, 0)
    # )

    # print(recommendation)

    # update_recommendation(
    # telemetry_id,
    # recommendation
    # )


def _ack(ch, delivery_tag):
    """Run on the pika I/O thread."""
    if ch.is_open:
        ch.basic_ack(delivery_tag=delivery_tag)


def _nack(ch, delivery_tag):
    """Run on the pika I/O thread."""
    if ch.is_open:
        ch.basic_nack(delivery_tag=delivery_tag, requeue=False)


def _do_work(connection, ch, delivery_tag, body):
    """Heavy processing runs in a worker thread so the pika I/O loop
    stays responsive to RabbitMQ heartbeats."""
    try:
        data = json.loads(body)

        # Normalize camelCase keys from the gateway producer
        # to the snake_case names used by the DB and ML layers.
        key_map = {
            "deviceId": "device_id",
            "powerFactor": "power_factor",
        }
        for src, dst in key_map.items():
            if src in data and dst not in data:
                data[dst] = data.pop(src)

        process_telemetry(data)

        connection.add_callback_threadsafe(
            lambda: _ack(ch, delivery_tag)
        )

    except Exception as ex:
        print(f"Error processing message: {ex}")
        connection.add_callback_threadsafe(
            lambda: _nack(ch, delivery_tag)
        )


def callback(
        ch,
        method,
        properties,
        body):

    connection = ch.connection
    delivery_tag = method.delivery_tag

    worker = threading.Thread(
        target=_do_work,
        args=(connection, ch, delivery_tag, body),
        daemon=True,
    )
    worker.start()


def start_consumer():

    connection = pika.BlockingConnection(
        pika.ConnectionParameters(
            host="localhost",
            heartbeat=600,
            blocked_connection_timeout=300,
        )
    )

    channel = connection.channel()

    channel.queue_declare(
        queue="telemetry",
        durable=True
    )

    channel.basic_qos(
        prefetch_count=1
    )

    channel.basic_consume(
        queue="telemetry",
        on_message_callback=callback
    )

    print(
        "Waiting for telemetry messages..."
    )

    channel.start_consuming()


if __name__ == "__main__":
    start_consumer()
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

_MODELS_DIR = Path(__file__).resolve().parent / "models"

# Column order MUST match the training order used in train_isolation_forest.py
_FEATURES = [
    "voltage",
    "current",
    "power_factor",
    "frequency",
    "temperature",
    "humidity",
]

#model = joblib.load(_MODELS_DIR / "isolation_forest.pkl")
#scaler = joblib.load(_MODELS_DIR / "scaler.pkl")

def detect_anomaly(data, model):

    device_id = data["device_id"]

    # Build a single-row DataFrame with the same column names used at
    # training time. This silences sklearn's "X does not have valid feature
    # names, but StandardScaler was fitted with feature names" warning.
    row = pd.DataFrame(
        [[data[f] for f in _FEATURES]],
        columns=_FEATURES,
    )

    #model = joblib.load(_MODELS_DIR / f"{device_id}_isolation_forest.pkl")
    #scaler = joblib.load(_MODELS_DIR / f"{device_id}_isolation_forest_scaler.pkl")

    #X = scaler.transform(row)

    prediction = model.predict(row)

    score = model.decision_function(row)[0]

    return {
        "anomaly": bool(prediction[0] == -1),
        "score": float(score)
    }
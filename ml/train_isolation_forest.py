import sys
from pathlib import Path

# Ensure project root is on sys.path so sibling packages (e.g. `database`)
# can be imported when this script is executed directly from the `ml/` dir.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlflow
import psycopg2
from sklearn.ensemble import IsolationForest
from joblib import dump
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

from database.db_config import get_db_config

from mlflow import MlflowClient
from sklearn.pipeline import Pipeline



# Point MLflow at the tracking + registry server BEFORE constructing the
# MlflowClient. Otherwise the client binds to the default local file store
# while log_model / register_model write to the HTTP server, causing
# "Registered Model ... not found" when looking up aliases.
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000/"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_registry_uri(MLFLOW_TRACKING_URI)

client = MlflowClient(
    tracking_uri=MLFLOW_TRACKING_URI,
    registry_uri=MLFLOW_TRACKING_URI,
)


features = [
    "voltage",
    "current",
    "power_factor",
    "frequency",
    "temperature",
    "humidity"
]

model_type = "isolation_forest"

# Resolve models directory relative to this script so it works regardless
# of the current working directory, and ensure it exists.
MODELS_DIR = Path(__file__).resolve().parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# create list of device ids

conn = psycopg2.connect(**get_db_config())

# Ensure the ml_models table exists, then self-heal the schema by adding
# any optional columns that newer versions of this script depend on. Both
# statements are idempotent (IF NOT EXISTS) so they're safe to re-run.
with conn.cursor() as cur:
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS ml_models (
            id           SERIAL PRIMARY KEY,
            device_id    TEXT        NOT NULL,
            model_type   TEXT        NOT NULL,
            trained_on   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            sample_count INTEGER,
            model_path   TEXT,
            scaler_path  TEXT
        )
        """
    )
    cur.execute(
        "ALTER TABLE ml_models ADD COLUMN IF NOT EXISTS scaler_path TEXT"
    )
    conn.commit()

with conn.cursor() as cur:
    cur.execute("SELECT device_id FROM device_profile")
    device_ids = [row[0] for row in cur.fetchall()]


mlflow.set_experiment("analytics-experiment")

# train a separate model for each device id
for device_id in device_ids:

 with mlflow.start_run(run_name=f"train_{device_id}"):

    # based on devive_id, load the training data from database table named "device_training_data" with columns  device_id, device_type, voltage, current, frequency, power_factor, temperature, humidity
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT voltage, current, power_factor, frequency, temperature, humidity
            FROM device_training_data
            WHERE device_id = %s
            """,
            (device_id,)
        )
        rows = cur.fetchall()
    

    df = pd.DataFrame(rows, columns=[   
        "voltage",
        "current",
        "power_factor",
        "frequency",
        "temperature",
        "humidity"
    ])

    X = df[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        contamination=0.02,
        n_estimators=200,
        random_state=42
    )

    model.fit(X_scaled)

    pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", IsolationForest(
        contamination=0.05,
        n_estimators=200,
        random_state=42
            ))
        ])

    pipeline.fit(X)  # Fit the pipeline on the original (unscaled) data; the scaler is part of the pipeline and will be applied automatically during prediction 


    print("Isolation Forest trained")

    model_path = MODELS_DIR / f"{device_id}_{model_type}.pkl"
    scaler_path = MODELS_DIR / f"{device_id}_{model_type}_scaler.pkl"

    dump(pipeline, model_path)
    #dump(scaler, scaler_path)

    print(f"Model saved to {model_path}")
   # print(f"Scaler saved to {scaler_path}")

  
    mlflow.log_param("algorithm", model_type)
    mlflow.log_param("contamination", 0.02)
    mlflow.log_param("n_estimators", 200)


    model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model"
        )

    model_uri = model_info.model_uri
    registered_name = f"{device_id}_{model_type}"

    # Explicitly register the logged model. This is more reliable across
    # MLflow versions than passing `registered_model_name=` to log_model,
    # and gives us back a ModelVersion with a stable `.version` attribute.
    model_version_info = mlflow.register_model(
        model_uri=model_uri,
        name=registered_name
    )
    model_version = model_version_info.version

    # add alias "champion" to the registered model version we just created above
    client.set_registered_model_alias(
        name=registered_name,
        alias="champion",
        version=str(model_version)
    )

    # mlflow.register_model(
    #         model_uri=model_uri,
    #         name=f"{device_id}_{model_type}",
    #             tags={
    #                 "device_id": device_id,
    #                 "model_type": model_type
    #             },
    #             registration_name=f"{device_id}_{model_type}",
    #             alias="champion"

            
    #     )

    

    print("Model registered successfully")
 

   
    # # log both model and scaler as artifacts in mlflow
    # mlflow.log_artifact(model_path, artifact_path="models")
    # mlflow.log_artifact(scaler_path, artifact_path="models")
    

    print(f"Model and scaler artifacts logged to mlflow for device_id={device_id}")
    print(f"Model metadata will be saved to database for device_id={device_id}")

    # update the table named ml_models with columns device_id, model_type, trained_on,sample_count, model_path, scaler_path
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO ml_models (
                device_id,
                model_type,
                trained_on,
                sample_count,
                model_path,
                scaler_path
            ) VALUES (%s, %s, NOW(), %s, %s, %s)
            """,
            (
                device_id,
                "isolation_forest",
                len(df),
                str(model_path),
                str(scaler_path)
            )
        )
        conn.commit()
    print(
        f"Model metadata saved to database for device_id={device_id}")
    
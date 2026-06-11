import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

df = pd.read_csv("training/telemetry.csv")

features = [
    "voltage",
    "current",
    "power_factor",
    "frequency",
    "temperature",
    "humidity"
]

X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = IsolationForest(
    contamination=0.02,
    n_estimators=200,
    random_state=42
)

model.fit(X_scaled)

joblib.dump(model, "models/isolation_forest.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("Isolation Forest trained")
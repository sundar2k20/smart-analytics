import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from keras.models import Model
from keras.layers import (
    Input,
    LSTM,
    RepeatVector
)

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

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

window = 2

sequences = []

for i in range(len(X_scaled)-window):
    sequences.append(
        X_scaled[i:i+window]
    )

if not sequences:
    raise ValueError(
        f"Not enough rows in telemetry.csv to build sequences of length {window}. "
        f"Got {len(X_scaled)} rows; need at least {window + 1}."
    )

X_train = np.array(sequences)

inputs = Input(shape=(window, len(features)))

encoded = LSTM(64)(inputs)

decoded = RepeatVector(window)(encoded)

decoded = LSTM(
    len(features),
    return_sequences=True
)(decoded)

model = Model(inputs, decoded)

model.compile(
    optimizer="adam",
    loss="mse"
)

model.fit(
    X_train,
    X_train,
    epochs=20,
    batch_size=32
)

model.save("models/lstm_autoencoder.keras")
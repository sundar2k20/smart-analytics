

from ml.anomaly_detector import detect_anomaly


detect_anomaly()


# Runs every 5–15 minutes using the last 60–300 telemetry samples from PostgreSQL/TimescaleDB.

# This separation scales much better when you have hundreds or thousands of meters, breakers, and sensors streaming data continuously.
import json
import pika
import psycopg2
from datetime import datetime
import sys
from pathlib import Path
import numpy as np

# Ensure the project root is on sys.path so sibling packages (e.g. `producer`)
# are importable when running this script directly: `python gateway/modbus_reader.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from psycopg2.pool import SimpleConnectionPool

from database.db_config import get_db_config, get_pool_config

pool = SimpleConnectionPool(**get_pool_config())


# ---------------- DB CONNECTION ----------------
conn = psycopg2.connect(**get_db_config())

# def process_telemetry(data):

#     anomaly_result = detect_anomaly(data)

#     save_to_db(data)

#     recent_data = get_last_60_records(
#         data["device_id"]
#     )

#     if len(recent_data) == 60:

#         degradation_result = detect_degradation(
#             recent_data
#         )

#         print(degradation_result)
        

# sequence = get_last_60_records("Breaker-101")

# if len(sequence) == 60:
#     result = detect_degradation(sequence)

#     print(result)

# Isolation Forest:

# anomaly = detect_anomaly(data)

# degradation = detect_degradation(
#     sequence
# )
# {
#   "abnormal": true,
#   "reconstruction_error": 0.041
# }


def get_last_60_records(device_id):

    conn = pool.getconn()

    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    voltage,
                    current,
                    power_factor,
                    frequency,
                    temperature,
                    humidity
                FROM telemetry
                WHERE device_id = %s
                ORDER BY ts DESC
                LIMIT 60
            """, (device_id,))

            rows = cur.fetchall()

        rows.reverse()

        return np.array(rows, dtype=np.float32)

    finally:
        pool.putconn(conn)
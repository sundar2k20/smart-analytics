import numpy as np
import pandas as pd
import psycopg2
from database.db_util import get_last_60_records

from psycopg2.pool import SimpleConnectionPool

from database.db_config import get_db_config, get_pool_config

pool = SimpleConnectionPool(**get_pool_config())


# ---------------- DB CONNECTION ----------------
conn = psycopg2.connect(**get_db_config())

def get_device_profile(device_id):
    cursor = conn.cursor()

    cursor.execute("""
        SELECT *
        FROM device_profile
        WHERE device_id = %s
    """, (device_id,))

    row = cursor.fetchone()

    if not row:
        return None

    columns = [desc[0] for desc in cursor.description]

    return dict(zip(columns, row))

def evaluate_rules(telemetry,profile):

    issues = []

    print(profile)


    if telemetry["voltage"] > profile["max_voltage"]:
        issues.append({
            "code": "OVER_VOLTAGE",
            "severity": "HIGH"
        })

    if telemetry["voltage"] < profile["min_voltage"]:
        issues.append({
            "code": "UNDER_VOLTAGE",
            "severity": "MEDIUM"
        })

    if telemetry["current"] > profile["max_current"]:
        issues.append({
            "code": "OVER_CURRENT",
            "severity": "HIGH"
        })

    if telemetry["power_factor"] < profile["min_power_factor"]:
        issues.append({
            "code": "LOW_POWER_FACTOR",
            "severity": "MEDIUM"
        })

    return issues
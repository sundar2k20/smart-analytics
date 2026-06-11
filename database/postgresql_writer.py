import psycopg2
import numpy as np
from psycopg2.pool import SimpleConnectionPool
import json
from pathlib import Path

from database.db_config import get_pool_config

pool = SimpleConnectionPool(**get_pool_config())


def save_telemetry(data):
    """
    Inserts telemetry record and returns generated ID
    """

    conn = pool.getconn()

    try:
        with conn.cursor() as cur:

            cur.execute("""
                INSERT INTO telemetry (
                    device_id,
                    ts,
                    voltage,
                    current,
                    power,
                    frequency,
                    power_factor,
                    temperature,
                    humidity
                )
                VALUES (
                    %s,%s,%s,%s,%s,%s,%s,%s,%s
                )
                RETURNING id
            """,
            (
                data["device_id"],
                data["timestamp"],
                data["voltage"],
                data["current"],
                data["power"],
                data["frequency"],
                data["power_factor"],
                data["temperature"],
                data["humidity"]
            ))

            telemetry_id = cur.fetchone()[0]

        conn.commit()

        return telemetry_id

    except Exception:
        conn.rollback()
        raise

    finally:
        pool.putconn(conn)


def get_last_60_records(device_id):
    """
    Returns last 60 telemetry records
    in ascending timestamp order
    for LSTM processing
    """

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

        return np.array(
            rows,
            dtype=np.float32
        )

    finally:
        pool.putconn(conn)


def update_ml_results(
        telemetry_id,
        anomaly,
        anomaly_score,
        health_score,
        root_cause):
    """
    Updates ML analysis results
    """

    conn = pool.getconn()

    try:
        with conn.cursor() as cur:

            cur.execute("""
                UPDATE telemetry
                SET
                    anomaly = %s,
                    anomaly_score = %s,
                    health_score = %s,
                    root_cause = %s
                WHERE id = %s
            """,
            (
                anomaly,
                anomaly_score,
                health_score,
                root_cause,
                telemetry_id
            ))

        conn.commit()

    except Exception:
        conn.rollback()
        raise

    finally:
        pool.putconn(conn)

def update_recommendation(
        telemetry_id,
        recommendation_json):

    conn = pool.getconn()

    try:
        with conn.cursor() as cur:

            cur.execute("""
                UPDATE telemetry
                SET recommendation_json = %s
                WHERE id = %s
            """,
            (
                json.dumps(recommendation_json),
                telemetry_id
            ))

        conn.commit()

    except Exception:
        conn.rollback()
        raise

    finally:
        pool.putconn(conn)

def close_pool():
    pool.closeall()

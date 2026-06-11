import random
import threading
import time
import sys
from pathlib import Path

# Ensure project root is on sys.path when running this script directly:
# `python gateway/modbus_generator.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psycopg2
from database.db_config import get_db_config

from pymodbus.server import StartTcpServer
from pymodbus.datastore import (
    ModbusSequentialDataBlock,
    ModbusSlaveContext,
    ModbusServerContext
)

# ---- Register layout ----
# Keep these in sync with gateway/modbus_reader.py
DEVICE_ID_REGS = 16          # 32 ASCII chars max
TOTAL_REGS = DEVICE_ID_REGS + 10   # +timestamp(2) +V +I +P(2) +F +PF +T +H

# get device ids from postgres table named "devices"

conn = psycopg2.connect(**get_db_config())
with conn.cursor() as cur:
    cur.execute("SELECT device_id FROM device_profile")
    devices = [row[0] for row in cur.fetchall()]



# Holding register block sized for the layout above
store = ModbusSlaveContext(
    hr=ModbusSequentialDataBlock(0, [0] * TOTAL_REGS)
)

context = ModbusServerContext(
    slaves=store,
    single=True
)


def encode_string(text: str, register_count: int):
    """
    Convert string to Modbus registers.
    2 ASCII chars per register.
    Truncates if `text` exceeds register_count * 2 bytes so the returned
    list is always exactly `register_count` registers long (prevents the
    rest of the payload from being shifted).
    """
    max_bytes = register_count * 2
    data = text.encode("ascii")[:max_bytes]
    data = data.ljust(max_bytes, b"\x00")

    regs = []

    for i in range(0, len(data), 2):
        regs.append((data[i] << 8) | data[i + 1])

    return regs


def update_values():

    while True:

        # Device ID
        #device_id = f"BRK{random.randint(1,999):03d}"
        device_id = random.choice(devices)

        # Unix timestamp
        timestamp = int(time.time())

        # voltage = random.randint(410, 420)
        # current = random.randint(180, 220)

        voltage = random.randint(500, 600)  # Inject voltage anomaly occasionally
        current = random.randint(230, 300)  # Inject current anomaly occasionally

        power_factor = round(
            random.uniform(0.90, 0.99),
            2
        )

        # Real power in watts (3-phase approximation)
        power = int(voltage * current * power_factor)

        # Frequency: 49.50 - 50.50 Hz
        frequency = round(
            random.uniform(49.50, 50.50),
            2
        )

        # Inject temperature anomaly occasionally
        if random.random() < 0.05:
            temperature = random.randint(90, 110)
        else:
            temperature = random.randint(35, 50)

        humidity = random.randint(30, 70)

        device_regs = encode_string(
            device_id,
            DEVICE_ID_REGS
        )

        registers = [
            *device_regs,                  # 0  .. DEVICE_ID_REGS-1
            timestamp & 0xFFFF,            # +0
            (timestamp >> 16) & 0xFFFF,    # +1
            voltage,                       # +2
            current,                       # +3
            power & 0xFFFF,                # +4
            (power >> 16) & 0xFFFF,        # +5
            int(frequency * 100),          # +6
            int(power_factor * 100),       # +7
            temperature,                   # +8
            humidity                       # +9
        ]

        store.setValues(
            3,      # Holding Registers
            0,
            registers
        )

        print(
            f"Device={device_id} "
            f"TS={timestamp} "
            f"V={voltage} "
            f"I={current} "
            f"P={power} "
            f"F={frequency} "
            f"PF={power_factor} "
            f"T={temperature} "
            f"H={humidity}"
        )

        time.sleep(10)  # Update every 5 minutes


threading.Thread(
    target=update_values,
    daemon=True
).start()

print("Starting Modbus TCP Server on port 5020")

StartTcpServer(
    context=context,
    address=("0.0.0.0", 5020)
)
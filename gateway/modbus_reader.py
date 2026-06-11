import sys
from pathlib import Path

# Ensure the project root is on sys.path so sibling packages (e.g. `producer`)
# are importable when running this script directly: `python gateway/modbus_reader.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pymodbus.client import ModbusTcpClient
from pymodbus.exceptions import ModbusException
from datetime import datetime
import time


# import RabbitMQ producer
from producer.rabbitmq_producer import RabbitMqProducer

# ---- Register layout ----
# Keep these in sync with gateway/modbus_generator.py
DEVICE_ID_REGS = 16          # 32 ASCII chars max
TOTAL_REGS = DEVICE_ID_REGS + 10

client = ModbusTcpClient("127.0.0.1", port=5020)

client.connect()

producer = RabbitMqProducer()

while True:

    try:
        if not client.connected:
            client.connect()

        result = client.read_holding_registers(
            address=0,
            count=TOTAL_REGS,
            slave=1
        )

        if result.isError():
            print(f"Modbus error response: {result}")
            time.sleep(5)
            continue

        regs = result.registers
    except ModbusException as exc:
        print(f"Modbus exception: {exc}")
        time.sleep(5)
        continue
    except Exception as exc:
        print(f"Unexpected error reading registers: {exc}")
        time.sleep(5)
        continue

    # Decode DeviceId
    raw = bytearray()

    for reg in regs[0:DEVICE_ID_REGS]:
        raw.append((reg >> 8) & 0xFF)
        raw.append(reg & 0xFF)

    device_id = raw.decode(
        "ascii",
        errors="ignore"
    ).rstrip("\x00")

    # Field offsets follow the device-id block
    base = DEVICE_ID_REGS

    # Decode timestamp
    timestamp = regs[base] | (regs[base + 1] << 16)

    voltage = regs[base + 2]
    current = regs[base + 3]
    power = regs[base + 4] | (regs[base + 5] << 16)
    frequency = regs[base + 6] / 100.0
    power_factor = regs[base + 7] / 100.0
    temperature = regs[base + 8]
    humidity = regs[base + 9]

    payload = {
        "deviceId": device_id,
        "timestamp": datetime.fromtimestamp(timestamp).isoformat(),
        "voltage": voltage,
        "current": current,
        "power": power,
        "frequency": frequency,
        "powerFactor": power_factor,
        "temperature": temperature,
        "humidity": humidity
    }

    print(payload)

    producer.publish(payload)

    time.sleep(30)  # Read every 5 minutes
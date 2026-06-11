"""Rule-based root cause analysis for electrical telemetry."""

# Operational thresholds (tune for your equipment)
VOLTAGE_NOMINAL = 415
VOLTAGE_TOLERANCE = 0.10          # ±10%
CURRENT_OVERLOAD = 100            # amps
POWER_FACTOR_LOW = 0.85
FREQUENCY_NOMINAL = 50
FREQUENCY_TOLERANCE = 1.0         # Hz
TEMPERATURE_HIGH = 60             # °C
HUMIDITY_HIGH = 80                # %RH


def determine_root_cause(telemetry, anomaly=False, degradation=False):
    """Return a short human-readable root cause string.

    `telemetry` is a dict with keys: voltage, current, power_factor,
    frequency, temperature, humidity (any may be missing).
    """

    causes = []

    voltage = telemetry.get("voltage")
    current = telemetry.get("current")
    pf = telemetry.get("power_factor")
    frequency = telemetry.get("frequency")
    temperature = telemetry.get("temperature")
    humidity = telemetry.get("humidity")

    if voltage is not None:
        low = VOLTAGE_NOMINAL * (1 - VOLTAGE_TOLERANCE)
        high = VOLTAGE_NOMINAL * (1 + VOLTAGE_TOLERANCE)
        if voltage < low:
            causes.append("undervoltage")
        elif voltage > high:
            causes.append("overvoltage")

    if current is not None and current > CURRENT_OVERLOAD:
        causes.append("current overload")

    if pf is not None and pf < POWER_FACTOR_LOW:
        causes.append("poor power factor")

    if frequency is not None and abs(frequency - FREQUENCY_NOMINAL) > FREQUENCY_TOLERANCE:
        causes.append("frequency deviation")

    if temperature is not None and temperature > TEMPERATURE_HIGH:
        causes.append("overheating")

    if humidity is not None and humidity > HUMIDITY_HIGH:
        causes.append("high humidity")

    if degradation:
        causes.append("long-term degradation pattern")

    if not causes:
        if anomaly:
            return "anomalous behavior (no specific rule match)"
        return "normal"

    return "; ".join(causes)

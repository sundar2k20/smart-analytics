

from database.db_util import get_last_60_records
from ml.lstm_detector import detect_degradation


sequence = get_last_60_records("Breaker-101")

if len(sequence) == 60:
    result = detect_degradation(sequence)

    print(result)

detect_degradation()
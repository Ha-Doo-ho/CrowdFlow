import json
import os
import time

import numpy as np


def to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def save_latest_result(grid_result, json_path, retries=20, retry_delay=0.05):
    directory = os.path.dirname(json_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    tmp_path = f"{json_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as result_file:
        json.dump(
            to_jsonable(grid_result),
            result_file,
            ensure_ascii=False,
            indent=2,
        )
    for attempt in range(retries + 1):
        try:
            os.replace(tmp_path, json_path)
            return
        except PermissionError:
            if attempt == retries:
                raise
            time.sleep(retry_delay)

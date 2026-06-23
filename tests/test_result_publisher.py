import json

import numpy as np

from core.result_publisher import save_latest_result


def test_save_latest_result_converts_numpy_values(tmp_path):
    output_path = tmp_path / "latest_result.json"
    result = {
        "grid": np.array([[0.25, 0.5]]),
        "count": np.array([[1, 2]], dtype=np.int64),
        "max_density": np.float64(0.5),
    }

    save_latest_result(result, str(output_path))

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["grid"] == [[0.25, 0.5]]
    assert saved["count"] == [[1, 2]]
    assert saved["max_density"] == 0.5
    assert not (tmp_path / "latest_result.json.tmp").exists()

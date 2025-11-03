import subprocess
import time

import pandas as pd

from evomind import EvoMind


def test_full_pipeline_produces_artifacts(tmp_path):
    df = pd.DataFrame(
        {
            "num_feature": [1, 2, 3, 4, 5, 6, 7, 8],
            "cat_feature": ["A", "B", "A", "B", "A", "B", "A", "B"],
            "target": [10, 12, 14, 16, 18, 20, 22, 24],
        }
    )

    config = {
        "engine": {
            "generations": 1,
            "population": 4,
            "epochs": 1,
            "batch_size": 4,
            "parallel": False,
        },
        "reporting": {"export_formats": ["html"]},
        "data": {"sensitive_feature": None},
    }

    runner = EvoMind(data=df, task="regression", insights=True, config=config)
    result = runner.run()

    output_dir = result.output_dir
    assert output_dir.exists()
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "model_card.html").exists()
    assert (output_dir / "reports" / "report.html").exists()

    proc = result.launch_dashboard(port=8760)
    try:
        time.sleep(2)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()

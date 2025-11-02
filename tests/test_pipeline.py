from pathlib import Path

from evomind import EvoMind


def test_evomind_run_smoke(tmp_path):
    config_overrides = {
        "engine": {
            "generations": 1,
            "population": 4,
            "epochs": 1,
            "batch_size": 8,
            "learning_rate": 1e-3,
            "parallel": False,
        },
        "data": {
            "sensitive_feature": None,
        },
    }
    runner = EvoMind(
        data=Path("data/grocery_chain_data.json"),
        task="auto",
        insights=False,
        config=config_overrides,
    )
    result = runner.run()
    assert result.metrics
    assert Path(result.output_dir).exists()

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


def test_multi_csv_ingestion(tmp_path: Path) -> None:
    file_a = tmp_path / "part_a.csv"
    file_b = tmp_path / "part_b.csv"
    file_a.write_text("feature,target\n1,10\n2,20\n", encoding="utf-8")
    file_b.write_text("feature,target\n3,30\n4,40\n", encoding="utf-8")
    evo = EvoMind(
        data=[file_a, file_b],
        task="regression",
        insights=False,
        config={"engine": {"generations": 1, "population": 2, "parallel": False}},
    )
    df, path = evo._load_dataframe()  # noqa: SLF001 - intentional for test
    assert path is None
    assert len(df) == 4


def test_custom_run_name(tmp_path: Path) -> None:
    csv_path = tmp_path / "tiny.csv"
    csv_path.write_text("feature,target\n1,1\n2,2\n", encoding="utf-8")
    evo = EvoMind(
        data=csv_path,
        task="regression",
        run_name="custom_run",
        config={"engine": {"generations": 1, "population": 2, "parallel": False}},
    )
    assert evo.run_id.startswith("custom-run")

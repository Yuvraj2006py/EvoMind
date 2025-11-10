from pathlib import Path

from evomind import EvoMind


def test_describe_config_dict_output() -> None:
    data = EvoMind.describe_config(section="engine")
    assert "engine" in data
    assert "population" in data["engine"]


def test_describe_config_markdown_console(capsys) -> None:
    EvoMind.describe_config(section="engine", as_markdown=True, to_console=True)
    captured = capsys.readouterr().out
    assert "EvoMind Configuration Reference" in captured or "## Engine" in captured


def test_explain_config_key() -> None:
    text = EvoMind.explain("population")
    assert "population" in text.lower()
    assert "Number of genomes" in text or "genomes" in text


def test_generate_config_docs(tmp_path: Path) -> None:
    output = tmp_path / "CONFIG.md"
    EvoMind.generate_config_docs(output)
    assert output.exists()
    content = output.read_text(encoding="utf-8")
    assert "Configuration Reference" in content

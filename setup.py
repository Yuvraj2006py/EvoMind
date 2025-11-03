from pathlib import Path

from setuptools import find_packages, setup

try:
    from evomind.utils.config_reference import write_markdown as write_config_markdown
except Exception as exc:  # pragma: no cover - setup-time safety
    print(f"Warning: unable to import config reference generator: {exc}")
    write_config_markdown = None

if write_config_markdown:
    try:
        write_config_markdown(Path("docs") / "config_reference.md")
    except Exception as exc:  # pragma: no cover - setup-time safety
        print(f"Warning: unable to generate config reference: {exc}")


setup(
    name="EvoMind",
    version="2.0",
    packages=find_packages(),
    install_requires=Path("requirements.txt").read_text(encoding="utf-8").splitlines(),
)

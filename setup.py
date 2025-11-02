from pathlib import Path

from setuptools import find_packages, setup

setup(
    name="EvoMind",
    version="2.0",
    packages=find_packages(),
    install_requires=Path("requirements.txt").read_text(encoding="utf-8").splitlines(),
)

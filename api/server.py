"""
FastAPI application exposing EvoMind AutoML as a remote service.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, UploadFile

from evomind import EvoMind

app = FastAPI(title="EvoMind API", version="2.0")


@app.post("/run")
async def run_task(file: UploadFile, task: str = "auto", insights: bool = True) -> Dict[str, Any]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename or "dataset").suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        dataset_path = Path(tmp.name)

    try:
        runner = EvoMind(data=dataset_path, task=task, insights=insights)
        result = runner.run()
        report_path = result.export_report("html")
        return {
            "run_id": result.run_id,
            "metrics": result.metrics,
            "report": str(report_path) if report_path else None,
        }
    finally:
        if dataset_path.exists():
            dataset_path.unlink()

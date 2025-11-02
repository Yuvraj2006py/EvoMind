"""
Build HTML and PDF reports for EvoMind experiments.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape

TEMPLATE_DIR = Path(__file__).resolve().parent

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <title>EvoMind Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 2rem; background: #f6f6f6; color: #222; }
        h1, h2 { color: #2c3e50; }
        section { background: #fff; padding: 1.5rem; margin-bottom: 1.5rem; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        table { width: 100%; border-collapse: collapse; margin-top: 1rem; }
        th, td { border: 1px solid #ccc; padding: 0.6rem; text-align: left; }
        th { background: #eef2f5; }
        .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 0.75rem; }
        .metric-card { background: linear-gradient(130deg, #3a7bd5, #00d2ff); color: #fff; padding: 1rem; border-radius: 10px; }
        .metric-card h3 { margin: 0 0 0.5rem 0; font-size: 1rem; }
        .metric-card p { margin: 0; font-size: 1.2rem; font-weight: bold; }
        .image-block { display: flex; flex-wrap: wrap; gap: 1rem; }
        .image-block img { max-width: 48%; border-radius: 6px; border: 1px solid #ddd; }
        .narrative { font-style: italic; background: #f0f5f9; padding: 1rem; border-left: 4px solid #3a7bd5; border-radius: 6px; }
    </style>
</head>
<body>
    <header>
        <h1>EvoMind Experiment Report</h1>
        <p>Run ID: {{ run_id }}</p>
    </header>

    <section>
        <h2>Executive Summary</h2>
        <p class="narrative">{{ insight_summary }}</p>
    </section>

    <section>
        <h2>Dataset Overview (Health Score: {{ data_profile.health_score }}%)</h2>
        <div class="metric-grid">
            <div class="metric-card"><h3>Rows</h3><p>{{ data_profile.rows }}</p></div>
            <div class="metric-card"><h3>Columns</h3><p>{{ data_profile.columns }}</p></div>
            <div class="metric-card"><h3>Missing Columns</h3><p>{{ data_profile.missing_columns }}</p></div>
            <div class="metric-card"><h3>Outlier Columns</h3><p>{{ data_profile.outlier_columns }}</p></div>
        </div>
        <table>
            <tr><th>Column</th><th>Missing %</th><th>Outlier %</th><th>Skewness</th><th>Kurtosis</th></tr>
            {% for row in data_profile.details %}
            <tr>
                <td>{{ row.column }}</td>
                <td>{{ "{:.2f}".format(row.missing_pct * 100) }}</td>
                <td>{{ "{:.2f}".format(row.outlier_pct * 100) }}</td>
                <td>{{ "{:.2f}".format(row.skewness or 0) }}</td>
                <td>{{ "{:.2f}".format(row.kurtosis or 0) }}</td>
            </tr>
            {% endfor %}
        </table>
    </section>

    <section>
        <h2>Model Performance</h2>
        <div class="metric-grid">
            {% for name, value in model_metrics.items() %}
            <div class="metric-card">
                <h3>{{ name.replace("_", " ").title() }}</h3>
                <p>{{ "{:.4f}".format(value) }}</p>
            </div>
            {% endfor %}
        </div>
        <p>Status: {{ stability_label }}</p>
    </section>\n\n    {% if fairness %}\n    <section>\n        <h2>Fairness Diagnostics</h2>\n        <p>Demographic parity gap: {{ '{:.3f}'.format(fairness.get('demographic_parity')) if fairness.get('demographic_parity') is not none else 'N/A' }}</p>\n        <p>Equal opportunity gap: {{ '{:.3f}'.format(fairness.get('equal_opportunity')) if fairness.get('equal_opportunity') is not none else 'N/A' }}</p>\n    </section>\n    {% endif %}\n\n    <section>\n        <h2>Feature Insights</h2>
        <p>Top Features: {{ top_features }}</p>
        <div class="image-block">
            {% for img in feature_images %}
            <div>
                <img src="{{ img }}" alt="Feature plot" />
            </div>
            {% endfor %}
        </div>
    </section>

    <section>
        <h2>Correlation & Relationships</h2>
        <p>Notable correlations: {{ correlations }}</p>
        {% if correlation_image %}
        <img src="{{ correlation_image }}" alt="Correlation heatmap" style="width:100%; border-radius: 8px; border: 1px solid #ddd;" />
        {% endif %}
        {% if mutual_info %}
        <table>
            <tr><th>Feature</th><th>Mutual Information</th></tr>
            {% for item in mutual_info %}
            <tr><td>{{ item.feature }}</td><td>{{ "{:.4f}".format(item.score) }}</td></tr>
            {% endfor %}
        </table>
        {% endif %}
    </section>

</body>
</html>
"""


def _render_html(context: Dict[str, object]) -> str:
    env = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.from_string(HTML_TEMPLATE)
    return template.render(**context)


def build_report(
    output_dir: Path,
    context: Dict[str, object],
) -> Dict[str, Optional[Path]]:
    """
    Generate HTML (and optionally PDF) reports summarising an EvoMind run.

    Parameters
    ----------
    output_dir : Path
        Directory where the report files should be written.
    context : dict
        Data required to populate the report template.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    html_content = _render_html(context)
    html_path = output_dir / "report.html"
    html_path.write_text(html_content, encoding="utf-8")

    report_paths: Dict[str, Optional[Path]] = {"html": html_path, "pdf": None}

    try:
        from weasyprint import HTML  # type: ignore[import]

        pdf_path = output_dir / "report.pdf"
        HTML(string=html_content, base_url=str(output_dir)).write_pdf(str(pdf_path))
        report_paths["pdf"] = pdf_path
    except Exception as exc:  # pragma: no cover - optional dependency
        fallback = output_dir / "report_warning.json"
        fallback.write_text(json.dumps({"warning": f"PDF generation skipped: {exc}"}), encoding="utf-8")

    return report_paths



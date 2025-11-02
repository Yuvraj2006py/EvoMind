"""
Streamlit dashboard surfacing EvoMind experiment insights.

The dashboard reads artifacts persisted by the EvoMind run (history, insights,
reports) and presents them through multiple navigation tabs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

# Ensure package imports resolve when running `streamlit run dashboard/app.py`.
ROOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from dashboard.components.charts import (
    render_bar_chart,
    render_heatmap,
    render_line_chart,
    render_scatter,
)
from dashboard.components.lineage import build_lineage_figure
from dashboard.plots import (
    correlation_heatmap,
    feature_importance_bar,
    fitness_line_chart,
    mutual_info_bar,
    pareto_front_scatter,
)
from dashboard.tabs.data_profile import render_data_profile_tab


EXPERIMENTS_DIR = Path("experiments")


def load_json(path: Optional[str]) -> Dict:
    if not path:
        return {}
    file_path = Path(path)
    if not file_path.exists():
        return {}
    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_csv(path: Optional[str | Path]) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(file_path)
    except Exception:
        return pd.DataFrame()


def load_manifest(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def list_run_manifests() -> Dict[str, Path]:
    manifests: Dict[str, Path] = {}
    if not EXPERIMENTS_DIR.exists():
        return manifests
    for subdir in EXPERIMENTS_DIR.iterdir():
        if subdir.is_dir():
            run_manifest = subdir / "manifest.json"
            if run_manifest.exists():
                manifests[subdir.name] = run_manifest
    return dict(sorted(manifests.items(), reverse=True))


def resolve_path(path_str: Optional[str]) -> Optional[Path]:
    if not path_str:
        return None
    return Path(path_str)


def render_overview(history_df: pd.DataFrame, model_metrics: Dict[str, float], stability: str) -> None:
    st.subheader("Experiment Summary")
    if history_df.empty:
        st.info("History file not found. Run an experiment to populate the dashboard.")
        return

    best_fitness = history_df["best_fitness"].max()
    final_fitness = history_df.iloc[-1]["best_fitness"]
    col1, col2, col3 = st.columns(3)
    col1.metric("Best Fitness", f"{best_fitness:.4f}")
    col2.metric("Final Generation Fitness", f"{final_fitness:.4f}")
    col3.metric("Model Stability", stability)

    chart = fitness_line_chart(history_df)
    render_line_chart("Best Fitness over Generations", chart)

    if model_metrics:
        st.subheader("Model Performance Metrics")
        metric_items = sorted(model_metrics.items())
        num_cols = min(4, len(metric_items))
        cols = st.columns(num_cols if num_cols else 1)
        for idx, (name, value) in enumerate(metric_items):
            cols[idx % num_cols].metric(name.replace("_", " ").title(), f"{value:.4f}")


def render_insights(manifest: Dict) -> None:
    st.subheader("Insight Narrative")
    summary = manifest.get("insight_summary", "")
    if summary:
        st.write(summary)
    else:
        st.info("Insight summary not available. Rerun with insights enabled.")

    explanations = manifest.get("explanations", {})
    insights_dir_info = manifest.get("insights_dir", {})
    insights_dir = resolve_path(insights_dir_info.get("latest") or insights_dir_info.get("archive"))

    feature_importance_path = explanations.get("feature_importance")
    if feature_importance_path:
        fi_df = load_csv(feature_importance_path)
        if not fi_df.empty:
            render_bar_chart("Feature Importance", feature_importance_bar(fi_df))
        else:
            st.info("Feature importance data not available.")

    shap_summary_path = explanations.get("shap_summary")
    if shap_summary_path:
        shap_summary_file = Path(shap_summary_path)
        if shap_summary_file.exists() and shap_summary_file.suffix.lower() in {".png", ".jpg"}:
            st.image(str(shap_summary_file), caption="SHAP Summary Plot")
        elif shap_summary_file.exists():
            st.warning(shap_summary_file.read_text(encoding="utf-8"))

    permutation_path = explanations.get("permutation_importance")
    if permutation_path:
        perm_df = load_csv(permutation_path)
        if not perm_df.empty:
            st.subheader("Permutation Importance")
            st.dataframe(perm_df)

    pdp_dir = explanations.get("pdp_dir")
    if pdp_dir:
        pdp_path = Path(pdp_dir)
        images = sorted(pdp_path.glob("*.png"))
        if images:
            st.subheader("Partial Dependence Plots")
            for img in images:
                st.image(str(img), caption=img.name)

    lime_dir = explanations.get("lime_dir")
    if lime_dir:
        st.caption(f"LIME explanations stored in {lime_dir}")

    fitness_log_info = manifest.get("model_fitness_log", {})
    fitness_log_path = resolve_path(fitness_log_info.get("latest") or fitness_log_info.get("archive"))
    if fitness_log_path and fitness_log_path.exists():
        fitness_df = load_csv(fitness_log_path)
        if not fitness_df.empty:
            st.subheader("Accuracy–Latency Trade-offs")
            pareto_fig = pareto_front_scatter(fitness_df, "val_accuracy", "latency")
            render_scatter("Pareto Front: Accuracy vs Latency", pareto_fig)


def render_lineage_section(manifest: Dict) -> None:
    lineage_info = manifest.get("lineage", {})
    lineage_path = resolve_path(lineage_info.get("archive") or lineage_info.get("latest"))
    if not lineage_path or not lineage_path.exists():
        st.info("Lineage information not available yet.")
        return
    try:
        records = json.loads(lineage_path.read_text(encoding="utf-8"))
    except Exception:
        st.warning("Unable to read lineage file.")
        return
    fig = build_lineage_figure(records)
    if fig is None:
        st.info("Lineage visualisation will appear once enough generations are logged.")
    else:
        render_scatter("Architecture Lineage", fig)


def render_reports(manifest: Dict) -> None:
    st.subheader("Experiment Reports & Logs")
    report_info = manifest.get("report", {})
    html_path = resolve_path(report_info.get("latest_html") or report_info.get("archive_html"))
    pdf_path = resolve_path(report_info.get("latest_pdf") or report_info.get("archive_pdf"))

    if html_path and html_path.exists():
        st.download_button(
            "Download HTML Report",
            data=html_path.read_bytes(),
            file_name=html_path.name,
            mime="text/html",
        )
    if pdf_path and pdf_path.exists():
        st.download_button(
            "Download PDF Report",
            data=pdf_path.read_bytes(),
            file_name=pdf_path.name,
            mime="application/pdf",
        )

    log_path = resolve_path(manifest.get("log_file"))
    if log_path and log_path.exists():
        st.download_button(
            "Download Run Log",
            data=log_path.read_bytes(),
            file_name=log_path.name,
            mime="text/plain",
        )


def main() -> None:
    st.set_page_config(page_title="EvoMind Dashboard", layout="wide")
    st.title("EvoMind Evolution Dashboard")

    latest_manifest_path = EXPERIMENTS_DIR / "manifest.json"
    latest_manifest = load_manifest(latest_manifest_path)
    run_manifests = list_run_manifests()

    if not latest_manifest and not run_manifests:
        st.warning("No experiments found. Run `python main.py --data ...` first.")
        return

    default_run_id = latest_manifest.get("run_id") if latest_manifest else None
    run_options = list(run_manifests.keys())
    if default_run_id and default_run_id not in run_options:
        run_options.insert(0, default_run_id)

    selected_run_idx = 0
    if default_run_id and default_run_id in run_options:
        selected_run_idx = run_options.index(default_run_id)
    selected_run = st.sidebar.selectbox("Experiment Run", run_options, index=selected_run_idx if run_options else 0)

    if selected_run and selected_run in run_manifests:
        manifest = load_manifest(run_manifests[selected_run]) or latest_manifest
    else:
        manifest = latest_manifest

    if manifest is None:
        st.error("Unable to load manifest for the selected run.")
        return

    page = st.sidebar.radio("Sections", ["Overview", "Insights", "Data Profile", "Lineage", "Reports"])

    history_info = manifest.get("history", {})
    history_path = resolve_path(history_info.get("latest") or history_info.get("archive"))
    history_df = load_csv(history_path) if history_path else pd.DataFrame()

    if page == "Overview":
        render_overview(history_df, manifest.get("model_metrics", {}), manifest.get("model_stability", "Unknown"))
    elif page == "Insights":
        render_insights(manifest)
    elif page == "Data Profile":
        data_profile_path = resolve_path(manifest.get("dataset_profile", {}).get("health_json"))
        profile_json = load_json(str(data_profile_path)) if data_profile_path else {}
        mutual_info_df = load_csv(manifest.get("mutual_info"))
        anomalies = profile_json.get("anomalies", {}) if profile_json else {}
        time_series_info = load_json(manifest.get("time_series"))
        render_data_profile_tab(profile_json, mutual_info_df, anomalies, time_series_info)
    elif page == "Lineage":
        render_lineage_section(manifest)
    elif page == "Reports":
        render_reports(manifest)


if __name__ == "__main__":
    main()
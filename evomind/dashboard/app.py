"""
Streamlit dashboard surfacing EvoMind experiment insights.

The dashboard reads artifacts persisted by the EvoMind run (history, insights,
reports) and presents them through multiple navigation tabs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

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

DARK_THEME = """
<style>
body, .stApp { background-color: #121212; color: #f0f0f0; }
.sidebar .sidebar-content { background-color: #1f1f1f; }
.stMetric { background-color: #1f1f1f; border-radius: 8px; padding: 8px; }
.stButton>button { background-color: #4dd0e1; color: #111; }
.stDownloadButton>button { background-color: #4dd0e1; color: #111; }
.stSelectbox, .stRadio, .stCheckbox { color: #f0f0f0; }
</style>
"""

LIGHT_THEME = """
<style>
body, .stApp { background-color: #fafafa; color: #202124; }
.sidebar .sidebar-content { background-color: #f0f2f6; }
.stMetric { background-color: #ffffff; border-radius: 8px; padding: 8px; }
.stButton>button { background-color: #1976d2; color: #fff; }
.stDownloadButton>button { background-color: #1976d2; color: #fff; }
</style>
"""


def _estimate_fitness_change(
    base_fitness: float,
    base_population: float,
    base_mutation: float,
    base_crossover: float,
    population: float,
    mutation: float,
    crossover: float,
) -> Tuple[float, float]:
    baseline = base_fitness or 1.0
    population_factor = np.log1p(max(population - base_population, -base_population + 1)) * 0.05
    mutation_factor = -abs(mutation - base_mutation) * 0.2 + 0.05
    crossover_factor = (crossover - base_crossover) * 0.1
    estimated = baseline * (1 + population_factor + mutation_factor + crossover_factor)
    return estimated, estimated - baseline


def render_what_if_panel(history_df: pd.DataFrame, config: Dict, model_metrics: Dict[str, float]) -> None:
    with st.expander("🧪 What-if Simulator", expanded=False):
        if history_df.empty or not config:
            st.info("Simulation requires history and config metadata from a completed run.")
            return
        engine_cfg = config.get("engine", {})
        base_population = int(engine_cfg.get("population", 20))
        base_mutation = float(engine_cfg.get("mutation_rate", 0.3))
        base_crossover = float(engine_cfg.get("crossover_rate", 0.6))
        best_fitness = float(history_df["best_fitness"].max())

        population = st.slider("Population Size", 10, 200, base_population, step=5)
        mutation = st.slider("Mutation Rate", 0.05, 0.9, base_mutation, step=0.05)
        crossover = st.slider("Crossover Rate", 0.1, 0.9, base_crossover, step=0.05)

        with st.spinner("Estimating impact..."):
            estimated, delta = _estimate_fitness_change(
                best_fitness, base_population, base_mutation, base_crossover, population, mutation, crossover
            )
        st.metric("Projected Fitness", f"{estimated:.3f}", delta=f"{delta:+.3f}")
        st.caption(
            "Heuristic projection combining run history with configurable hyperparameters. "
            "Use for directional insight rather than absolute guarantees."
        )
        progress = min(max((estimated / (best_fitness + 1e-6)) * 0.5 + 0.5, 0), 1)
        st.progress(progress)


def render_worker_monitor(executor_stats: Dict[str, Any]) -> None:
    st.subheader("Real-time Worker Monitor")
    if not executor_stats:
        st.info("Worker telemetry unavailable for this run.")
        return
    backend = executor_stats.get("backend", "threads")
    history = executor_stats.get("history") or []
    worker_count = executor_stats.get("worker_count") or executor_stats.get("max_workers") or "auto"
    payloads = history[-1]["payloads"] if history else executor_stats.get("payloads", 0)
    cols = st.columns(3)
    cols[0].metric("Backend", backend.title())
    cols[1].metric("Workers", worker_count)
    cols[2].metric("Last Payloads", payloads)

    if history:
        df = pd.DataFrame(history)
        if "generation" not in df.columns:
            df["generation"] = range(len(df))
        duration_fig = px.line(df, x="generation", y="duration", markers=True, title="Evaluation Duration (s)")
        duration_fig.update_layout(showlegend=False)
        st.plotly_chart(duration_fig, use_container_width=True)

    if backend == "ray":
        available = executor_stats.get("available_resources")
        if available:
            st.caption("Ray cluster resources")
            st.json(available)


def render_feature_metric_network(fi_df: pd.DataFrame, metrics: Dict[str, float]) -> None:
    if fi_df.empty or not metrics:
        return
    st.subheader("Feature → Metric Influence Graph")
    feature_nodes = fi_df.head(8)
    metric_items = [
        (name, value)
        for name, value in metrics.items()
        if isinstance(value, (int, float, np.floating)) and not np.isnan(value)
    ]
    if not feature_nodes.empty and metric_items:
        metric_items = metric_items[:5]
        graph = nx.Graph()
        for feature in feature_nodes["feature"]:
            graph.add_node(feature, bipartite=0)
        for metric, value in metric_items:
            graph.add_node(metric, bipartite=1)
            for _, row in feature_nodes.iterrows():
                weight = abs(row["importance"]) * (abs(value) + 1e-6)
                graph.add_edge(row["feature"], metric, weight=weight)
        pos = nx.spring_layout(graph, seed=42)
        traces = []
        for edge in graph.edges(data=True):
            traces.append(
                go.Scatter(
                    x=[pos[edge[0]][0], pos[edge[1]][0]],
                    y=[pos[edge[0]][1], pos[edge[1]][1]],
                    mode="lines",
                    line=dict(width=max(1.0, edge[2]["weight"] * 5), color="#90caf9"),
                    hoverinfo="text",
                    text=f"{edge[0]} → {edge[1]}: {edge[2]['weight']:.3f}",
                )
            )
        node_trace = go.Scatter(
            x=[pos[node][0] for node in graph.nodes()],
            y=[pos[node][1] for node in graph.nodes()],
            mode="markers+text",
            text=list(graph.nodes()),
            textposition="top center",
            marker=dict(size=14, color="#4dd0e1"),
        )
        fig = go.Figure(traces + [node_trace])
        fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)


def trigger_success_animation(manifest: Dict) -> None:
    run_id = manifest.get("run_id")
    if not run_id:
        return
    if st.session_state.get("celebrated_run") == run_id:
        return
    st.session_state["celebrated_run"] = run_id
    st.success("Run loaded successfully! 🎉")
    st.balloons()


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


def render_correlation_network(corr_dict: Dict[str, Dict[str, float]], threshold: float) -> None:
    """Render a correlation network graph using NetworkX and Plotly."""

    if not corr_dict:
        st.info("Correlation data unavailable.")
        return

    corr_df = pd.DataFrame(corr_dict).fillna(0.0)
    if corr_df.empty:
        st.info("Correlation data unavailable.")
        return

    graph = nx.Graph()
    for feature in corr_df.columns:
        graph.add_node(feature)

    for i, feature_i in enumerate(corr_df.columns):
        for feature_j in corr_df.columns[i + 1 :]:
            corr_value = corr_df.at[feature_i, feature_j]
            if abs(corr_value) >= threshold and feature_i != feature_j:
                graph.add_edge(feature_i, feature_j, weight=corr_value)

    if graph.number_of_edges() == 0:
        st.info("No correlations exceed the selected threshold.")
        return

    pos = nx.spring_layout(graph, seed=42)
    edge_x: List[float] = []
    edge_y: List[float] = []
    edge_color: List[str] = []
    for edge in graph.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_color.append("rgba(67, 160, 71, 0.7)" if edge[2]["weight"] > 0 else "rgba(244, 67, 54, 0.7)")

    node_x = [pos[node][0] for node in graph.nodes()]
    node_y = [pos[node][1] for node in graph.nodes()]

    fig = go.Figure()
    for idx, edge in enumerate(graph.edges(data=True)):
        fig.add_trace(
            go.Scatter(
                x=edge_x[idx * 3 : idx * 3 + 3],
                y=edge_y[idx * 3 : idx * 3 + 3],
                mode="lines",
                line=dict(width=2),
                hoverinfo="text",
                text=f"{edge[0]} ↔ {edge[1]}: {edge[2]['weight']:.2f}",
                line_color=edge_color[idx],
            )
        )

    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=list(graph.nodes()),
            textposition="top center",
            marker=dict(size=14, color="#4dd0e1"),
            hoverinfo="text",
        )
    )

    fig.update_layout(
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)


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


def render_overview(
    history_df: pd.DataFrame,
    model_metrics: Dict[str, float],
    stability: str,
    config: Dict,
    executor_stats: Dict[str, Any],
) -> None:
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

    render_what_if_panel(history_df, config, model_metrics)
    render_worker_monitor(executor_stats)


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
    summary_path = resolve_path(manifest.get("dataset_profile", {}).get("summary_json"))
    summary_data = load_json(str(summary_path)) if summary_path else {}
    statistics = summary_data.get("statistics", {}) if isinstance(summary_data, dict) else {}

    metrics = manifest.get("model_metrics", {})

    feature_importance_path = explanations.get("feature_importance")
    fi_df = load_csv(feature_importance_path) if feature_importance_path else pd.DataFrame()
    if not fi_df.empty:
        render_bar_chart("Feature Importance", feature_importance_bar(fi_df))
        with st.expander("Raw feature importance values", expanded=False):
            st.dataframe(fi_df)
    elif feature_importance_path:
        st.info("Feature importance data not available.")
    render_feature_metric_network(fi_df, metrics)

    shap_summary_path = explanations.get("shap_summary")
    if shap_summary_path:
        shap_summary_file = Path(shap_summary_path)
        if shap_summary_file.exists():
            image_suffixes = {".png", ".jpg", ".jpeg"}
            if shap_summary_file.suffix.lower() in image_suffixes:
                try:
                    st.image(str(shap_summary_file), caption="SHAP Summary Plot")
                except Exception:
                    st.warning(shap_summary_file.read_text(encoding="utf-8"))
            else:
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
        lime_path = Path(lime_dir)
        html_files = sorted(lime_path.glob("*.html"))
        if html_files:
            st.subheader("Interactive LIME Explorer")
            selected = st.selectbox("Sample", [file.name for file in html_files])
            html_path = lime_path / selected
            components.html(html_path.read_text(encoding="utf-8"), height=420, scrolling=True)
        else:
            st.caption(f"LIME explanations stored in {lime_dir}")

    fitness_log_info = manifest.get("model_fitness_log", {})
    fitness_log_path = resolve_path(fitness_log_info.get("latest") or fitness_log_info.get("archive"))
    if fitness_log_path and fitness_log_path.exists():
        fitness_df = load_csv(fitness_log_path)
        if not fitness_df.empty:
            st.subheader("Accuracy–Latency Trade-offs")
            pareto_fig = pareto_front_scatter(fitness_df, "val_accuracy", "latency")
            render_scatter("Pareto Front: Accuracy vs Latency", pareto_fig)

    if not fi_df.empty and statistics:
        st.subheader("What-if Analysis")
        numeric_stats = {
            col: stats
            for col, stats in statistics.items()
            if isinstance(stats, dict) and isinstance(stats.get("mean"), (int, float, np.floating))
        }
        top_features = [f for f in fi_df["feature"].tolist() if f in numeric_stats]
        if top_features:
            total_importance = fi_df.set_index("feature").loc[top_features]["importance"].abs().sum() or 1.0
            for feature in top_features[:3]:
                stats = numeric_stats[feature]
                min_val = float(stats.get("min", stats.get("mean", 0.0)))
                max_val = float(stats.get("max", stats.get("mean", 0.0)))
                if min_val == max_val:
                    max_val += 1.0
                mean_val = float(stats.get("mean", (min_val + max_val) / 2))
                step = max((max_val - min_val) / 100.0, 1e-6)
                selected = st.slider(
                    f"{feature} value",
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    step=step,
                )
                weight = float(fi_df.loc[fi_df["feature"] == feature, "importance"].iloc[0])
                delta = selected - mean_val
                estimated = delta * weight / total_importance
                st.metric(
                    label=f"{feature} adjustment",
                    value=f"{selected:.2f}",
                    delta=f"{estimated:+.3f} estimated impact",
                )
            st.caption("Estimates derive from normalised feature importance and are intended for directional insight, not exact predictions.")
        else:
            st.info("What-if analysis requires numeric feature statistics; none were detected.")


def render_lineage_section(manifest: Dict, history_df: pd.DataFrame) -> None:
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

    if not history_df.empty:
        st.subheader("Lineage Replay")
        max_generation = int(history_df["generation"].max())
        min_generation = int(history_df["generation"].min())
        if max_generation == min_generation:
            selected = max_generation
            st.info("Only one generation recorded; replay slider disabled.")
        else:
            selected = st.slider("Generation", min_generation, max_generation, value=max_generation)
        current_row = history_df.loc[history_df["generation"] == selected]
        if not current_row.empty:
            fitness_value = float(current_row["best_fitness"].iloc[0])
            st.metric("Fitness at Generation", f"{fitness_value:.4f}", delta=None)
        timeline_fig = px.line(history_df, x="generation", y="best_fitness", markers=True)
        timeline_fig.add_vline(x=selected, line_dash="dash", line_color="#ef5350")
        timeline_fig.update_layout(showlegend=False, title="Fitness Timeline Replay")
        st.plotly_chart(timeline_fig, use_container_width=True)


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
    theme_choice = st.sidebar.radio("Theme", ["Dark", "Light"], index=0)
    st.markdown(DARK_THEME if theme_choice == "Dark" else LIGHT_THEME, unsafe_allow_html=True)
    st.title("EvoMind Evolution Dashboard")

    with st.spinner("Loading experiment manifests..."):
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

    trigger_success_animation(manifest)

    with st.sidebar.expander("Run Details", expanded=False):
        st.write(f"**Run ID:** {manifest.get('run_id')}")
        st.write(f"**Stability:** {manifest.get('model_stability', 'Unknown')}")
        metrics_preview = manifest.get("model_metrics", {}) or {}
        if metrics_preview:
            st.write("**Top Metrics:**")
            for name, value in list(metrics_preview.items())[:3]:
                if isinstance(value, (int, float, np.floating)):
                    st.write(f"- {name}: {value:.4f}")
                else:
                    st.write(f"- {name}: {value}")
        sources = manifest.get("data_sources") or []
        if sources:
            st.write("**Data Sources:**")
            for src in sources[:3]:
                st.write(f"- {src}")
            if len(sources) > 3:
                remaining = len(sources) - 3
                st.write(f"... and {remaining} more")
        model_card_path = manifest.get("model_card")
        if model_card_path:
            st.markdown(f"[Open Model Card]({model_card_path})")
        report_info = manifest.get("report", {})
        latest_html = report_info.get("latest_html") or report_info.get("archive_html")
        if latest_html:
            st.markdown(f"[Latest Report]({latest_html})")

    page = st.sidebar.radio("Sections", ["Overview", "Insights", "Data Profile", "Lineage", "Reports"])
    config_blob = manifest.get("config") or {}
    executor_stats = manifest.get("executor") or {}

    history_info = manifest.get("history", {})
    history_path = resolve_path(history_info.get("latest") or history_info.get("archive"))
    history_df = load_csv(history_path) if history_path else pd.DataFrame()
    if not history_df.empty and "generation" not in history_df.columns:
        history_df = history_df.reset_index(drop=True)
        history_df.insert(0, "generation", history_df.get("generation", range(len(history_df))))

    if page == "Overview":
        render_overview(
            history_df,
            manifest.get("model_metrics", {}),
            manifest.get("model_stability", "Unknown"),
            config_blob,
            executor_stats,
        )
    elif page == "Insights":
        render_insights(manifest)
    elif page == "Data Profile":
        data_profile_path = resolve_path(manifest.get("dataset_profile", {}).get("health_json"))
        profile_json = load_json(str(data_profile_path)) if data_profile_path else {}
        mutual_info_df = load_csv(manifest.get("mutual_info"))
        anomalies = profile_json.get("anomalies", {}) if profile_json else {}
        time_series_info = load_json(manifest.get("time_series"))
        correlation_data = profile_json.get("correlation", {}) if profile_json else {}
        render_data_profile_tab(
            profile_json,
            mutual_info_df,
            anomalies,
            time_series_info,
            correlation_network=lambda threshold: render_correlation_network(correlation_data, threshold),
        )
    elif page == "Lineage":
        render_lineage_section(manifest, history_df)
    elif page == "Reports":
        render_reports(manifest)


if __name__ == "__main__":
    main()

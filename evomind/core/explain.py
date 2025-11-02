"""
Explainability helpers wrapping SHAP and LIME integrations.

The utilities attempt to compute post-hoc explanations for trained models while
falling back to lightweight heuristics when optional dependencies are missing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
import torch


def _prepare_tensor_batch(model: torch.nn.Module, data: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        tensor = torch.as_tensor(data, dtype=torch.float32)
        preds = model(tensor).detach().cpu().numpy()
    return preds


def _fallback_importance(model: torch.nn.Module, feature_names: List[str]) -> Dict[str, float]:
    """
    Heuristic feature importance derived from the first linear layer weights.

    This is used when SHAP is unavailable to ensure the dashboard still has a
    notion of relative feature influence.
    """

    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            weights = module.weight.detach().abs().mean(dim=0).cpu().numpy()
            return dict(zip(feature_names, weights.tolist()))
    return {name: 0.0 for name in feature_names}


def generate_explanations(
    model: torch.nn.Module,
    X_sample: np.ndarray,
    feature_names: Iterable[str],
    output_dir: Path,
) -> Dict[str, Path]:
    """
    Produce SHAP and LIME based explanation artifacts for the supplied model.

    Parameters
    ----------
    model : nn.Module
        Trained torch model.
    X_sample : np.ndarray
        Sampled input data used to compute explanations.
    feature_names : Iterable[str]
        Ordered feature names aligned with columns in X_sample.
    output_dir : Path
        Location where generated artifacts should be stored.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    feature_names_list = list(feature_names)
    shap_path = output_dir / "shap_summary.png"
    feature_csv = output_dir / "feature_importance.csv"
    lime_dir = output_dir / "lime_examples"
    lime_dir.mkdir(exist_ok=True)
    permutation_csv = output_dir / "permutation_importance.csv"
    pdp_dir = output_dir / "partial_dependence"
    pdp_dir.mkdir(exist_ok=True)

    explanation_paths: Dict[str, Path] = {
        "shap_summary": shap_path,
        "feature_importance": feature_csv,
        "lime_dir": lime_dir,
        "permutation_importance": permutation_csv,
        "pdp_dir": pdp_dir,
    }

    # Attempt SHAP computation.
    shap_values = None
    shap_array = None
    try:  # pragma: no cover - shap integration requires optional dependency.
        import shap

        model.eval()
        sample_size = min(200, X_sample.shape[0])
        sample = X_sample[:sample_size]
        background = shap.kmeans(sample, min(20, sample_size))

        def predict(data: np.ndarray) -> np.ndarray:
            return _prepare_tensor_batch(model, data)

        explainer = shap.KernelExplainer(predict, background)
        shap_values = explainer.shap_values(sample, nsamples=100)
        shap_array = np.array(shap_values[0]) if isinstance(shap_values, list) else np.array(shap_values)
        shap.summary_plot(shap_array, sample, feature_names=feature_names_list, show=False)
        import matplotlib.pyplot as plt

        plt.tight_layout()
        plt.savefig(shap_path, dpi=150)
        plt.close()
        mean_abs = np.abs(shap_array).mean(axis=0)
        pd.DataFrame({"feature": feature_names_list, "importance": mean_abs}).to_csv(feature_csv, index=False)

        # Dependence plots for top three features.
        top_indices = np.argsort(mean_abs)[::-1][: min(3, len(mean_abs))]
        for idx in top_indices:
            feature = feature_names_list[idx]
            shap.dependence_plot(
                feature,
                shap_array,
                sample,
                feature_names=feature_names_list,
                show=False,
            )
            import matplotlib.pyplot as plt  # noqa: WPS442

            plt.tight_layout()
            plt.savefig(pdp_dir / f"shap_dependence_{feature}.png", dpi=120)
            plt.close()
    except Exception:
        # Fallback importance written when SHAP is unavailable.
        importance = _fallback_importance(model, feature_names_list)
        pd.DataFrame({"feature": list(importance.keys()), "importance": list(importance.values())}).to_csv(
            feature_csv, index=False
        )
        shap_path.write_text(
            "SHAP summary plot could not be generated because the optional dependency is missing.",
            encoding="utf-8",
        )

    # Attempt a lightweight LIME explanation for the first few samples.
    try:  # pragma: no cover - lime integration optional.
        from lime.lime_tabular import LimeTabularExplainer

        model.eval()
        sample_size = min(100, X_sample.shape[0])
        sample = X_sample[:sample_size]
        explainer = LimeTabularExplainer(
            sample,
            feature_names=feature_names_list,
            class_names=["prediction"],
            discretize_continuous=True,
        )
        for idx in range(min(5, sample_size)):
            exp = explainer.explain_instance(
                sample[idx],
                lambda x: _prepare_tensor_batch(model, x),
                num_features=min(10, len(feature_names_list)),
            )
            html_path = lime_dir / f"lime_{idx}.html"
            exp.save_to_file(str(html_path))
    except Exception:
        placeholder = lime_dir / "README.txt"
        placeholder.write_text(
            "LIME explanations could not be generated because the optional dependency is missing.",
            encoding="utf-8",
        )

    # Permutation importance using sklearn.
    perm_scores = []
    try:
        model.eval()
        baseline_preds = _prepare_tensor_batch(model, X_sample)
        for idx, feature in enumerate(feature_names_list):
            X_permuted = X_sample.copy()
            np.random.shuffle(X_permuted[:, idx])
            permuted_preds = _prepare_tensor_batch(model, X_permuted)
            importance = float(np.mean(np.abs(baseline_preds - permuted_preds)))
            perm_scores.append((feature, importance))
        pd.DataFrame(
            {"feature": [name for name, _ in perm_scores], "importance": [score for _, score in perm_scores]}
        ).to_csv(permutation_csv, index=False)
    except Exception:
        permutation_csv.write_text(
            "Permutation importance could not be computed due to estimator incompatibility.", encoding="utf-8"
        )

    # Partial dependence plots for top two features.
    try:
        import matplotlib.pyplot as plt  # noqa: WPS433

        if shap_array is not None:
            mean_abs = np.abs(shap_array).mean(axis=0)
            order = np.argsort(mean_abs)[::-1]
            candidate_features = [feature_names_list[i] for i in order]
        elif perm_scores:
            candidate_features = [name for name, _ in sorted(perm_scores, key=lambda item: item[1], reverse=True)]
        else:
            candidate_features = feature_names_list

        top_features = candidate_features[:2]
        for feature in top_features:
            idx = feature_names_list.index(feature)
            values = np.linspace(
                np.percentile(X_sample[:, idx], 5),
                np.percentile(X_sample[:, idx], 95),
                num=20,
            )
            pdp_values = []
            for value in values:
                X_copy = X_sample.copy()
                X_copy[:, idx] = value
                preds = _prepare_tensor_batch(model, X_copy)
                pdp_values.append(float(np.mean(preds)))
            plt.figure()
            plt.plot(values, pdp_values, marker="o")
            plt.title(f"Partial Dependence: {feature}")
            plt.xlabel(feature)
            plt.ylabel("Prediction")
            plt.tight_layout()
            plt.savefig(pdp_dir / f"pdp_{feature}.png", dpi=120)
            plt.close()
    except Exception:
        pass

    return explanation_paths

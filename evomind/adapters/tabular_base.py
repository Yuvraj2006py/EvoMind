"""
Shared utilities for tabular EvoMind adapters.

The refreshed :class:`TabularAdapter` implements the new BaseAdapter contract by
handling datasource loading, schema aware preprocessing, and convenience hooks
for generating reports.  Concrete adapters only need to override :meth:`train`
and :meth:`evaluate` to provide task specific baselines.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from evomind.adapters.base_adapter import BaseAdapter
from evomind.adapters.data_utils import create_one_hot_encoder, load_dataframe
from evomind.exceptions import EvoMindAdapterError


class TabularAdapter(BaseAdapter):
    """Base class for adapters operating on structured/tabular datasets."""

    def __init__(
        self,
        *,
        task_type: str,
        schema: Optional[Dict[str, Any]] = None,
        default_target: str = "target",
        config: Optional[Dict[str, Any]] = None,
        data: Any | None = None,
    ) -> None:
        super().__init__(schema=schema, data=data, target_col=default_target, config=config)
        self.task_type = task_type
        self.default_target = default_target
        self.scaler = StandardScaler()
        self.encoder = create_one_hot_encoder()
        self.label_encoder: LabelEncoder | None = None
        self._latest_frame: pd.DataFrame | None = None

    # ------------------------------------------------------------------ IO API
    def load_data(self) -> pd.DataFrame:
        if isinstance(self.data_source, pd.DataFrame):
            return self.data_source.copy()
        if isinstance(self.data_source, (str, Path)):
            path = Path(self.data_source)
            if not path.exists():
                raise EvoMindAdapterError(
                    f"Datasource does not exist: {path}",
                    context={"adapter": self.__class__.__name__},
                )
            return self._load_dataframe(path)
        raise EvoMindAdapterError(
            "Adapter requires a data source (path or DataFrame).",
            context={"adapter": self.__class__.__name__},
        )

    def _load_dataframe(self, path: Path) -> pd.DataFrame:
        return load_dataframe(path)

    # --------------------------------------------------------------- Preprocess
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        target = self._resolve_target_column(df)
        target_series = self._prepare_target(df[target])
        valid_mask = target_series.notna()
        if valid_mask.sum() < 2:
            raise EvoMindAdapterError(
                "Target column must contain at least two valid rows after preprocessing.",
                context={"adapter": self.__class__.__name__, "target": target},
            )
        if not valid_mask.all():
            df = df.loc[valid_mask].copy()
            target_series = target_series.loc[valid_mask]
        X = self._sanitize_features(df.drop(columns=[target], errors="ignore"))

        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = [col for col in X.columns if col not in numeric_cols]

        feature_blocks: List[np.ndarray] = []
        feature_names: List[str] = []

        if numeric_cols:
            num_matrix = self.scaler.fit_transform(X[numeric_cols])
            feature_blocks.append(num_matrix)
            feature_names.extend(numeric_cols)
        if cat_cols:
            encoded = self.encoder.fit_transform(X[cat_cols])
            feature_blocks.append(encoded)
            feature_names.extend(list(self.encoder.get_feature_names_out(cat_cols)))

        if not feature_blocks:
            raise EvoMindAdapterError(
                "No usable features detected after preprocessing.",
                context={"adapter": self.__class__.__name__},
            )

        matrix = feature_blocks[0] if len(feature_blocks) == 1 else np.hstack(feature_blocks)
        processed = pd.DataFrame(matrix, columns=feature_names, index=df.index)
        processed[target] = target_series.values

        self.feature_names_ = feature_names
        self._latest_frame = processed.copy()
        return processed

    # ---------------------------------------------------------------- Baseline
    def train(self, X, y) -> Any:  # pragma: no cover - implemented by subclasses
        raise NotImplementedError("TabularAdapter subclasses must implement train().")

    def evaluate(self, model, X_test, y_test) -> Dict[str, float]:  # pragma: no cover
        raise NotImplementedError("TabularAdapter subclasses must implement evaluate().")

    # ------------------------------------------------------------ Report hooks
    def summarize_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        frame = self._latest_frame if self._latest_frame is not None else df
        return super().summarize_data(frame)

    def generate_profile(self, df: pd.DataFrame, output_path: Path) -> Path:
        frame = self._latest_frame if self._latest_frame is not None else df
        return super().generate_profile(frame, output_path)

    # ---------------------------------------------------------------- Helpers
    def _resolve_target_column(self, df: pd.DataFrame) -> str:
        if self.target_column and self.target_column in df.columns:
            return self.target_column
        target = self.schema.get("target")
        if isinstance(target, str) and target in df.columns:
            self.target_column = target
            return target
        fallback = df.columns[-1]
        self.target_column = fallback
        return fallback

    def _sanitize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if numeric_cols.any():
            df[numeric_cols] = df[numeric_cols].apply(lambda col: col.fillna(col.mean()))
            drop_numeric = [col for col in numeric_cols if df[col].isna().all()]
            if drop_numeric:
                df = df.drop(columns=drop_numeric)
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns
        for column in categorical_cols:
            cat_series = df[column].astype(object).copy()
            mask = cat_series.isna()
            if mask.any():
                cat_series.loc[mask] = "missing"
            df[column] = cat_series.astype(str)
        return df

    def _prepare_target(self, series: pd.Series) -> pd.Series:
        if self.task_type == "classification":
            encoded = series.astype(str)
            self.label_encoder = LabelEncoder()
            return pd.Series(self.label_encoder.fit_transform(encoded), index=series.index)
        numeric = self._coerce_numeric_series(series)
        if numeric.notna().sum() < 2:
            raise EvoMindAdapterError(
                "Target column must contain numeric values for regression tasks.",
                context={"adapter": self.__class__.__name__, "target": self.target_column},
            )
        return numeric.astype(float)

    @staticmethod
    def _coerce_numeric_series(series: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(series):
            return pd.to_numeric(series, errors="coerce")
        as_str = series.astype(str).str.strip()
        cleaned = (
            as_str.str.replace(",", "", regex=False)
            .str.replace(r"[^\d\.\-eE]", "", regex=True)
            .replace({"": pd.NA, ".": pd.NA, "-": pd.NA})
        )
        numeric = pd.to_numeric(cleaned, errors="coerce")
        return numeric

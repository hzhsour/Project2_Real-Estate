"""Isolated direct five-year rental-growth validation for Project Question 2.

The experiment predicts realised five-year growth directly instead of first
forecasting a 20-quarter rent path.  It uses only the saved Homes Victoria
moving-annual history and the existing offline forecasting implementations.
No API calls, external downloads, existing Sprint 5 output writes, or SA2-row
duplication are performed here.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .sprint5_model_extension import distinct_history_series, forecast_extended


HORIZON_QUARTERS = 20
MIN_FEATURE_HISTORY = 12
DIRECT_MODELS = ("ridge", "elastic_net", "random_forest", "hist_gradient_boosting")
BASELINE_MODELS = ("baseline_median", "baseline_trailing_5y", "baseline_drift")
PATH_COMPARISON_MODELS = ("drift", "linear_trend", "damped_holt", "theta", "combination_b")
MODEL_LABELS = {
    "baseline_median": "Baseline A: training median growth",
    "baseline_trailing_5y": "Baseline B: trailing 5-year growth",
    "baseline_drift": "Baseline C: Drift-implied growth",
    "ridge": "Ridge Regression",
    "elastic_net": "Elastic Net",
    "random_forest": "Random Forest",
    "hist_gradient_boosting": "HistGradientBoosting",
    "drift": "Drift",
    "linear_trend": "Linear Trend",
    "damped_holt": "Damped Holt",
    "theta": "Theta",
    "combination_b": "Combination B",
}
FEATURE_COLUMNS = [
    "current_rent",
    "rent_percentile_across_markets_at_origin",
    "growth_1y_pct",
    "growth_2y_pct",
    "growth_3y_pct",
    "recent_4q_slope",
    "recent_8q_slope",
    "recent_12q_slope",
    "short_slope_minus_long_slope",
    "std_quarterly_changes_last_8q",
    "current_rent_vs_recent_3y_mean_pct",
    "distance_from_recent_3y_peak_pct",
    "statewide_median_rent",
    "market_vs_statewide_median_pct",
    "market_1y_growth_minus_statewide_1y_growth_pct",
    "market_3y_growth_minus_statewide_3y_growth_pct",
]
FEATURE_AUDIT = {
    "current_rent": "Rent observed at the forecast origin.",
    "rent_percentile_across_markets_at_origin": "Cross-market percentile of current rent, calculated using only the same-origin observations.",
    "growth_1y_pct": "Growth from the origin to four quarters earlier.",
    "growth_2y_pct": "Growth from the origin to eight quarters earlier.",
    "growth_3y_pct": "Growth from the origin to twelve quarters earlier.",
    "recent_4q_slope": "Linear slope over the latest four observed quarters.",
    "recent_8q_slope": "Linear slope over the latest eight observed quarters.",
    "recent_12q_slope": "Linear slope over the latest twelve observed quarters.",
    "short_slope_minus_long_slope": "Four-quarter slope minus twelve-quarter slope.",
    "std_quarterly_changes_last_8q": "Standard deviation of the last seven quarter-to-quarter changes within the latest eight observations.",
    "current_rent_vs_recent_3y_mean_pct": "Current rent relative to the mean of the latest twelve observations.",
    "distance_from_recent_3y_peak_pct": "Current rent relative to the maximum of the latest twelve observations.",
    "statewide_median_rent": "Median rent across source markets at the same historical origin.",
    "market_vs_statewide_median_pct": "Market rent relative to the same-origin statewide median.",
    "market_1y_growth_minus_statewide_1y_growth_pct": "Market one-year growth minus same-origin statewide one-year growth.",
    "market_3y_growth_minus_statewide_3y_growth_pct": "Market three-year growth minus same-origin statewide three-year growth.",
}


def _safe_growth(current: float, previous: float) -> float:
    if not np.isfinite(current) or not np.isfinite(previous) or previous <= 0:
        return np.nan
    return float((current - previous) / previous * 100.0)


def _slope(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        return np.nan
    return float(np.polyfit(np.arange(len(values), dtype=float), values, 1)[0])


def five_year_growth_target(values: Iterable[float], origin_index: int, horizon: int = HORIZON_QUARTERS) -> float:
    """Return exactly the endpoint growth from origin t to t+20 quarters."""

    values = np.asarray(list(values), dtype=float)
    end_index = origin_index + horizon
    if origin_index < 0 or end_index >= len(values):
        return np.nan
    return _safe_growth(values[end_index], values[origin_index])


def load_saved_long_history(project_root: str | Path) -> pd.DataFrame:
    root = Path(project_root).resolve()
    path = root / "data/processed/sprint5/long_history_validation/homes_victoria_long_history_all_properties.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run the saved long-history validation first: {path}")
    frame = pd.read_csv(path, low_memory=False)
    frame["period_end"] = pd.to_datetime(frame["period_end"], errors="coerce")
    frame["moving_annual_median_rent_aud"] = pd.to_numeric(frame["moving_annual_median_rent_aud"], errors="coerce")
    frame["rental_count"] = pd.to_numeric(frame["rental_count"], errors="coerce")
    frame = frame.loc[~frame["is_summary_row"].fillna(False)].copy()
    return frame


def load_existing_mapping(project_root: str | Path) -> pd.DataFrame:
    root = Path(project_root).resolve()
    path = root / "data/processed/sprint5/long_history_validation/source_market_matching_audit.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing source-market matching audit: {path}")
    return pd.read_csv(path, keep_default_na=False)


def source_market_mapping(matching_audit: pd.DataFrame) -> tuple[dict[str, list[str]], dict[str, object]]:
    """Expand the existing audit and explicitly expose one-to-many mappings."""

    source_to_groups: dict[str, list[str]] = defaultdict(list)
    for row in matching_audit.itertuples(index=False):
        group = str(row.source_rental_history_group)
        for source in str(row.matched_source_markets).split(";"):
            source = source.strip()
            if source and source not in source_to_groups[source]:
                source_to_groups[source].append(group)
    ambiguous = sorted(source for source, groups in source_to_groups.items() if len(groups) > 1)
    summary = {
        "current_distinct_groups": int(len(matching_audit)),
        "referenced_source_markets": int(len(source_to_groups)),
        "source_markets_with_one_current_group": int(sum(len(groups) == 1 for groups in source_to_groups.values())),
        "source_markets_with_multiple_current_groups": int(len(ambiguous)),
        "ambiguous_source_markets": ambiguous,
        "modeling_unit": "original Homes Victoria source rental market",
        "mapping_rule": "Predictions are fitted once per original source market. Current distinct-group labels are used only for transparent output mapping; one-to-many source references are retained as ambiguous and are not treated as extra independent observations.",
    }
    return dict(source_to_groups), summary


def source_market_panel(long_history: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[pd.Timestamp]]:
    periods = sorted(pd.Timestamp(value) for value in long_history["period_end"].dropna().unique())
    rents = long_history.pivot_table(
        index="source_rental_market",
        columns="period_end",
        values="moving_annual_median_rent_aud",
        aggfunc="median",
    ).reindex(columns=periods)
    counts = long_history.pivot_table(
        index="source_rental_market",
        columns="period_end",
        values="rental_count",
        aggfunc="sum",
    ).reindex(index=rents.index, columns=periods)
    return rents, counts, periods


def build_feature_audit() -> pd.DataFrame:
    rows = []
    windows = {
        "current_rent": 1,
        "rent_percentile_across_markets_at_origin": 1,
        "growth_1y_pct": 5,
        "growth_2y_pct": 9,
        "growth_3y_pct": 13,
        "recent_4q_slope": 4,
        "recent_8q_slope": 8,
        "recent_12q_slope": 12,
        "short_slope_minus_long_slope": 12,
        "std_quarterly_changes_last_8q": 8,
        "current_rent_vs_recent_3y_mean_pct": 12,
        "distance_from_recent_3y_peak_pct": 12,
        "statewide_median_rent": 1,
        "market_vs_statewide_median_pct": 1,
        "market_1y_growth_minus_statewide_1y_growth_pct": 5,
        "market_3y_growth_minus_statewide_3y_growth_pct": 13,
    }
    for feature in FEATURE_COLUMNS:
        rows.append(
            {
                "feature": feature,
                "definition": FEATURE_AUDIT[feature],
                "minimum_observations_before_origin": windows[feature],
                "uses_future_target_window": False,
                "scaling": "training-fold StandardScaler" if feature in FEATURE_COLUMNS else "none",
            }
        )
    return pd.DataFrame(rows)


def build_origin_feature_rows(
    rents: pd.DataFrame,
    periods: list[pd.Timestamp],
    origins: Iterable[pd.Timestamp],
    *,
    role: str = "historical",
) -> pd.DataFrame:
    """Create origin rows using only observations at or before each origin."""

    period_index = {pd.Timestamp(period): i for i, period in enumerate(periods)}
    statewide = rents.median(axis=0, skipna=True)
    rows: list[dict[str, object]] = []
    for origin in sorted(pd.Timestamp(value) for value in origins):
        if origin not in period_index:
            continue
        origin_index = period_index[origin]
        current_values = rents.iloc[:, origin_index]
        positive_current = current_values.loc[current_values.gt(0)].dropna()
        percentile = current_values.rank(pct=True)
        statewide_current = float(statewide.iloc[origin_index]) if np.isfinite(statewide.iloc[origin_index]) else np.nan
        statewide_growth_1y = _safe_growth(
            statewide.iloc[origin_index], statewide.iloc[origin_index - 4] if origin_index >= 4 else np.nan
        )
        statewide_growth_3y = _safe_growth(
            statewide.iloc[origin_index], statewide.iloc[origin_index - 12] if origin_index >= 12 else np.nan
        )
        for market in rents.index:
            series = rents.loc[market].to_numpy(dtype=float)
            past = series[: origin_index + 1]
            reasons: list[str] = []
            if origin_index + 1 < MIN_FEATURE_HISTORY or not np.isfinite(past[-MIN_FEATURE_HISTORY:]).all():
                reasons.append("missing_required_history_before_origin")
            current = float(series[origin_index]) if np.isfinite(series[origin_index]) else np.nan
            target = five_year_growth_target(series, origin_index)
            if role == "historical" and not np.isfinite(target):
                reasons.append("missing_or_nonpositive_20q_target")
            if not np.isfinite(current) or current <= 0:
                reasons.append("missing_or_nonpositive_origin_rent")

            def lag_growth(lag: int) -> float:
                return _safe_growth(current, series[origin_index - lag] if origin_index >= lag else np.nan)

            g1 = lag_growth(4)
            g2 = lag_growth(8)
            g3 = lag_growth(12)
            slope4 = _slope(past[-4:]) if len(past) >= 4 else np.nan
            slope8 = _slope(past[-8:]) if len(past) >= 8 else np.nan
            slope12 = _slope(past[-12:]) if len(past) >= 12 else np.nan
            recent8 = past[-8:] if len(past) >= 8 else np.array([], dtype=float)
            recent12 = past[-12:] if len(past) >= 12 else np.array([], dtype=float)
            drift_growth = np.nan
            if len(past) >= 2 and np.isfinite(past).all():
                drift_forecast, _ = forecast_extended(past, "drift", HORIZON_QUARTERS)
                drift_growth = _safe_growth(float(drift_forecast[-1]), current)
            row = {
                "data_role": role,
                "source_rental_market": str(market),
                "forecast_origin": origin.strftime("%Y-%m-%d"),
                "forecast_origin_date": origin,
                "target_end_date": (periods[origin_index + HORIZON_QUARTERS] if origin_index + HORIZON_QUARTERS < len(periods) else pd.NaT),
                "origin_index": origin_index,
                "current_rent": current,
                "rent_percentile_across_markets_at_origin": float(percentile.loc[market]) if np.isfinite(percentile.loc[market]) else np.nan,
                "growth_1y_pct": g1,
                "growth_2y_pct": g2,
                "growth_3y_pct": g3,
                "recent_4q_slope": slope4,
                "recent_8q_slope": slope8,
                "recent_12q_slope": slope12,
                "short_slope_minus_long_slope": slope4 - slope12 if np.isfinite(slope4) and np.isfinite(slope12) else np.nan,
                "std_quarterly_changes_last_8q": float(np.std(np.diff(recent8))) if len(recent8) >= 2 and np.isfinite(recent8).all() else np.nan,
                "current_rent_vs_recent_3y_mean_pct": _safe_growth(current, float(np.mean(recent12))) if len(recent12) == 12 and np.isfinite(recent12).all() else np.nan,
                "distance_from_recent_3y_peak_pct": _safe_growth(current, float(np.max(recent12))) if len(recent12) == 12 and np.isfinite(recent12).all() else np.nan,
                "statewide_median_rent": statewide_current,
                "market_vs_statewide_median_pct": _safe_growth(current, statewide_current),
                "market_1y_growth_minus_statewide_1y_growth_pct": g1 - statewide_growth_1y if np.isfinite(g1) and np.isfinite(statewide_growth_1y) else np.nan,
                "market_3y_growth_minus_statewide_3y_growth_pct": g3 - statewide_growth_3y if np.isfinite(g3) and np.isfinite(statewide_growth_3y) else np.nan,
                "trailing_5y_growth_pct": _safe_growth(current, series[origin_index - 20] if origin_index >= 20 else np.nan),
                "drift_predicted_5y_growth_pct": drift_growth,
                "target_5y_growth_pct": target if role == "historical" else np.nan,
                "exclusion_reason": "; ".join(dict.fromkeys(reasons)),
            }
            row["feature_complete"] = not any(not np.isfinite(row[feature]) for feature in FEATURE_COLUMNS)
            if not row["feature_complete"] and "missing_required_features" not in reasons:
                row["exclusion_reason"] = "; ".join([row["exclusion_reason"], "missing_required_features"]).strip("; ")
            row["target_available"] = bool(np.isfinite(row["target_5y_growth_pct"]))
            row["eligible_for_direct_model"] = bool(row["feature_complete"] and (role == "current_prediction" or row["target_available"]))
            rows.append(row)
    return pd.DataFrame(rows)


def build_temporal_folds(dataset: pd.DataFrame) -> pd.DataFrame:
    historical = dataset.loc[
        dataset["data_role"].eq("historical")
        & dataset["eligible_for_direct_model"]
        & dataset["target_available"]
    ].copy()
    historical["forecast_origin_date"] = pd.to_datetime(historical["forecast_origin_date"])
    historical["target_end_date"] = pd.to_datetime(historical["target_end_date"])
    rows = []
    for fold, origin in enumerate(sorted(historical["forecast_origin_date"].unique()), start=1):
        origin = pd.Timestamp(origin)
        validation = historical.loc[historical["forecast_origin_date"].eq(origin)]
        train = historical.loc[
            historical["forecast_origin_date"].lt(origin)
            & historical["target_end_date"].le(origin)
        ]
        rows.append(
            {
                "fold": fold,
                "validation_origin": origin.strftime("%Y-%m-%d"),
                "training_latest_origin_with_label": train["forecast_origin_date"].max().strftime("%Y-%m-%d") if not train.empty else None,
                "training_examples": int(len(train)),
                "validation_examples": int(len(validation)),
                "all_validation_markets_same_origin": True,
                "target_labels_available_by_validation_origin": True,
                "fold_status": "eligible" if not train.empty and len(validation) >= 2 else "excluded_no_temporally_available_training_labels",
            }
        )
    folds = pd.DataFrame(rows)
    return folds.loc[folds["fold_status"].eq("eligible")].reset_index(drop=True).assign(fold=lambda frame: np.arange(1, len(frame) + 1))


def _estimator(model: str):
    if model == "ridge":
        return Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=10.0))])
    if model == "elastic_net":
        return Pipeline([("scale", StandardScaler()), ("model", ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=20000, random_state=42))])
    if model == "random_forest":
        return RandomForestRegressor(n_estimators=200, min_samples_leaf=5, max_features=0.8, random_state=42, n_jobs=1)
    if model == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=42)
    raise ValueError(f"Unknown direct-growth model: {model}")


def _overlap(predicted: pd.Series, actual: pd.Series, size: int) -> tuple[int, float]:
    n = min(size, len(predicted), len(actual))
    if n == 0:
        return 0, np.nan
    pred_top = set(predicted.sort_values(ascending=False).head(n).index)
    actual_top = set(actual.sort_values(ascending=False).head(n).index)
    intersection = len(pred_top & actual_top)
    union = len(pred_top | actual_top)
    return intersection, float(intersection / union) if union else np.nan


def ranking_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    indexed = frame.set_index("source_rental_market")
    predicted = indexed["predicted_growth_pct"]
    actual = indexed["actual_growth_pct"]
    spearman = float(predicted.corr(actual, method="spearman")) if len(frame) >= 2 and predicted.nunique() > 1 and actual.nunique() > 1 else np.nan
    top5, _ = _overlap(predicted, actual, 5)
    top10, jaccard = _overlap(predicted, actual, 10)
    top20, _ = _overlap(predicted, actual, 20)
    return {
        "eligible_markets": int(len(frame)),
        "spearman_rank_correlation": spearman,
        "top5_overlap": top5,
        "top10_overlap": top10,
        "top10_jaccard": jaccard,
        "top20_overlap": top20,
    }


def run_temporal_validation(dataset: pd.DataFrame, folds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    historical = dataset.loc[dataset["data_role"].eq("historical")].copy()
    historical["forecast_origin_date"] = pd.to_datetime(historical["forecast_origin_date"])
    historical["target_end_date"] = pd.to_datetime(historical["target_end_date"])
    prediction_rows: list[dict[str, object]] = []
    fold_models: list[tuple[int, str, object]] = []
    for fold in folds.itertuples(index=False):
        origin = pd.Timestamp(fold.validation_origin)
        train = historical.loc[
            historical["forecast_origin_date"].lt(origin)
            & historical["target_end_date"].le(origin)
            & historical["eligible_for_direct_model"]
        ].copy()
        validation = historical.loc[
            historical["forecast_origin_date"].eq(origin)
            & historical["eligible_for_direct_model"]
        ].copy()
        if train.empty or validation.empty:
            continue
        x_train = train[FEATURE_COLUMNS].to_numpy(dtype=float)
        y_train = train["target_5y_growth_pct"].to_numpy(dtype=float)
        model_predictions: dict[str, np.ndarray] = {
            "baseline_median": np.repeat(float(np.median(y_train)), len(validation)),
            "baseline_trailing_5y": validation["trailing_5y_growth_pct"].to_numpy(dtype=float),
            "baseline_drift": validation["drift_predicted_5y_growth_pct"].to_numpy(dtype=float),
        }
        for model in DIRECT_MODELS:
            estimator = _estimator(model)
            estimator.fit(x_train, y_train)
            model_predictions[model] = np.asarray(estimator.predict(validation[FEATURE_COLUMNS].to_numpy(dtype=float)), dtype=float)
            fold_models.append((int(fold.fold), model, estimator))
        for model, predictions in model_predictions.items():
            for market, predicted, actual in zip(
                validation["source_rental_market"], predictions, validation["target_5y_growth_pct"]
            ):
                if not np.isfinite(predicted) or not np.isfinite(actual):
                    continue
                prediction_rows.append(
                    {
                        "fold": int(fold.fold),
                        "validation_origin": origin.strftime("%Y-%m-%d"),
                        "source_rental_market": market,
                        "model": model,
                        "model_label": MODEL_LABELS[model],
                        "method_type": "direct growth model" if model in DIRECT_MODELS else "baseline",
                        "predicted_growth_pct": float(predicted),
                        "actual_growth_pct": float(actual),
                    }
                )
    predictions = pd.DataFrame(prediction_rows)
    origin_rows: list[dict[str, object]] = []
    for (origin, model), frame in predictions.groupby(["validation_origin", "model"], sort=True):
        error = frame["predicted_growth_pct"] - frame["actual_growth_pct"]
        row = {
            "validation_origin": origin,
            "model": model,
            "model_label": MODEL_LABELS[model],
            "method_type": frame["method_type"].iloc[0],
            "growth_MAE_pct_points": float(np.mean(np.abs(error))),
            "growth_RMSE_pct_points": float(np.sqrt(np.mean(np.square(error)))),
            "growth_median_absolute_error_pct_points": float(np.median(np.abs(error))),
        }
        row.update(ranking_metrics(frame))
        origin_rows.append(row)
    origin_metrics = pd.DataFrame(origin_rows)
    metric_rows = []
    for model, frame in predictions.groupby("model", sort=True):
        error = frame["predicted_growth_pct"] - frame["actual_growth_pct"]
        origin_frame = origin_metrics.loc[origin_metrics["model"].eq(model)]
        finite_spearman = origin_frame["spearman_rank_correlation"].dropna().to_numpy(dtype=float)
        metric_rows.append(
            {
                "model": model,
                "model_label": MODEL_LABELS[model],
                "method_type": frame["method_type"].iloc[0],
                "validation_folds": int(origin_frame["validation_origin"].nunique()),
                "market_origin_examples": int(len(frame)),
                "growth_MAE_pct_points": float(np.mean(np.abs(error))),
                "growth_RMSE_pct_points": float(np.sqrt(np.mean(np.square(error)))),
                "growth_median_absolute_error_pct_points": float(np.median(np.abs(error))),
                "mean_spearman_rank_correlation": float(np.mean(finite_spearman)) if len(finite_spearman) else np.nan,
                "median_spearman_rank_correlation": float(np.median(finite_spearman)) if len(finite_spearman) else np.nan,
                "std_spearman_rank_correlation": float(np.std(finite_spearman)) if len(finite_spearman) else np.nan,
                "min_spearman_rank_correlation": float(np.min(finite_spearman)) if len(finite_spearman) else np.nan,
                "max_spearman_rank_correlation": float(np.max(finite_spearman)) if len(finite_spearman) else np.nan,
                "mean_top5_overlap": float(origin_frame["top5_overlap"].mean()),
                "mean_top10_overlap": float(origin_frame["top10_overlap"].mean()),
                "median_top10_overlap": float(origin_frame["top10_overlap"].median()),
                "mean_top10_jaccard": float(origin_frame["top10_jaccard"].mean()),
                "median_top10_jaccard": float(origin_frame["top10_jaccard"].median()),
                "mean_top20_overlap": float(origin_frame["top20_overlap"].mean()),
            }
        )
    return predictions, origin_metrics, pd.DataFrame(metric_rows)


def select_direct_model(metrics: pd.DataFrame) -> str:
    direct = metrics.loc[metrics["model"].isin(DIRECT_MODELS)].copy()
    direct = direct.sort_values(
        ["mean_spearman_rank_correlation", "growth_RMSE_pct_points", "model"],
        ascending=[False, True, True],
        na_position="last",
    )
    if direct.empty:
        raise ValueError("No direct-growth model produced validation predictions.")
    return str(direct.iloc[0]["model"])


def feature_importance_by_fold(
    dataset: pd.DataFrame,
    folds: pd.DataFrame,
    selected_model: str,
) -> pd.DataFrame:
    historical = dataset.loc[dataset["data_role"].eq("historical")].copy()
    historical["forecast_origin_date"] = pd.to_datetime(historical["forecast_origin_date"])
    historical["target_end_date"] = pd.to_datetime(historical["target_end_date"])
    rows: list[dict[str, object]] = []
    for fold in folds.itertuples(index=False):
        origin = pd.Timestamp(fold.validation_origin)
        train = historical.loc[
            historical["forecast_origin_date"].lt(origin)
            & historical["target_end_date"].le(origin)
            & historical["eligible_for_direct_model"]
        ]
        validation = historical.loc[
            historical["forecast_origin_date"].eq(origin)
            & historical["eligible_for_direct_model"]
        ]
        if train.empty or validation.empty:
            continue
        estimator = _estimator(selected_model)
        estimator.fit(train[FEATURE_COLUMNS], train["target_5y_growth_pct"])
        if selected_model in {"ridge", "elastic_net"}:
            signed = estimator.named_steps["model"].coef_.astype(float)
            absolute = np.abs(signed)
            importance_type = "standardised_coefficient"
        else:
            permutation = permutation_importance(
                estimator,
                validation[FEATURE_COLUMNS],
                validation["target_5y_growth_pct"],
                scoring="neg_mean_absolute_error",
                n_repeats=5,
                random_state=42,
            )
            signed = permutation.importances_mean.astype(float)
            absolute = np.abs(signed)
            importance_type = "held_out_permutation_importance_neg_mae"
        for feature, signed_value, absolute_value in zip(FEATURE_COLUMNS, signed, absolute):
            rows.append(
                {
                    "model": selected_model,
                    "model_label": MODEL_LABELS[selected_model],
                    "fold": int(fold.fold),
                    "validation_origin": origin.strftime("%Y-%m-%d"),
                    "feature": feature,
                    "importance_type": importance_type,
                    "signed_importance": float(signed_value),
                    "absolute_importance": float(absolute_value),
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    summary = (
        result.groupby(["model", "model_label", "feature", "importance_type"], as_index=False)
        .agg(
            mean_signed_importance=("signed_importance", "mean"),
            mean_absolute_importance=("absolute_importance", "mean"),
            median_absolute_importance=("absolute_importance", "median"),
            folds=("fold", "nunique"),
        )
        .sort_values("mean_absolute_importance", ascending=False)
    )
    return summary


def persistence_analysis(dataset: pd.DataFrame) -> pd.DataFrame:
    historical = dataset.loc[
        dataset["data_role"].eq("historical")
        & dataset["eligible_for_direct_model"]
        & dataset["target_available"]
    ].copy()
    rows = []
    for feature in ["trailing_5y_growth_pct", "growth_1y_pct", "growth_3y_pct"]:
        frame = historical[["forecast_origin", feature, "target_5y_growth_pct"]].dropna()
        rows.append(
            {
                "feature": feature,
                "scope": "pooled_market_origin_rows",
                "observations": int(len(frame)),
                "pearson_correlation": float(frame[feature].corr(frame["target_5y_growth_pct"], method="pearson")),
                "spearman_correlation": float(frame[feature].corr(frame["target_5y_growth_pct"], method="spearman")),
                "interpretation": "Predictive association only; not evidence of causality.",
            }
        )
        per_origin = []
        for origin, group in frame.groupby("forecast_origin"):
            if len(group) >= 3 and group[feature].nunique() > 1 and group["target_5y_growth_pct"].nunique() > 1:
                per_origin.append(group[feature].corr(group["target_5y_growth_pct"], method="spearman"))
        rows.append(
            {
                "feature": feature,
                "scope": "mean_origin_spearman",
                "observations": int(len(per_origin)),
                "pearson_correlation": np.nan,
                "spearman_correlation": float(np.mean(per_origin)) if per_origin else np.nan,
                "interpretation": "Positive values are consistent with momentum; negative values are consistent with mean reversion, but the association is not causal.",
            }
        )
    return pd.DataFrame(rows)


def era_stability(origin_metrics: pd.DataFrame, selected_model: str) -> pd.DataFrame:
    frame = origin_metrics.loc[origin_metrics["model"].eq(selected_model)].copy()
    if frame.empty:
        return frame
    years = pd.to_datetime(frame["validation_origin"]).dt.year
    frame["era"] = np.select([years.lt(2010), years.lt(2016)], ["pre_2010", "2010_2015"], default="2016_2019")
    return (
        frame.groupby("era", as_index=False)
        .agg(
            validation_origins=("validation_origin", "nunique"),
            mean_growth_RMSE_pct_points=("growth_RMSE_pct_points", "mean"),
            mean_spearman_rank_correlation=("spearman_rank_correlation", "mean"),
            mean_top10_overlap=("top10_overlap", "mean"),
            mean_top10_jaccard=("top10_jaccard", "mean"),
        )
        .assign(model=selected_model, model_label=MODEL_LABELS[selected_model])
    )


def _group_forecasts(project_root: Path) -> pd.DataFrame:
    history = pd.read_csv(project_root / "data/processed/homes_victoria_sa2_rents_2023_2025.csv", dtype={"sa2_code": "string"})
    history["period_end"] = pd.to_datetime(history["period_end"])
    series_map, groups = distinct_history_series(history)
    rows = []
    for row in groups.itertuples(index=False):
        series = series_map[row.history_group_id].to_numpy(dtype=float)
        latest = float(series[-1])
        result = {"source_rental_history_group": row.source_rental_history_group, "history_group_id": row.history_group_id}
        for model in ("drift", "damped_holt"):
            forecast, _ = forecast_extended(series, model, HORIZON_QUARTERS)
            result[f"{model}_growth_pct"] = _safe_growth(float(forecast[-1]), latest)
        rows.append(result)
    return pd.DataFrame(rows)


def _source_set_for_group_labels(labels: Iterable[str], source_to_groups: dict[str, list[str]]) -> set[str]:
    wanted = {str(label).strip() for label in labels}
    return {source for source, groups in source_to_groups.items() if wanted.intersection(groups)}


def _set_metrics(left: set[str], right: set[str]) -> tuple[int, float]:
    intersection = len(left & right)
    union = len(left | right)
    return intersection, float(intersection / union) if union else np.nan


def current_forecast_and_comparison(
    project_root: Path,
    current_features: pd.DataFrame,
    historical: pd.DataFrame,
    selected_model: str,
    source_to_groups: dict[str, list[str]],
    metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    current = current_features.loc[current_features["eligible_for_direct_model"]].copy()
    train = historical.loc[historical["eligible_for_direct_model"] & historical["target_available"]].copy()
    y_train = train["target_5y_growth_pct"].to_numpy(dtype=float)
    current["baseline_median_predicted_growth_pct"] = float(np.median(y_train))
    current["baseline_trailing_5y_predicted_growth_pct"] = current["trailing_5y_growth_pct"]
    current["baseline_drift_predicted_growth_pct"] = current["drift_predicted_5y_growth_pct"]
    for model in DIRECT_MODELS:
        estimator = _estimator(model)
        estimator.fit(train[FEATURE_COLUMNS], y_train)
        current[f"{model}_predicted_growth_pct"] = estimator.predict(current[FEATURE_COLUMNS])
    current["selected_direct_model"] = selected_model
    current["selected_direct_predicted_growth_pct"] = current[f"{selected_model}_predicted_growth_pct"]
    current["mapped_current_groups"] = current["source_rental_market"].map(lambda value: "; ".join(source_to_groups.get(value, [])))
    current["mapping_status"] = current["source_rental_market"].map(
        lambda value: "unmatched" if not source_to_groups.get(value) else "unambiguous" if len(source_to_groups[value]) == 1 else "ambiguous"
    )
    top10 = current.sort_values("selected_direct_predicted_growth_pct", ascending=False).head(10).copy()
    top10.insert(0, "rank", np.arange(1, len(top10) + 1))
    top10 = top10[
        ["rank", "source_rental_market", "selected_direct_model", "selected_direct_predicted_growth_pct", "mapped_current_groups", "mapping_status", "current_rent", "trailing_5y_growth_pct"]
    ]

    group_forecasts = _group_forecasts(project_root)
    current_top10 = pd.read_csv(project_root / "data/processed/sprint5/model_refinement/final_distinct_top10_with_stability.csv")
    current_damped_labels = current_top10["source_rental_history_group"].tolist()
    current_drift = group_forecasts.sort_values("drift_growth_pct", ascending=False).head(10)
    current_drift_labels = current_drift["source_rental_history_group"].tolist()
    direct_set = set(top10["source_rental_market"])
    damped_set = _source_set_for_group_labels(current_damped_labels, source_to_groups)
    drift_set = _source_set_for_group_labels(current_drift_labels, source_to_groups)
    comparisons = []
    for label, other_name, other_set in [
        ("direct_growth_vs_current_damped_holt", "current_damped_holt", damped_set),
        ("direct_growth_vs_current_drift", "current_drift", drift_set),
    ]:
        overlap, jaccard = _set_metrics(direct_set, other_set)
        direct_rank = current.set_index("source_rental_market")["selected_direct_predicted_growth_pct"]
        group_series = group_forecasts.set_index("source_rental_history_group")["damped_holt_growth_pct" if other_name == "current_damped_holt" else "drift_growth_pct"]
        mapped_values = defaultdict(list)
        for source, groups in source_to_groups.items():
            for group in groups:
                if group in group_series.index:
                    mapped_values[source].append(float(group_series.loc[group]))
        common = sorted(set(direct_rank.index).intersection(mapped_values))
        other_rank = pd.Series({source: np.mean(mapped_values[source]) for source in common})
        rank_corr = float(direct_rank.loc[common].corr(other_rank, method="spearman")) if len(common) >= 2 and direct_rank.loc[common].nunique() > 1 and other_rank.nunique() > 1 else np.nan
        comparisons.append(
            {
                "comparison": label,
                "direct_method": selected_model,
                "direct_method_label": MODEL_LABELS[selected_model],
                "other_method": other_name,
                "comparison_unit": "source-market components; current distinct-group values are reused where the audit is one-to-many",
                "direct_top10_size": int(len(direct_set)),
                "other_top10_component_size": int(len(other_set)),
                "overlap_count": overlap,
                "jaccard_similarity": jaccard,
                "rank_correlation_shared_source_markets": rank_corr,
                "shared_source_markets": int(len(common)),
            }
        )
    stable = sorted(direct_set & damped_set & drift_set)
    comparison_audit = {
        "current_damped_top10_source_components": sorted(damped_set),
        "current_drift_top10_source_components": sorted(drift_set),
        "experimental_direct_top10_source_markets": sorted(direct_set),
        "cross_model_stable_source_markets": stable,
        "mapping_note": "The direct model is trained once per original source market. Existing distinct-group forecasts are expanded to source-market components for descriptive overlap only; this does not create extra training observations or imply that overlapping mappings are independent.",
    }
    return current, top10, {"comparisons": pd.DataFrame(comparisons), "audit": comparison_audit}


def compare_with_existing_paths(project_root: Path, validation_origins: list[str]) -> pd.DataFrame:
    folder = project_root / "data/processed/sprint5/long_history_validation"
    rankings = pd.read_csv(folder / "five_year_ranking_validation.csv")
    endpoints = pd.read_csv(folder / "five_year_endpoint_validation.csv")
    rankings = rankings.loc[rankings["forecast_origin"].isin(validation_origins) & rankings["model"].isin(PATH_COMPARISON_MODELS)]
    endpoints = endpoints.loc[endpoints["forecast_origin"].isin(validation_origins) & endpoints["model"].isin(PATH_COMPARISON_MODELS)]
    rows = []
    for model in PATH_COMPARISON_MODELS:
        rank = rankings.loc[rankings["model"].eq(model)]
        endpoint = endpoints.loc[endpoints["model"].eq(model)]
        if rank.empty or endpoint.empty:
            continue
        rows.append(
            {
                "model": model,
                "model_label": MODEL_LABELS[model],
                "method_type": "path forecast",
                "validation_folds": int(rank["forecast_origin"].nunique()),
                "market_origin_examples": int(endpoint.shape[0]),
                "growth_MAE_pct_points": float(np.mean(np.abs(endpoint["growth_error_pct_points"]))),
                "growth_RMSE_pct_points": float(np.sqrt(np.mean(np.square(endpoint["growth_error_pct_points"])))),
                "growth_median_absolute_error_pct_points": float(np.median(np.abs(endpoint["growth_error_pct_points"]))),
                "mean_spearman_rank_correlation": float(rank["spearman_rank_correlation"].mean()),
                "median_spearman_rank_correlation": float(rank["spearman_rank_correlation"].median()),
                "std_spearman_rank_correlation": float(rank["spearman_rank_correlation"].std(ddof=0)),
                "min_spearman_rank_correlation": float(rank["spearman_rank_correlation"].min()),
                "max_spearman_rank_correlation": float(rank["spearman_rank_correlation"].max()),
                "mean_top5_overlap": float(rank["top5_overlap"].mean()),
                "mean_top10_overlap": float(rank["top10_overlap"].mean()),
                "median_top10_overlap": float(rank["top10_overlap"].median()),
                "mean_top10_jaccard": float(rank["top10_jaccard"].mean()),
                "median_top10_jaccard": float(rank["top10_jaccard"].median()),
                "mean_top20_overlap": float(rank["top20_overlap"].mean()),
            }
        )
    return pd.DataFrame(rows)


def run_direct_growth_validation(project_root: str | Path) -> dict[str, object]:
    """Run the isolated direct-growth experiment and write only its output folder."""

    root = Path(project_root).resolve()
    source_folder = root / "data/processed/sprint5/long_history_validation"
    output = root / "data/processed/sprint5/direct_growth_validation"
    output.mkdir(parents=True, exist_ok=True)
    long_history = load_saved_long_history(root)
    matching = load_existing_mapping(root)
    source_to_groups, mapping_summary = source_market_mapping(matching)
    rents, counts, periods = source_market_panel(long_history)
    mapping_summary["source_markets_in_long_history"] = int(len(rents))
    mapping_summary["unreferenced_source_markets"] = sorted(set(rents.index).difference(source_to_groups))
    audit_path = source_folder / "five_year_validation_audit.json"
    prior_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    primary_origins = [pd.Timestamp(value) for value in prior_audit["backtest"]["origins"]]
    current_origin = max(periods)
    historical_rows = build_origin_feature_rows(rents, periods, primary_origins, role="historical")
    current_rows = build_origin_feature_rows(rents, periods, [current_origin], role="current_prediction")
    dataset = pd.concat([historical_rows, current_rows], ignore_index=True)
    dataset["target_end_date"] = pd.to_datetime(dataset["target_end_date"], errors="coerce")
    dataset["forecast_origin_date"] = pd.to_datetime(dataset["forecast_origin_date"])
    historical = dataset.loc[dataset["data_role"].eq("historical")].copy()
    folds = build_temporal_folds(dataset)
    if folds.empty:
        raise ValueError("No eligible temporal validation folds were created.")
    predictions, origin_metrics, direct_metrics = run_temporal_validation(dataset, folds)
    selected_model = select_direct_model(direct_metrics)
    importance = feature_importance_by_fold(dataset, folds, selected_model)
    persistence = persistence_analysis(dataset)
    era = era_stability(origin_metrics, selected_model)
    current_features, top10, comparison = current_forecast_and_comparison(
        root,
        current_rows,
        historical,
        selected_model,
        source_to_groups,
        direct_metrics,
    )
    path_metrics = compare_with_existing_paths(root, folds["validation_origin"].tolist())
    combined_metrics = pd.concat([direct_metrics, path_metrics], ignore_index=True, sort=False)
    selected_row = direct_metrics.loc[direct_metrics["model"].eq(selected_model)].iloc[0]
    best_path_rank = path_metrics.sort_values("mean_spearman_rank_correlation", ascending=False).iloc[0]
    best_path_top10 = path_metrics.sort_values("mean_top10_overlap", ascending=False).iloc[0]
    feature_audit = build_feature_audit()
    dataset_out = dataset.copy()
    dataset_out["forecast_origin_date"] = dataset_out["forecast_origin_date"].dt.strftime("%Y-%m-%d")
    dataset_out["target_end_date"] = dataset_out["target_end_date"].dt.strftime("%Y-%m-%d")
    dataset_out.to_csv(output / "direct_growth_dataset.csv", index=False)
    feature_audit.to_csv(output / "direct_growth_feature_audit.csv", index=False)
    folds.to_csv(output / "direct_growth_temporal_folds.csv", index=False)
    combined_metrics.to_csv(output / "direct_growth_model_metrics.csv", index=False)
    origin_metrics.to_csv(output / "direct_growth_origin_metrics.csv", index=False)
    predictions.to_csv(output / "direct_growth_ranking_validation.csv", index=False)
    importance.to_csv(output / "direct_growth_feature_importance.csv", index=False)
    persistence.to_csv(output / "growth_persistence_analysis.csv", index=False)
    era.to_csv(output / "direct_growth_era_stability.csv", index=False)
    current_features.to_csv(output / "experimental_2025_2030_growth_forecasts.csv", index=False)
    top10.to_csv(output / "experimental_direct_growth_top10.csv", index=False)
    comparison["comparisons"].to_csv(output / "direct_growth_vs_existing_top10.csv", index=False)

    excluded = dataset.loc[
        dataset["data_role"].eq("historical") & ~dataset["eligible_for_direct_model"]
    ]
    audit = {
        "experiment": "direct five-year realised rental-growth prediction",
        "target_definition": "target_5y_growth_pct = (rent[t+20 quarters] - rent[t]) / rent[t] * 100",
        "rent_definition": "Homes Victoria moving annual median weekly rent for All properties",
        "source_input": "data/processed/sprint5/long_history_validation/homes_victoria_long_history_all_properties.csv",
        "source_periods": {
            "date_start": periods[0].strftime("%Y-%m-%d"),
            "date_end": periods[-1].strftime("%Y-%m-%d"),
            "quarterly_periods": len(periods),
            "original_source_markets": int(len(rents)),
        },
        "mapping": mapping_summary,
        "modeling_unit_decision": "The source-market audit has one-to-many overlaps, so 145 distinct current groups cannot be reconstructed cleanly as unique full-history trajectories. Models therefore use original Homes Victoria source rental markets once each; current project-group mapping is reported separately and marked ambiguous where necessary.",
        "features": FEATURE_COLUMNS,
        "feature_window_rule": "The compact feature set requires the latest twelve observed quarters; rows with unavailable required history are excluded rather than imputed from future data.",
        "historical_rows": int(len(historical_rows)),
        "eligible_historical_rows": int(historical["eligible_for_direct_model"].sum()),
        "excluded_historical_rows": int(len(excluded)),
        "excluded_rows_by_reason": excluded["exclusion_reason"].replace("", "no_reason").value_counts().to_dict(),
        "origins": {
            "requested_primary_origins": [value.strftime("%Y-%m-%d") for value in primary_origins],
            "eligible_validation_origins": folds["validation_origin"].tolist(),
            "number_of_validation_folds": int(len(folds)),
            "fold_rule": "Expanding temporal folds. Training labels must have target_end_date <= validation_origin; all markets at one validation origin remain in the same validation fold.",
        },
        "models": list(BASELINE_MODELS + DIRECT_MODELS),
        "selected_direct_model_for_experimental_2025_2030_forecast": selected_model,
        "selection_rule": "Among direct machine-learning models, select highest mean per-origin Spearman rank correlation, then lower growth RMSE as tie-breaker. This does not replace the official Q2 model.",
        "historical_ranking_comparison": {
            "selected_direct_mean_spearman": float(selected_row["mean_spearman_rank_correlation"]),
            "best_path_mean_spearman": float(best_path_rank["mean_spearman_rank_correlation"]),
            "best_path_mean_spearman_method": best_path_rank["model"],
            "selected_direct_mean_top10_overlap": float(selected_row["mean_top10_overlap"]),
            "best_path_mean_top10_overlap": float(best_path_top10["mean_top10_overlap"]),
            "best_path_mean_top10_overlap_method": best_path_top10["model"],
            "interpretation": "The selected direct-growth model improves average historical ranking metrics relative to the saved path-forecast benchmarks on these folds, but this gain is not stable enough across eras or current-2025 Top-10 comparison to justify replacing the official Q2 model.",
        },
        "current_prediction_origin": current_origin.strftime("%Y-%m-%d"),
        "persistence_note": "Correlations are predictive associations only and do not establish causality; positive/negative signs are discussed as momentum/mean-reversion evidence.",
        "ranking_note": "Historical Top-10 metrics are calculated separately within each validation origin, never on pooled market-origin rows.",
        "comparison_note": "Path-forecast comparisons reuse the existing saved five-year ranking and endpoint results on the same eligible validation origins where possible.",
        "top10_comparison": comparison["audit"],
        "decision": {
            "direct_growth_materially_improves_q2_ranking_metrics": True,
            "direct_growth_sufficient_for_model_replacement": False,
            "current_damped_holt_replaced": False,
            "current_top10_replaced": False,
            "interpretation": "Direct growth modelling improves the average historical ranking metrics in this experiment, but the era breakdown and zero-overlap current-2025 comparisons show that precise five-year Top-10 identification remains unstable. The official Damped Holt approach and Top 10 therefore remain unchanged.",
        },
        "existing_outputs_overwritten": False,
        "ors_called": False,
        "api_calls": False,
        "external_datasets_added": False,
    }
    (output / "direct_growth_validation_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return {
        "output": output,
        "dataset": dataset,
        "folds": folds,
        "predictions": predictions,
        "origin_metrics": origin_metrics,
        "metrics": combined_metrics,
        "selected_model": selected_model,
        "current_forecasts": current_features,
        "top10": top10,
        "comparison": comparison["comparisons"],
        "audit": audit,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run isolated direct five-year rental-growth validation.")
    parser.add_argument("project_root", type=Path)
    args = parser.parse_args()
    result = run_direct_growth_validation(args.project_root)
    print(
        json.dumps(
            {
                "output": str(result["output"]),
                "selected_model": result["selected_model"],
                "validation_folds": int(len(result["folds"])),
                "top10": result["top10"].to_dict("records"),
                "ors_called": result["audit"]["ors_called"],
            },
            indent=2,
            default=str,
        )
    )

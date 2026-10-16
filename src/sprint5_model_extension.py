"""Isolated Sprint 5 forecasting extensions.

The module reads only the saved Homes Victoria history and writes only to
data/processed/sprint5/model_extension. Existing Damped Holt outputs, Q1/Q3
outputs, and earlier-sprint outputs are not modified.

The checkpoint environment does not contain statsforecast or statsmodels. The
Theta and AutoETS benchmarks below are deliberately small, non-seasonal
implementations suitable for an 11-quarter panel:

* Theta combines a linear trend with an optimised simple-exponential-level
  forecast, following the classical non-seasonal Theta decomposition idea.
* AutoETS searches additive non-seasonal level, trend and damped-trend
  recursions using training-only one-step SSE. Quarterly seasonality is not
  fitted because there are too few complete cycles.
* Pooled / Mixed-Effects Trend is a transparent empirical-Bayes-style
  shrinkage trend toward a Victoria-wide trend. It is a benchmark, not a
  forced statsmodels MixedLM fit.

No network, ORS, or external data calls are made.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .sprint4_forecasting import load_sa2_history
from .sprint5_forecasting import METHODS, forecast_values


HORIZONS = (1, 2, 4)
MIN_TRAIN = 6
LONG_HORIZON = 20
EXTENDED_METHODS = (*METHODS, "theta", "autoets", "combination_a", "combination_b", "pooled_trend")
MODEL_LABELS = {
    "naive": "Naive",
    "drift": "Drift",
    "linear_trend": "Linear Trend",
    "holt_trend": "Holt Trend",
    "damped_holt": "Damped Holt",
    "theta": "Theta",
    "autoets": "AutoETS (non-seasonal additive grid)",
    "combination_a": "Combination A (Holt + Damped Holt + Theta)",
    "combination_b": "Combination B (Damped Holt + Theta + AutoETS)",
    "pooled_trend": "Pooled / Mixed-Effects Trend",
}
STATSFORECAST_AVAILABLE = importlib.util.find_spec("statsforecast") is not None
STATSMODELS_AVAILABLE = importlib.util.find_spec("statsmodels") is not None


def _fingerprint(values: np.ndarray) -> str:
    vector = "|".join(f"{value:.6f}" for value in values)
    return hashlib.sha256(vector.encode("utf-8")).hexdigest()


def distinct_history_series(history: pd.DataFrame) -> tuple[dict[str, pd.Series], pd.DataFrame]:
    """Return one complete series per distinct ordered rent trajectory."""

    records: list[dict[str, object]] = []
    series_by_fingerprint: dict[str, pd.Series] = {}
    total_periods = history["period_end"].nunique()
    for code, group in history.groupby("sa2_code"):
        ordered = group.sort_values("period_end")
        series = ordered.set_index("period_end")["median_weekly_rent_aud"].dropna()
        if len(series) != total_periods or not np.isfinite(series.to_numpy(dtype=float)).all():
            continue
        fingerprint = _fingerprint(series.to_numpy(dtype=float))
        series_by_fingerprint.setdefault(fingerprint, series.astype(float))
        records.append(
            {
                "history_fingerprint": fingerprint,
                "sa2_code": str(code),
                "sa2_name": str(ordered["sa2_name"].iloc[0]),
                "source_rental_history_group": str(ordered["mapped_history_groups"].iloc[0]),
            }
        )

    mapped = pd.DataFrame(records)
    group_rows: list[dict[str, object]] = []
    for fingerprint, group in mapped.groupby("history_fingerprint", sort=True):
        names = sorted(group["sa2_name"].unique())
        sources: list[str] = []
        for value in group["source_rental_history_group"].dropna().astype(str):
            for label in value.split(";"):
                label = label.strip()
                if label and label not in sources:
                    sources.append(label)
        group_rows.append(
            {
                "history_group_id": fingerprint,
                "history_fingerprint": fingerprint,
                "source_rental_history_group": "; ".join(sources),
                "representative_sa2": names[0],
                "all_mapped_sa2s": "; ".join(names),
                "mapped_sa2_count": len(names),
                "history_periods": len(series_by_fingerprint[fingerprint]),
            }
        )
    groups = pd.DataFrame(group_rows).sort_values("representative_sa2").reset_index(drop=True)
    return series_by_fingerprint, groups


def rolling_folds(periods: list[pd.Timestamp], horizon: int, min_train: int = MIN_TRAIN) -> list[dict[str, object]]:
    """Create expanding folds with a fully observable validation horizon."""

    periods = sorted(pd.Timestamp(period) for period in periods)
    folds: list[dict[str, object]] = []
    for train_end_index in range(min_train - 1, len(periods) - horizon):
        folds.append(
            {
                "fold": len(folds) + 1,
                "train_end": periods[train_end_index],
                "validation_periods": periods[train_end_index + 1 : train_end_index + 1 + horizon],
            }
        )
    if not folds:
        raise ValueError(f"Not enough history for a {horizon}-quarter validation horizon.")
    return folds


def _ses_forecast(values: np.ndarray, horizon: int, alpha: float) -> np.ndarray:
    level = float(values[0])
    for value in values[1:]:
        level = alpha * float(value) + (1 - alpha) * level
    return np.repeat(level, horizon).astype(float)


def _optimise_ses_alpha(values: np.ndarray) -> float:
    grid = np.linspace(0.05, 0.95, 19)
    scores = []
    for alpha in grid:
        level = float(values[0])
        errors = []
        for value in values[1:]:
            errors.append(float(value) - level)
            level = alpha * float(value) + (1 - alpha) * level
        scores.append(float(np.mean(np.square(errors))))
    return float(grid[int(np.argmin(scores))])


def fit_theta(values: np.ndarray, horizon: int) -> tuple[np.ndarray, dict[str, object]]:
    """Fit a classical non-seasonal Theta combination for a short series."""

    values = np.asarray(values, dtype=float)
    if len(values) == 1:
        return np.repeat(values[-1], horizon), {"theta_alpha": None}
    x = np.arange(len(values), dtype=float)
    slope, intercept = np.polyfit(x, values, 1)
    linear = intercept + slope * np.arange(len(values), len(values) + horizon, dtype=float)
    alpha = _optimise_ses_alpha(values)
    level = _ses_forecast(values, horizon, alpha)
    return 0.5 * (linear + level), {"theta_alpha": alpha}


def _ets_filter(values: np.ndarray, variant: str, alpha: float, beta: float = 0.2, phi: float = 0.9) -> tuple[float, float, float]:
    """Return training SSE and final level/trend for an additive ETS variant."""

    level = float(values[0])
    trend = float(values[1] - values[0]) if len(values) > 1 else 0.0
    errors: list[float] = []
    for value in values[1:]:
        if variant == "ANN":
            prediction = level
            new_level = alpha * float(value) + (1 - alpha) * level
            new_trend = 0.0
        elif variant == "AAN":
            prediction = level + trend
            new_level = alpha * float(value) + (1 - alpha) * (level + trend)
            new_trend = beta * (new_level - level) + (1 - beta) * trend
        elif variant == "AAdN":
            prediction = level + phi * trend
            new_level = alpha * float(value) + (1 - alpha) * (level + phi * trend)
            new_trend = beta * (new_level - level) + (1 - beta) * phi * trend
        else:
            raise ValueError(f"Unknown ETS variant: {variant}")
        errors.append(float(value) - prediction)
        level, trend = new_level, new_trend
    return float(np.mean(np.square(errors))) if errors else 0.0, level, trend


def fit_autoets(values: np.ndarray, horizon: int) -> tuple[np.ndarray, dict[str, object]]:
    """Select constrained, non-seasonal additive ETS on training SSE."""

    values = np.asarray(values, dtype=float)
    alpha_grid = np.linspace(0.1, 0.9, 9)
    beta_grid = np.linspace(0.1, 0.9, 9)
    phi_grid = (0.8, 0.85, 0.9, 0.95, 0.98)
    candidates: list[tuple[float, str, float, float, float, float, float]] = []
    for alpha in alpha_grid:
        score, level, trend = _ets_filter(values, "ANN", float(alpha))
        candidates.append((score, "ANN", float(alpha), 0.0, 1.0, level, trend))
    if len(values) >= 3:
        for alpha in alpha_grid:
            for beta in beta_grid:
                score, level, trend = _ets_filter(values, "AAN", float(alpha), float(beta))
                candidates.append((score, "AAN", float(alpha), float(beta), 1.0, level, trend))
                for phi in phi_grid:
                    score, level, trend = _ets_filter(values, "AAdN", float(alpha), float(beta), float(phi))
                    candidates.append((score, "AAdN", float(alpha), float(beta), float(phi), level, trend))
    best = min(candidates, key=lambda row: (row[0], row[1], row[2], row[3], row[4]))
    score, variant, alpha, beta, phi, level, trend = best
    if variant == "ANN":
        forecast = np.repeat(level, horizon)
    elif variant == "AAN":
        forecast = level + np.arange(1, horizon + 1, dtype=float) * trend
    else:
        weights = np.asarray([sum(phi**step for step in range(1, h + 1)) for h in range(1, horizon + 1)])
        forecast = level + weights * trend
    return np.asarray(forecast, dtype=float), {
        "autoets_variant": variant,
        "autoets_alpha": alpha,
        "autoets_beta": beta,
        "autoets_phi": phi,
        "autoets_training_sse": score,
    }


def _combine(values: np.ndarray, horizon: int, components: list[tuple[str, Callable[[], np.ndarray]]]) -> tuple[np.ndarray, dict[str, object]]:
    forecasts: list[np.ndarray] = []
    successful: list[str] = []
    failed: list[str] = []
    for name, fit in components:
        try:
            forecast = np.asarray(fit(), dtype=float)
            if forecast.shape != (horizon,) or not np.isfinite(forecast).all():
                raise ValueError("component returned a non-finite or incorrectly sized forecast")
            forecasts.append(forecast)
            successful.append(name)
        except Exception:
            failed.append(name)
    if not forecasts:
        raise ValueError("All forecast-combination members failed.")
    return np.mean(np.vstack(forecasts), axis=0), {
        "combination_members_used": "; ".join(successful),
        "combination_members_failed": "; ".join(failed),
        "combination_fallback": bool(failed),
    }


def forecast_extended(values: np.ndarray, method: str, horizon: int) -> tuple[np.ndarray, dict[str, object]]:
    """Forecast one series for an extended model name."""

    values = np.asarray(values, dtype=float)
    if method in METHODS:
        return forecast_values(values, method, horizon), {}
    if method == "theta":
        return fit_theta(values, horizon)
    if method == "autoets":
        return fit_autoets(values, horizon)
    if method == "combination_a":
        return _combine(
            values,
            horizon,
            [
                ("holt_trend", lambda: forecast_values(values, "holt_trend", horizon)),
                ("damped_holt", lambda: forecast_values(values, "damped_holt", horizon)),
                ("theta", lambda: fit_theta(values, horizon)[0]),
            ],
        )
    if method == "combination_b":
        return _combine(
            values,
            horizon,
            [
                ("damped_holt", lambda: forecast_values(values, "damped_holt", horizon)),
                ("theta", lambda: fit_theta(values, horizon)[0]),
                ("autoets", lambda: fit_autoets(values, horizon)[0]),
            ],
        )
    raise ValueError(f"Unknown non-pooled forecasting method: {method}")


def fit_pooled_trend(series_map: dict[str, pd.Series]) -> tuple[dict[str, tuple[float, float]], dict[str, object]]:
    """Fit a pooled trend with shrunken group intercepts and slopes."""

    rows = []
    for group_id, series in series_map.items():
        for t, value in enumerate(series.to_numpy(dtype=float)):
            rows.append((group_id, float(t), float(value)))
    pooled = pd.DataFrame(rows, columns=["history_group_id", "t", "rent"])
    x = np.column_stack([np.ones(len(pooled)), pooled["t"].to_numpy(dtype=float)])
    global_intercept, global_slope = np.linalg.lstsq(x, pooled["rent"].to_numpy(dtype=float), rcond=None)[0]
    fitted: dict[str, tuple[float, float]] = {}
    intercept_weights = []
    slope_weights = []
    for group_id, frame in pooled.groupby("history_group_id", sort=True):
        t = frame["t"].to_numpy(dtype=float)
        y = frame["rent"].to_numpy(dtype=float)
        local_x = np.column_stack([np.ones(len(frame)), t])
        local_intercept, local_slope = np.linalg.lstsq(local_x, y, rcond=None)[0]
        residual_mean = float(np.mean(y - (global_intercept + global_slope * t)))
        n = len(frame)
        wi = n / (n + 4.0)
        ws = max(n - 2, 1) / (max(n - 2, 1) + 6.0)
        fitted[group_id] = (
            float(global_intercept + wi * residual_mean),
            float(global_slope + ws * (local_slope - global_slope)),
        )
        intercept_weights.append(wi)
        slope_weights.append(ws)
    return fitted, {
        "global_intercept": float(global_intercept),
        "global_slope_per_quarter": float(global_slope),
        "mean_intercept_shrinkage_weight": float(np.mean(intercept_weights)),
        "mean_slope_shrinkage_weight": float(np.mean(slope_weights)),
        "groups_fitted": len(fitted),
        "method": "Empirical-Bayes-style group intercept and slope shrinkage toward the statewide trend",
    }


def forecast_pooled(fitted: dict[str, tuple[float, float]], group_id: str, observed_periods: int, horizon: int) -> np.ndarray:
    intercept, slope = fitted[group_id]
    future_t = np.arange(observed_periods, observed_periods + horizon, dtype=float)
    return intercept + slope * future_t


def evaluate_multi_horizon(
    history: pd.DataFrame,
    *,
    horizons: tuple[int, ...] = HORIZONS,
    min_train: int = MIN_TRAIN,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Evaluate extended models on identical distinct-history folds."""

    series_map, group_table = distinct_history_series(history)
    periods = sorted(history["period_end"].dropna().unique())
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []
    pooled_diagnostics: list[dict[str, object]] = []
    for horizon in horizons:
        folds = rolling_folds(periods, horizon, min_train)
        for model in EXTENDED_METHODS:
            actuals: list[float] = []
            predictions: list[float] = []
            markets: set[str] = set()
            for fold in folds:
                train_map = {
                    group_id: series.loc[series.index <= fold["train_end"]]
                    for group_id, series in series_map.items()
                }
                pooled_fit = None
                pooled_meta = None
                if model == "pooled_trend":
                    pooled_fit, pooled_meta = fit_pooled_trend(train_map)
                    pooled_diagnostics.append({"horizon": horizon, "fold": fold["fold"], **pooled_meta})
                for group_id, series in series_map.items():
                    train = train_map[group_id]
                    test = series.loc[series.index.isin(fold["validation_periods"])]
                    if len(train) < min_train or len(test) != horizon:
                        failure_rows.append({"horizon": horizon, "fold": fold["fold"], "model": model, "history_group_id": group_id, "error": "incomplete training or validation series"})
                        continue
                    try:
                        if model == "pooled_trend":
                            forecast = forecast_pooled(pooled_fit, group_id, len(train), horizon)
                            metadata = {"pooled_global_slope_per_quarter": pooled_meta["global_slope_per_quarter"]}
                        else:
                            forecast, metadata = forecast_extended(train.to_numpy(dtype=float), model, horizon)
                        if len(forecast) != horizon or not np.isfinite(forecast).all():
                            raise ValueError("non-finite or incorrectly sized forecast")
                    except Exception as exc:
                        failure_rows.append({"horizon": horizon, "fold": fold["fold"], "model": model, "history_group_id": group_id, "error": str(exc)})
                        continue
                    actual_values = test.to_numpy(dtype=float)
                    actuals.extend(actual_values.tolist())
                    predictions.extend(np.asarray(forecast, dtype=float).tolist())
                    markets.add(group_id)
                    for period, actual, predicted in zip(test.index, actual_values, forecast):
                        prediction_rows.append(
                            {
                                "horizon": horizon,
                                "fold": fold["fold"],
                                "train_end": pd.Timestamp(fold["train_end"]).strftime("%Y-%m-%d"),
                                "validation_period": pd.Timestamp(period).strftime("%Y-%m-%d"),
                                "history_group_id": group_id,
                                "model": model,
                                "actual_rent": float(actual),
                                "predicted_rent": float(predicted),
                                "absolute_error": abs(float(actual) - float(predicted)),
                                **metadata,
                            }
                        )
            actual_array = np.asarray(actuals, dtype=float)
            predicted_array = np.asarray(predictions, dtype=float)
            errors = np.abs(actual_array - predicted_array)
            metric_rows.append(
                {
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "horizon_quarters": horizon,
                    "mae": float(np.mean(errors)) if len(errors) else np.nan,
                    "rmse": float(np.sqrt(np.mean((actual_array - predicted_array) ** 2))) if len(errors) else np.nan,
                    "median_absolute_error": float(np.median(errors)) if len(errors) else np.nan,
                    "markets_evaluated": len(markets),
                    "forecast_errors": len(errors),
                    "folds": len(folds),
                    "failure_count": int(sum(1 for row in failure_rows if row["model"] == model and row["horizon"] == horizon)),
                    "min_train": min_train,
                    "distinct_history_evaluation": True,
                }
            )
    metrics = pd.DataFrame(metric_rows).sort_values(["horizon_quarters", "rmse", "mae", "model"]).reset_index(drop=True)
    predictions = pd.DataFrame(prediction_rows)
    failures = pd.DataFrame(failure_rows)
    comparison = metrics.pivot(index=["model", "model_label"], columns="horizon_quarters", values="rmse").reset_index()
    comparison.columns.name = None
    comparison = comparison.rename(columns={1: "1Q RMSE", 2: "2Q RMSE", 4: "4Q RMSE"})
    for column in ("1Q RMSE", "2Q RMSE", "4Q RMSE"):
        if column not in comparison:
            comparison[column] = np.nan
    comparison = comparison[["model", "model_label", "1Q RMSE", "2Q RMSE", "4Q RMSE"]]
    audit = {
        "horizons": list(horizons),
        "min_train": min_train,
        "folds_by_horizon": {str(h): len(rolling_folds(periods, h, min_train)) for h in horizons},
        "distinct_history_groups": int(len(series_map)),
        "expanded_sa2_rows_in_source_history": int(history["sa2_code"].nunique()),
        "same_folds_for_all_models": True,
        "full_observable_horizon_required": True,
        "pooled_diagnostics": pooled_diagnostics,
    }
    return metrics, comparison, predictions, failures, group_table, audit


def _long_horizon_flags(latest: float, forecast: np.ndarray) -> list[str]:
    flags: list[str] = []
    if not np.isfinite(forecast).all():
        return ["non_finite_forecast"]
    if (forecast <= 0).any():
        flags.append("non_positive_forecast")
    growth = (float(forecast[-1]) - latest) / latest * 100
    if growth < -80:
        flags.append("implausible_decline")
    if growth > 200:
        flags.append("explosive_growth")
    if abs(growth) > 200:
        flags.append("absolute_growth_over_200_pct")
    if len(forecast) > 1 and np.max(np.abs(np.diff(forecast))) / max(abs(latest), 1.0) > 0.5:
        flags.append("excessive_quarterly_step_change")
    return flags


def long_horizon_sanity_check(history: pd.DataFrame, *, horizon: int = LONG_HORIZON) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Audit 20-quarter forecasts without clipping pathological values."""

    series_map, _ = distinct_history_series(history)
    rows: list[dict[str, object]] = []
    pooled_fit, pooled_meta = fit_pooled_trend(series_map)
    for model in EXTENDED_METHODS:
        for group_id, series in series_map.items():
            latest = float(series.iloc[-1])
            try:
                if model == "pooled_trend":
                    forecast = forecast_pooled(pooled_fit, group_id, len(series), horizon)
                else:
                    forecast = forecast_extended(series.to_numpy(dtype=float), model, horizon)[0]
                flags = _long_horizon_flags(latest, np.asarray(forecast, dtype=float))
                growth = (float(forecast[-1]) - latest) / latest * 100 if np.isfinite(forecast).all() else np.nan
                rows.append({"model": model, "history_group_id": group_id, "status": "flagged" if flags else "included", "flags": "; ".join(flags), "five_year_growth_pct": growth})
            except Exception as exc:
                rows.append({"model": model, "history_group_id": group_id, "status": "failed", "flags": "fit_failure", "error": str(exc), "five_year_growth_pct": np.nan})
    detail = pd.DataFrame(rows)
    summary_rows: list[dict[str, object]] = []
    for model in EXTENDED_METHODS:
        model_detail = detail.loc[detail["model"].eq(model)]
        valid_growth = pd.to_numeric(model_detail["five_year_growth_pct"], errors="coerce").dropna()
        flags = model_detail.loc[model_detail["status"].eq("flagged"), "flags"].fillna("")
        summary_rows.append(
            {
                "model": model,
                "model_label": MODEL_LABELS[model],
                "number_of_forecasts": int(len(model_detail)),
                "number_failed": int(model_detail["status"].eq("failed").sum()),
                "number_flagged": int(model_detail["status"].eq("flagged").sum()),
                "median_5y_growth": float(valid_growth.median()) if len(valid_growth) else np.nan,
                "p10_5y_growth": float(valid_growth.quantile(0.10)) if len(valid_growth) else np.nan,
                "p90_5y_growth": float(valid_growth.quantile(0.90)) if len(valid_growth) else np.nan,
                "min_5y_growth": float(valid_growth.min()) if len(valid_growth) else np.nan,
                "max_5y_growth": float(valid_growth.max()) if len(valid_growth) else np.nan,
                "non_positive_forecasts": int(flags.str.contains("non_positive_forecast", regex=False).sum()),
                "implausible_declines": int(flags.str.contains("implausible_decline", regex=False).sum()),
                "explosive_growth": int(flags.str.contains("explosive_growth", regex=False).sum()),
                "excessive_step_change": int(flags.str.contains("excessive_quarterly_step_change", regex=False).sum()),
            }
        )
    return pd.DataFrame(summary_rows), detail, {"horizon_quarters": horizon, "pooled_full_history": pooled_meta, "flags_are_reported_not_clipped": True}


def forecast_combination_stability(sanity_detail: pd.DataFrame) -> pd.DataFrame:
    """Summarise descriptive dispersion of long-horizon forecasts."""

    rows = []
    damped_values = pd.to_numeric(sanity_detail.loc[sanity_detail["model"].eq("damped_holt"), "five_year_growth_pct"], errors="coerce").dropna()
    damped_std = float(damped_values.std()) if len(damped_values) > 1 else np.nan
    for model in ("damped_holt", "theta", "autoets", "combination_a", "combination_b", "pooled_trend"):
        values = pd.to_numeric(sanity_detail.loc[sanity_detail["model"].eq(model), "five_year_growth_pct"], errors="coerce").dropna()
        rows.append(
            {
                "model": model,
                "model_label": MODEL_LABELS[model],
                "forecasts": int(len(values)),
                "std_5y_growth_pct": float(values.std()) if len(values) > 1 else np.nan,
                "iqr_5y_growth_pct": float(values.quantile(0.75) - values.quantile(0.25)) if len(values) else np.nan,
                "median_absolute_5y_growth_pct": float(values.abs().median()) if len(values) else np.nan,
                "p10_5y_growth_pct": float(values.quantile(0.10)) if len(values) else np.nan,
                "p90_5y_growth_pct": float(values.quantile(0.90)) if len(values) else np.nan,
                "flag_rate": float(sanity_detail.loc[sanity_detail["model"].eq(model), "status"].eq("flagged").mean()),
                "std_change_vs_damped_holt": float(values.std() - damped_std) if len(values) > 1 and pd.notna(damped_std) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def select_extended_model(metrics: pd.DataFrame, sanity_detail: pd.DataFrame, failures: pd.DataFrame) -> dict[str, object]:
    """Use a conservative multi-horizon and long-horizon decision rule."""

    work = metrics.copy()
    work["rmse_rank"] = work.groupby("horizon_quarters")["rmse"].rank(method="min")
    work["mae_rank"] = work.groupby("horizon_quarters")["mae"].rank(method="min")
    summary = work.groupby(["model", "model_label"], as_index=False).agg(
        average_rmse_rank=("rmse_rank", "mean"),
        average_mae_rank=("mae_rank", "mean"),
        best_rmse_horizons=("rmse_rank", lambda values: int((values == 1).sum())),
        best_mae_horizons=("mae_rank", lambda values: int((values == 1).sum())),
        mean_rmse=("rmse", "mean"),
        mean_mae=("mae", "mean"),
    )
    flag_rate = sanity_detail.assign(flagged=sanity_detail["status"].eq("flagged")).groupby("model", as_index=False)["flagged"].mean().rename(columns={"flagged": "long_horizon_flag_rate"})
    if failures.empty:
        failure_rate = pd.DataFrame({"model": metrics["model"].unique(), "failure_events": 0.0})
    else:
        failure_rate = failures.groupby("model").size().div(max(len(metrics), 1)).rename("failure_events").reset_index()
    summary = summary.merge(flag_rate, on="model", how="left").merge(failure_rate, on="model", how="left")
    summary[["long_horizon_flag_rate", "failure_events"]] = summary[["long_horizon_flag_rate", "failure_events"]].fillna(0)
    damped = summary.loc[summary["model"].eq("damped_holt")].iloc[0]
    alternatives = summary.loc[~summary["model"].eq("damped_holt")].copy()
    alternatives["dominates_damped"] = (
        alternatives["average_rmse_rank"].lt(damped["average_rmse_rank"])
        & alternatives["average_mae_rank"].le(damped["average_mae_rank"])
        & alternatives["long_horizon_flag_rate"].le(damped["long_horizon_flag_rate"])
        & alternatives["failure_events"].le(damped["failure_events"])
    )
    dominating = alternatives.loc[alternatives["dominates_damped"]].sort_values(["average_rmse_rank", "average_mae_rank", "long_horizon_flag_rate", "model"])
    recommended = str(dominating.iloc[0]["model"]) if not dominating.empty else "damped_holt"
    return {
        "model_summary": summary.sort_values(["average_rmse_rank", "average_mae_rank", "long_horizon_flag_rate", "model"]).reset_index(drop=True),
        "recommended_primary_model": recommended,
        "existing_damped_holt_decision_changed": recommended != "damped_holt",
        "selection_rule": "An alternative must strictly improve average RMSE rank, not worsen average MAE rank, long-horizon flag rate, or failure count across 1Q/2Q/4Q. Otherwise retain Damped Holt.",
        "q2_ranking_action": "regenerate distinct-trajectory ranking only if the recommended primary model changes; otherwise retain the existing distinct-history ranking",
    }


def run_model_extension(project_root: str | Path) -> dict[str, object]:
    """Run the extension and write only isolated model-extension outputs."""

    root = Path(project_root).resolve()
    output = root / "data" / "processed" / "sprint5" / "model_extension"
    output.mkdir(parents=True, exist_ok=True)
    history_path = root / "data" / "processed" / "homes_victoria_sa2_rents_2023_2025.csv"
    history = load_sa2_history(history_path)
    metrics, comparison, predictions, failures, groups, validation_audit = evaluate_multi_horizon(history)
    sanity, sanity_detail, sanity_audit = long_horizon_sanity_check(history)
    combinations = forecast_combination_stability(sanity_detail)
    decision = select_extended_model(metrics, sanity_detail, failures)
    metrics.to_csv(output / "multi_horizon_extended_validation.csv", index=False)
    comparison.to_csv(output / "extended_model_comparison.csv", index=False)
    mae_comparison = metrics.pivot(index=["model", "model_label"], columns="horizon_quarters", values="mae").reset_index().rename(columns={1: "1Q MAE", 2: "2Q MAE", 4: "4Q MAE"})
    mae_comparison.to_csv(output / "extended_model_comparison_mae.csv", index=False)
    predictions.to_csv(output / "extended_rolling_predictions.csv", index=False)
    failures.to_csv(output / "extended_model_failures.csv", index=False)
    groups.to_csv(output / "distinct_history_groups.csv", index=False)
    sanity.to_csv(output / "long_horizon_model_sanity.csv", index=False)
    sanity_detail.to_csv(output / "long_horizon_forecast_detail.csv", index=False)
    combinations.to_csv(output / "forecast_combination_comparison.csv", index=False)
    pd.DataFrame(validation_audit["pooled_diagnostics"]).to_csv(output / "pooled_trend_diagnostics.csv", index=False)
    decision["model_summary"].to_csv(output / "extended_model_decision_summary.csv", index=False)
    decision_json = {key: value for key, value in decision.items() if key != "model_summary"}
    audit = {
        "source_history": str(history_path),
        "expanded_sa2_rows": int(history["sa2_code"].nunique()),
        "distinct_history_trajectories": int(len(groups)),
        "duplicate_history_groups": int((groups["mapped_sa2_count"] > 1).sum()),
        "statsforecast_installed": STATSFORECAST_AVAILABLE,
        "statsmodels_installed": STATSMODELS_AVAILABLE,
        "official_statsforecast_note": "statsforecast is not installed in the checkpoint environment; local Theta and constrained non-seasonal AutoETS benchmarks are used. Optional official backend: python -m pip install statsforecast",
        "pooled_trend_note": "statsmodels is not installed; the pooled benchmark is an empirical-Bayes-style shrinkage trend, not a forced MixedLM fit.",
        "validation": validation_audit,
        "long_horizon_sanity": sanity_audit,
        "decision": decision_json,
        "ors_called": False,
        "api_calls": False,
        "external_datasets_added": False,
        "sprint1_to_sprint4_outputs_overwritten": False,
        "existing_damped_holt_results_modified": False,
        "q1_q3_modified": False,
        "twenty_quarter_backtest_claimed": False,
    }
    (output / "selected_model_decision.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return {"metrics": metrics, "comparison": comparison, "predictions": predictions, "failures": failures, "groups": groups, "sanity": sanity, "sanity_detail": sanity_detail, "combinations": combinations, "decision": decision, "audit": audit, "output": output}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run isolated Sprint 5 forecasting model extensions.")
    parser.add_argument("project_root", type=Path)
    args = parser.parse_args()
    result = run_model_extension(args.project_root)
    print(json.dumps({"output": str(result["output"]), "decision": result["audit"]["decision"], "validation": result["audit"]["validation"]}, indent=2, default=str))

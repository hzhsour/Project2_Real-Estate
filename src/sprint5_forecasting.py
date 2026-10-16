"""Sprint 5 rolling-origin forecasting using saved local rental history only.

The module deliberately avoids external requests and does not modify Sprint 1--4
outputs.  The Holt implementations are small deterministic additive baselines;
they avoid introducing a new dependency for a short 11-quarter panel.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error

from .sa2_context import normalise_sa2_code
from .sprint4_forecasting import load_sa2_history


METHODS = ("naive", "drift", "linear_trend", "holt_trend", "damped_holt")
DEFAULT_MIN_TRAIN = 6
DEFAULT_HORIZON = 1
DEFAULT_FORECAST_YEARS = 5
DEFAULT_MAX_ABS_GROWTH_PCT = 200.0


def _naive(values: np.ndarray, horizon: int) -> np.ndarray:
    return np.repeat(values[-1], horizon).astype(float)


def _drift(values: np.ndarray, horizon: int) -> np.ndarray:
    slope = (values[-1] - values[0]) / max(len(values) - 1, 1)
    return values[-1] + np.arange(1, horizon + 1, dtype=float) * slope


def _linear_trend(values: np.ndarray, horizon: int) -> np.ndarray:
    x = np.arange(len(values), dtype=float)
    slope, intercept = np.polyfit(x, values, 1)
    return intercept + slope * np.arange(len(values), len(values) + horizon, dtype=float)


def _holt(values: np.ndarray, horizon: int, *, damped: bool = False, alpha: float = 0.8, beta: float = 0.2, phi: float = 0.9) -> np.ndarray:
    if len(values) == 1:
        return _naive(values, horizon)
    level = float(values[0])
    trend = float(values[1] - values[0])
    for value in values[1:]:
        previous_level = level
        previous_trend = trend
        effective_trend = phi * previous_trend if damped else previous_trend
        level = alpha * float(value) + (1 - alpha) * (level + effective_trend)
        trend = beta * (level - previous_level) + (1 - beta) * effective_trend
    if damped:
        horizon_weights = np.asarray([sum(phi**step for step in range(1, h + 1)) for h in range(1, horizon + 1)])
        return level + horizon_weights * trend
    return level + np.arange(1, horizon + 1, dtype=float) * trend


def forecast_values(values: list[float] | np.ndarray, method: str, horizon: int) -> np.ndarray:
    """Forecast a finite series with one of the five interpretable baselines."""

    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Forecast input must be a non-empty finite one-dimensional series.")
    if method == "naive":
        return _naive(values, horizon)
    if method == "drift":
        return _drift(values, horizon)
    if method == "linear_trend":
        return _linear_trend(values, horizon)
    if method == "holt_trend":
        return _holt(values, horizon)
    if method == "damped_holt":
        return _holt(values, horizon, damped=True)
    raise ValueError(f"Unknown forecasting method: {method}")


def rolling_folds(periods: list[pd.Timestamp], *, min_train: int = DEFAULT_MIN_TRAIN, horizon: int = DEFAULT_HORIZON) -> list[dict[str, object]]:
    """Create chronological expanding-window folds shared by every model."""

    periods = sorted(pd.Timestamp(period) for period in periods)
    folds = []
    for train_end_index in range(min_train - 1, len(periods) - horizon):
        folds.append({
            "fold": len(folds) + 1,
            "train_end": periods[train_end_index],
            "validation_periods": periods[train_end_index + 1 : train_end_index + 1 + horizon],
        })
    if not folds:
        raise ValueError("Not enough historical periods for the requested rolling folds.")
    return folds


def _series_by_area(history: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        str(code): group.sort_values("period_end").set_index("period_end")["median_weekly_rent_aud"].dropna()
        for code, group in history.groupby("sa2_code")
    }


def _area_class_map(history: pd.DataFrame, geography: pd.DataFrame | None = None) -> pd.DataFrame:
    """Label regional versus inner/middle-outer Melbourne for stability QA."""

    areas = history[["sa2_code", "sa2_name"]].drop_duplicates("sa2_code").copy()
    if geography is not None and {"sa2_code", "gcc_name"}.issubset(geography.columns):
        geo = geography[["sa2_code", "gcc_name"]].copy()
        geo["sa2_code"] = normalise_sa2_code(geo["sa2_code"])
        geo = geo.drop_duplicates("sa2_code")
        areas = areas.merge(geo, on="sa2_code", how="left")
    else:
        areas["gcc_name"] = pd.NA
    inner_tokens = ("Melbourne CBD", "Docklands", "Carlton", "East Melbourne", "West Melbourne", "Southbank", "North Melbourne")
    areas["geography_scope"] = np.where(
        areas["gcc_name"].eq("Rest of Vic."),
        "regional Victoria",
        np.where(areas["sa2_name"].str.contains("|".join(inner_tokens), case=False, na=False), "inner Melbourne", "middle/outer Melbourne"),
    )
    return areas[["sa2_code", "sa2_name", "geography_scope"]]


def rolling_validate(
    history: pd.DataFrame,
    *,
    geography: pd.DataFrame | None = None,
    min_train: int = DEFAULT_MIN_TRAIN,
    horizon: int = DEFAULT_HORIZON,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Run identical expanding-window folds for all methods and areas."""

    periods = sorted(history["period_end"].dropna().unique())
    folds = rolling_folds(periods, min_train=min_train, horizon=horizon)
    series_map = _series_by_area(history)
    area_map = _area_class_map(history, geography)
    predictions: list[dict[str, object]] = []
    for fold in folds:
        train_end = fold["train_end"]
        validation_periods = fold["validation_periods"]
        for code, series in series_map.items():
            train = series.loc[series.index <= train_end]
            test = series.loc[series.index.isin(validation_periods)]
            if len(train) < min_train or len(test) != horizon:
                continue
            area_row = area_map.loc[area_map["sa2_code"].eq(code)]
            if area_row.empty:
                continue
            for method in METHODS:
                predicted = forecast_values(train.to_numpy(), method, horizon)
                for period, actual, prediction in zip(test.index, test.to_numpy(dtype=float), predicted):
                    predictions.append({
                        "fold": fold["fold"],
                        "train_end": train_end.strftime("%Y-%m-%d"),
                        "validation_period": pd.Timestamp(period).strftime("%Y-%m-%d"),
                        "sa2_code": code,
                        "sa2_name": area_row.iloc[0]["sa2_name"],
                        "geography_scope": area_row.iloc[0]["geography_scope"],
                        "model": method,
                        "actual_rent": float(actual),
                        "predicted_rent": float(prediction),
                        "absolute_error": abs(float(actual) - float(prediction)),
                    })
    prediction_frame = pd.DataFrame(predictions)
    if prediction_frame.empty:
        raise ValueError("Rolling validation produced no forecasts.")
    grouped = []
    for scope, group in [("all areas", prediction_frame), *prediction_frame.groupby("geography_scope")]:
        for method, method_group in group.groupby("model"):
            actual = method_group["actual_rent"]
            predicted = method_group["predicted_rent"]
            grouped.append({
                "geography_scope": scope,
                "model": method,
                "rolling_MAE_AUD_week": float(mean_absolute_error(actual, predicted)),
                "rolling_RMSE_AUD_week": float(np.sqrt(mean_squared_error(actual, predicted))),
                "median_absolute_error_AUD_week": float(median_absolute_error(actual, predicted)),
                "forecasts_evaluated": int(len(method_group)),
                "areas_evaluated": int(method_group["sa2_code"].nunique()),
                "folds": int(method_group["fold"].nunique()),
                "min_train": min_train,
                "horizon": horizon,
            })
    metrics = pd.DataFrame(grouped).sort_values(["geography_scope", "rolling_RMSE_AUD_week", "rolling_MAE_AUD_week", "model"]).reset_index(drop=True)
    audit = {
        "min_training_periods": min_train,
        "validation_horizon_periods": horizon,
        "rolling_folds": len(folds),
        "folds": [{"fold": f["fold"], "train_end": f["train_end"].strftime("%Y-%m-%d"), "validation_periods": [p.strftime("%Y-%m-%d") for p in f["validation_periods"]]} for f in folds],
        "areas_evaluated_overall": int(prediction_frame["sa2_code"].nunique()),
        "forecasts_evaluated_overall": int(len(prediction_frame)),
        "chronological": True,
    }
    return metrics, prediction_frame, audit


def _sanity_check(area: pd.Series, forecast: np.ndarray, *, max_abs_growth_pct: float) -> str:
    latest = float(area["latest_observed_rent"])
    if not np.isfinite(forecast).all():
        return "non_finite_forecast"
    if (forecast <= 0).any():
        return "non_positive_forecast"
    growth = (float(forecast[-1]) - latest) / latest * 100
    if abs(growth) > max_abs_growth_pct:
        return "absolute_five_year_growth_over_threshold"
    return ""


def _full_forecasts(
    history: pd.DataFrame,
    *,
    selected_model: str,
    forecast_years: int,
    max_abs_growth_pct: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    horizon = forecast_years * 4
    periods = pd.date_range(history["period_end"].max() + pd.offsets.QuarterEnd(1), periods=horizon, freq="QE")
    names = history[["sa2_code", "sa2_name"]].drop_duplicates("sa2_code").set_index("sa2_code")["sa2_name"]
    rows, exclusions = [], []
    for code in sorted(history["sa2_code"].dropna().unique()):
        series = history.loc[history["sa2_code"].eq(code)].sort_values("period_end").set_index("period_end")["median_weekly_rent_aud"].dropna()
        latest = float(series.iloc[-1])
        row = {"sa2_code": code, "sa2_name": names.loc[code], "history_observations": int(len(series)), "latest_observed_period": series.index[-1].strftime("%Y-%m-%d"), "latest_observed_rent": latest, "forecast_1y_period": periods[3].strftime("%Y-%m-%d"), "forecast_3y_period": periods[11].strftime("%Y-%m-%d"), "forecast_5y_period": periods[19].strftime("%Y-%m-%d")}
        for method in METHODS:
            forecast = forecast_values(series.to_numpy(), method, horizon)
            reason = _sanity_check(pd.Series(row), forecast, max_abs_growth_pct=max_abs_growth_pct)
            exclusions.append({"sa2_code": code, "sa2_name": names.loc[code], "model": method, "sanity_status": "included" if not reason else "excluded", "exclusion_reason": reason})
            if method == selected_model:
                row["forecast_model"] = method
                row["forecast_1y_rent"] = float(forecast[3])
                row["forecast_3y_rent"] = float(forecast[11])
                row["forecast_5y_rent"] = float(forecast[19])
                row["forecast_sanity_status"] = "included" if not reason else "excluded"
                row["forecast_exclusion_reason"] = reason
            if method == "naive":
                row["naive_forecast_1y_rent"] = float(forecast[3])
                row["naive_forecast_3y_rent"] = float(forecast[11])
                row["naive_forecast_5y_rent"] = float(forecast[19])
        if row.get("forecast_sanity_status") == "included":
            for prefix in ("", "naive_"):
                for horizon_label in ("1y", "3y", "5y"):
                    rent_column = f"{prefix}forecast_{horizon_label}_rent"
                    growth_column = f"{prefix}predicted_growth_{horizon_label}_pct"
                    row[growth_column] = (row[rent_column] - latest) / latest * 100
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(exclusions)


def select_model(metrics: pd.DataFrame, exclusions: pd.DataFrame) -> tuple[str, dict[str, object]]:
    """Select by overall rolling RMSE, then MAE and sanity-exclusion rate."""

    overall = metrics.loc[metrics["geography_scope"].eq("all areas")].copy()
    exclusion_rate = exclusions.assign(excluded=exclusions["sanity_status"].eq("excluded")).groupby("model").excluded.mean()
    overall["exclusion_rate"] = overall["model"].map(exclusion_rate).fillna(1.0)
    ranked = overall.sort_values(["rolling_RMSE_AUD_week", "rolling_MAE_AUD_week", "exclusion_rate", "model"]).reset_index(drop=True)
    selected = str(ranked.iloc[0]["model"])
    audit = {"selection_rule": "Lowest overall rolling RMSE, then MAE, then lowest sanity-exclusion rate; interpretability and long-horizon sanity are reported separately.", "ranked_models": ranked[["model", "rolling_RMSE_AUD_week", "rolling_MAE_AUD_week", "exclusion_rate"]].to_dict("records"), "selected_model": selected}
    return selected, audit


def run(project_root: str | Path, *, min_train: int = DEFAULT_MIN_TRAIN, horizon: int = DEFAULT_HORIZON, forecast_years: int = DEFAULT_FORECAST_YEARS, max_abs_growth_pct: float = DEFAULT_MAX_ABS_GROWTH_PCT) -> dict[str, object]:
    """Run Sprint 5 rolling validation and write only ``data/processed/sprint5``."""

    root = Path(project_root).resolve()
    output = root / "data" / "processed" / "sprint5"
    output.mkdir(parents=True, exist_ok=True)
    history_path = root / "data" / "processed" / "homes_victoria_sa2_rents_2023_2025.csv"
    listing_path = root / "data" / "processed" / "victoria_listings_with_accessibility_week8_amenities.csv"
    history = load_sa2_history(history_path)
    geography = pd.read_csv(listing_path, usecols=["sa2_code", "gcc_name"], dtype={"sa2_code": "string"}, low_memory=False)
    metrics, predictions, rolling_audit = rolling_validate(history, geography=geography, min_train=min_train, horizon=horizon)
    # Full-history sanity checks are performed before the selection is finalised.
    _, exclusions = _full_forecasts(history, selected_model="naive", forecast_years=forecast_years, max_abs_growth_pct=max_abs_growth_pct)
    selected_model, selection_audit = select_model(metrics, exclusions)
    forecasts, exclusions = _full_forecasts(history, selected_model=selected_model, forecast_years=forecast_years, max_abs_growth_pct=max_abs_growth_pct)
    valid_selected = forecasts["forecast_sanity_status"].eq("included")
    differentiated = valid_selected & forecasts["predicted_growth_5y_pct"].ne(forecasts["predicted_growth_5y_pct"].iloc[0])
    meaningful_ranking = bool(differentiated.any()) and selected_model != "naive"
    if meaningful_ranking:
        ranking = forecasts.loc[valid_selected].nlargest(10, "predicted_growth_5y_pct")
        ranking.to_csv(output / "preliminary_top10_sa2.csv", index=False)
    else:
        ranking = pd.DataFrame()
        old_ranking = output / "preliminary_top10_sa2.csv"
        if old_ranking.exists():
            old_ranking.unlink()
    metrics.to_csv(output / "rolling_forecast_validation.csv", index=False)
    forecasts.to_csv(output / "sprint5_rent_forecasts.csv", index=False)
    exclusions.to_csv(output / "forecast_exclusions.csv", index=False)
    metrics.loc[metrics["geography_scope"].eq("all areas")].to_csv(output / "forecast_model_comparison.csv", index=False)
    audit = {
        "forecasting_granularity": "SA2",
        "history_periods": int(history["period_end"].nunique()),
        "history_start": history["period_end"].min().strftime("%Y-%m-%d"),
        "history_end": history["period_end"].max().strftime("%Y-%m-%d"),
        "min_train": min_train,
        "validation_horizon": horizon,
        "forecast_years": forecast_years,
        "rolling_validation": rolling_audit,
        "selection": selection_audit,
        "selected_model": selected_model,
        "successful_selected_forecasts": int(valid_selected.sum()),
        "meaningful_differentiated_growth_ranking": meaningful_ranking,
        "top10_status": "produced" if meaningful_ranking else "not produced: current validated models do not yet support a reliable differentiated growth ranking",
        "sanity_threshold_absolute_five_year_growth_pct": max_abs_growth_pct,
        "sanity_excluded_selected_forecasts": int((~valid_selected).sum()),
        "ors_called": False,
        "api_calls": False,
        "sprint1_to_sprint4_outputs_overwritten": False,
    }
    (output / "sprint5_audit.json").write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
    return {"history": history, "metrics": metrics, "predictions": predictions, "forecasts": forecasts, "exclusions": exclusions, "selected_model": selected_model, "ranking": ranking, "audit": audit, "output": output}

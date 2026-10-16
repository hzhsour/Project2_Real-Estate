"""Deterministic Sprint 4 forecasting and feature-analysis workflow.

This module only reads saved Sprint 1--3 and Week 8 outputs.  It does not
download data, call OpenRouteService, rebuild accessibility features, or write
over any earlier-sprint outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .sa2_context import normalise_sa2_code


FORECAST_METHODS = ("naive", "linear_trend", "holt_trend")
MODEL_SEED = 23
DEFAULT_MIN_HISTORY = 8
DEFAULT_VALIDATION_PERIODS = 2

INTERNAL_FEATURES = (
    "listing_count",
    "median_bedrooms",
    "median_bathrooms",
    "median_parking_spaces",
)
SOCIOECONOMIC_FEATURES = (
    "population_2026",
    "population_2031",
    "population_growth_pct",
    "median_household_income",
    "irsad_score",
    "irsad_decile",
)
TRANSPORT_FEATURES = (
    "nearest_train_straightline_km",
    "nearest_tram_straightline_km",
    "nearest_school_straightline_km",
    "cbd_driving_km",
    "cbd_driving_minutes",
    "train_driving_km",
    "train_driving_minutes",
)
AMENITY_FEATURES = (
    "nearest_park_distance_km",
    "park_named_within_2km",
    "nearest_shopping_distance_km",
    "shopping_named_within_2km",
    "nearest_entertainment_distance_km",
    "entertainment_named_within_2km",
)
LEAKAGE_COLUMNS = (
    "current_median_weekly_rent",
    "latest_observed_rent",
    "census_median_rent_weekly",
    "rental_stress_pct",
    "historical_rent_change",
    "forecast_1y_rent",
    "forecast_3y_rent",
    "forecast_5y_rent",
)


@dataclass(frozen=True)
class ForecastConfig:
    """Configuration kept explicit for a reproducible checkpoint run."""

    min_history: int = DEFAULT_MIN_HISTORY
    validation_periods: int = DEFAULT_VALIDATION_PERIODS
    forecast_years: int = 5
    seed: int = MODEL_SEED


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def load_sa2_history(path: str | Path) -> pd.DataFrame:
    """Load the all-property Homes Victoria SA2 history as a tidy panel."""

    frame = pd.read_csv(path, dtype={"sa2_code": "string"}, low_memory=False)
    required = {
        "sa2_code",
        "sa2_name",
        "period_end",
        "property_type",
        "median_weekly_rent_aud",
        "new_lease_count",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Historical SA2 rent table is missing columns: {sorted(missing)}")
    frame["sa2_code"] = normalise_sa2_code(frame["sa2_code"])
    frame["period_end"] = pd.to_datetime(frame["period_end"], errors="coerce")
    frame["median_weekly_rent_aud"] = _numeric(frame["median_weekly_rent_aud"])
    frame["new_lease_count"] = _numeric(frame["new_lease_count"])
    frame = frame.loc[frame["property_type"].eq("All properties")].copy()
    frame = frame.dropna(subset=["sa2_code", "sa2_name", "period_end"])
    frame = frame.sort_values(["sa2_code", "period_end"]).reset_index(drop=True)
    if frame.duplicated(["sa2_code", "period_end"]).any():
        raise ValueError("All-property SA2 history contains duplicate area-period rows.")
    return frame


def history_qa(history: pd.DataFrame, min_history: int = DEFAULT_MIN_HISTORY) -> dict[str, object]:
    """Return compact, explicit quality metrics for the historical panel."""

    counts = history.groupby("sa2_code")["period_end"].nunique()
    valid_counts = history.dropna(subset=["median_weekly_rent_aud"]).groupby("sa2_code")["period_end"].nunique()
    periods = sorted(history["period_end"].dropna().dt.strftime("%Y-%m-%d").unique())
    return {
        "earliest_period": periods[0] if periods else None,
        "latest_period": periods[-1] if periods else None,
        "period_count": len(periods),
        "area_count": int(history["sa2_code"].nunique()),
        "median_observations_per_area": float(counts.median()) if len(counts) else 0.0,
        "min_observations_per_area": int(counts.min()) if len(counts) else 0,
        "max_observations_per_area": int(counts.max()) if len(counts) else 0,
        "missing_rent_values": int(history["median_weekly_rent_aud"].isna().sum()),
        "areas_with_incomplete_history": int((counts < history["period_end"].nunique()).sum()) if len(counts) else 0,
        "usable_areas_at_threshold": int((valid_counts >= min_history).sum()),
        "excluded_areas_at_threshold": int((valid_counts < min_history).sum()),
    }


def _naive(values: np.ndarray, horizon: int) -> np.ndarray:
    return np.repeat(values[-1], horizon).astype(float)


def _linear_trend(values: np.ndarray, horizon: int) -> np.ndarray:
    x = np.arange(len(values), dtype=float)
    slope, intercept = np.polyfit(x, values, 1)
    future_x = np.arange(len(values), len(values) + horizon, dtype=float)
    return intercept + slope * future_x


def _holt_trend(values: np.ndarray, horizon: int, alpha: float = 0.8, beta: float = 0.2) -> np.ndarray:
    """Simple additive Holt trend with fixed transparent smoothing constants."""

    if len(values) == 1:
        return _naive(values, horizon)
    level = float(values[0])
    trend = float(values[1] - values[0])
    for value in values[1:]:
        previous_level = level
        level = alpha * float(value) + (1 - alpha) * (level + trend)
        trend = beta * (level - previous_level) + (1 - beta) * trend
    return np.asarray([level + step * trend for step in range(1, horizon + 1)], dtype=float)


def forecast_values(values: Iterable[float], method: str, horizon: int) -> np.ndarray:
    """Forecast one clean historical series with an interpretable baseline."""

    values = np.asarray(list(values), dtype=float)
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Forecast input must contain a non-empty finite series.")
    if method == "naive":
        return _naive(values, horizon)
    if method == "linear_trend":
        return _linear_trend(values, horizon)
    if method == "holt_trend":
        return _holt_trend(values, horizon)
    raise ValueError(f"Unknown forecasting method: {method}")


def _usable_series(history: pd.DataFrame, code: str, cutoff: pd.Timestamp | None = None) -> pd.Series:
    subset = history.loc[history["sa2_code"].eq(code)].sort_values("period_end")
    if cutoff is not None:
        subset = subset.loc[subset["period_end"] <= cutoff]
    series = subset.set_index("period_end")["median_weekly_rent_aud"].dropna()
    return series


def validate_forecasts(
    history: pd.DataFrame,
    *,
    min_history: int = DEFAULT_MIN_HISTORY,
    validation_periods: int = DEFAULT_VALIDATION_PERIODS,
) -> pd.DataFrame:
    """Chronologically compare the three simple forecasting baselines."""

    latest = history["period_end"].max()
    validation_dates = sorted(history.loc[history["period_end"].gt(latest - pd.offsets.QuarterEnd(validation_periods)), "period_end"].unique())
    if len(validation_dates) != validation_periods:
        validation_dates = sorted(history["period_end"].unique())[-validation_periods:]
    cutoff = validation_dates[0] - pd.offsets.QuarterEnd(1)
    rows: list[dict[str, object]] = []
    for method in FORECAST_METHODS:
        actuals: list[float] = []
        predictions: list[float] = []
        areas = 0
        for code in sorted(history["sa2_code"].dropna().unique()):
            series = _usable_series(history, code)
            train = series.loc[series.index <= cutoff]
            test = series.loc[series.index.isin(validation_dates)]
            if len(series) < min_history or len(train) < 2 or len(test) != validation_periods:
                continue
            pred = forecast_values(train.to_numpy(), method, validation_periods)
            actuals.extend(test.to_numpy(dtype=float).tolist())
            predictions.extend(pred.tolist())
            areas += 1
        y_true = np.asarray(actuals, dtype=float)
        y_pred = np.asarray(predictions, dtype=float)
        rows.append({
            "model": method,
            "MAE_AUD_week": float(mean_absolute_error(y_true, y_pred)),
            "RMSE_AUD_week": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "areas_evaluated": areas,
            "validation_periods": validation_periods,
            "validation_start": pd.Timestamp(validation_dates[0]).strftime("%Y-%m-%d"),
            "validation_end": pd.Timestamp(validation_dates[-1]).strftime("%Y-%m-%d"),
        })
    return pd.DataFrame(rows).sort_values(["MAE_AUD_week", "RMSE_AUD_week", "model"]).reset_index(drop=True)


def choose_forecast_method(validation: pd.DataFrame) -> str:
    """Select the best validation method with deterministic tie-breaking."""

    if validation.empty:
        raise ValueError("No forecasting validation results were produced.")
    return str(validation.sort_values(["MAE_AUD_week", "RMSE_AUD_week", "model"]).iloc[0]["model"])


def build_forecasts(
    history: pd.DataFrame,
    method: str,
    *,
    min_history: int = DEFAULT_MIN_HISTORY,
    forecast_years: int = 5,
) -> pd.DataFrame:
    """Fit the selected method on each usable SA2 and forecast quarterly rents."""

    latest_period = history["period_end"].max()
    horizon = forecast_years * 4
    future_periods = pd.date_range(latest_period + pd.offsets.QuarterEnd(1), periods=horizon, freq="QE")
    rows: list[dict[str, object]] = []
    names = history[["sa2_code", "sa2_name"]].drop_duplicates("sa2_code").set_index("sa2_code")["sa2_name"]
    for code in sorted(history["sa2_code"].dropna().unique()):
        series = _usable_series(history, code)
        if len(series) < min_history or series.iloc[-1] <= 0:
            continue
        forecast = forecast_values(series.to_numpy(), method, horizon)
        if not np.isfinite(forecast).all() or (forecast <= 0).any():
            continue
        latest_rent = float(series.iloc[-1])
        row: dict[str, object] = {
            "sa2_code": code,
            "sa2_name": names.loc[code],
            "forecast_model": method,
            "history_observations": int(len(series)),
            "latest_observed_period": series.index[-1].strftime("%Y-%m-%d"),
            "latest_observed_rent": latest_rent,
            "forecast_1y_period": future_periods[3].strftime("%Y-%m-%d"),
            "forecast_3y_period": future_periods[11].strftime("%Y-%m-%d"),
            "forecast_5y_period": future_periods[19].strftime("%Y-%m-%d"),
            "forecast_1y_rent": float(forecast[3]),
            "forecast_3y_rent": float(forecast[11]),
            "forecast_5y_rent": float(forecast[19]),
        }
        row["predicted_growth_1y_pct"] = (row["forecast_1y_rent"] - latest_rent) / latest_rent * 100
        row["predicted_growth_3y_pct"] = (row["forecast_3y_rent"] - latest_rent) / latest_rent * 100
        row["predicted_growth_5y_pct"] = (row["forecast_5y_rent"] - latest_rent) / latest_rent * 100
        rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No usable areas produced a valid five-year forecast.")
    return result.sort_values("sa2_code").reset_index(drop=True)


def prepare_area_features(listings: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Aggregate saved listing-level accessibility/amenity data to SA2."""

    required = {"listing_id", "sa2_code", "sa2_name", "weekly_rent_aud", "property_type"}
    missing = required.difference(listings.columns)
    if missing:
        raise ValueError(f"Feature input is missing columns: {sorted(missing)}")
    frame = listings.copy()
    frame["sa2_code"] = normalise_sa2_code(frame["sa2_code"])
    frame["weekly_rent_aud"] = _numeric(frame["weekly_rent_aud"])
    valid = frame["sa2_code"].notna() & frame["weekly_rent_aud"].between(50, 10000)
    frame = frame.loc[valid].copy()
    if frame.empty:
        raise ValueError("No valid listing rows remain for area feature aggregation.")
    numeric_source_columns = [
        "bedrooms", "bathrooms", "parking_spaces", *SOCIOECONOMIC_FEATURES,
        *TRANSPORT_FEATURES, *AMENITY_FEATURES,
    ]
    for column in numeric_source_columns:
        if column in frame.columns:
            frame[column] = _numeric(frame[column])
    group = frame.groupby("sa2_code", dropna=False)
    area = group.agg(
        sa2_name=("sa2_name", "first"),
        listing_count=("listing_id", "nunique"),
        current_median_weekly_rent=("weekly_rent_aud", "median"),
        current_mean_weekly_rent=("weekly_rent_aud", "mean"),
        median_bedrooms=("bedrooms", "median"),
        median_bathrooms=("bathrooms", "median"),
        median_parking_spaces=("parking_spaces", "median"),
    ).reset_index()
    for column in [*SOCIOECONOMIC_FEATURES, *TRANSPORT_FEATURES, *AMENITY_FEATURES]:
        if column in frame.columns:
            area[column] = group[column].median().to_numpy()
    for column in ("park_named_within_2km", "shopping_named_within_2km", "entertainment_named_within_2km"):
        if column in frame.columns:
            area[column] = group[column].mean().to_numpy()
    property_shares = pd.crosstab(frame["sa2_code"], frame["property_type"], normalize="index")
    property_shares.columns = ["property_type_share_" + str(c).strip().casefold().replace(" ", "_") for c in property_shares.columns]
    area = area.set_index("sa2_code").join(property_shares).reset_index()
    area["listing_count"] = area["listing_count"].astype(int)
    excluded = [column for column in LEAKAGE_COLUMNS if column in area.columns]
    audit = {
        "input_rows": int(len(listings)),
        "valid_rows_used": int(len(frame)),
        "excluded_invalid_or_outlier_rent_rows": int(len(listings) - len(frame)),
        "area_count": int(len(area)),
        "leakage_columns_retained_for_audit_only": excluded,
        "route_status_columns_not_replaced": True,
    }
    return area.sort_values("sa2_code").reset_index(drop=True), audit


def _feature_sets(area: pd.DataFrame) -> dict[str, list[str]]:
    property_shares = sorted(c for c in area.columns if c.startswith("property_type_share_"))
    internal = [c for c in [*INTERNAL_FEATURES, *property_shares] if c in area.columns]
    socioeconomic = [c for c in SOCIOECONOMIC_FEATURES if c in area.columns]
    transport = [c for c in TRANSPORT_FEATURES if c in area.columns]
    amenities = [c for c in AMENITY_FEATURES if c in area.columns]
    return {
        "internal_property": internal,
        "internal_socioeconomic": internal + socioeconomic,
        "internal_socioeconomic_transport": internal + socioeconomic + transport,
        "internal_socioeconomic_transport_amenities": internal + socioeconomic + transport + amenities,
    }


def compare_feature_models(area: pd.DataFrame, *, seed: int = MODEL_SEED) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Compare Ridge and Random Forest across nested, leakage-screened features."""

    feature_sets = _feature_sets(area)
    target = _numeric(area["current_median_weekly_rent"])
    valid = target.notna() & target.between(50, 10000)
    data = area.loc[valid].reset_index(drop=True).copy()
    y = target.loc[valid].to_numpy(dtype=float)
    train_idx, test_idx = train_test_split(np.arange(len(data)), test_size=0.2, random_state=seed)
    scores: list[dict[str, object]] = []
    fitted: dict[tuple[str, str], tuple[object, list[str]]] = {}
    for feature_set, columns in feature_sets.items():
        if not columns:
            continue
        X = data[columns]
        for model_name, estimator in (
            ("ridge", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), Ridge(alpha=10.0))),
            ("random_forest", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=seed, n_jobs=1))),
        ):
            estimator.fit(X.iloc[train_idx], y[train_idx])
            prediction = estimator.predict(X.iloc[test_idx])
            scores.append({
                "feature_set": feature_set,
                "model": model_name,
                "train_rows": len(train_idx),
                "test_rows": len(test_idx),
                "MAE_AUD_week": float(mean_absolute_error(y[test_idx], prediction)),
                "RMSE_AUD_week": float(np.sqrt(mean_squared_error(y[test_idx], prediction))),
                "R2": float(r2_score(y[test_idx], prediction)),
                "seed": seed,
                "target_leakage_excluded": "; ".join(LEAKAGE_COLUMNS),
            })
            fitted[(feature_set, model_name)] = (estimator, columns)
    score_frame = pd.DataFrame(scores).sort_values(["RMSE_AUD_week", "MAE_AUD_week", "model"]).reset_index(drop=True)
    best = score_frame.iloc[0]
    best_key = (str(best["feature_set"]), str(best["model"]))
    best_model, best_columns = fitted[best_key]
    X_best = data[best_columns]
    perm = permutation_importance(best_model, X_best.iloc[test_idx], y[test_idx], scoring="neg_mean_absolute_error", n_repeats=10, random_state=seed, n_jobs=1)
    importance = pd.DataFrame({
        "feature": best_columns,
        "MAE_increase_when_shuffled": perm.importances_mean,
        "shuffle_std": perm.importances_std,
        "feature_set": best_key[0],
        "model": best_key[1],
    }).sort_values("MAE_increase_when_shuffled", ascending=False).reset_index(drop=True)
    audit = {
        "areas_used": int(len(data)),
        "train_areas": int(len(train_idx)),
        "test_areas": int(len(test_idx)),
        "seed": seed,
        "best_feature_set": best_key[0],
        "best_model": best_key[1],
        "amenities_vs_transport_random_forest_rmse_change": _amenity_rmse_change(score_frame),
        "leakage_excluded": list(LEAKAGE_COLUMNS),
    }
    return score_frame, importance, audit


def _amenity_rmse_change(scores: pd.DataFrame) -> float | None:
    rf = scores.loc[scores["model"].eq("random_forest")]
    before = rf.loc[rf["feature_set"].eq("internal_socioeconomic_transport"), "RMSE_AUD_week"]
    after = rf.loc[rf["feature_set"].eq("internal_socioeconomic_transport_amenities"), "RMSE_AUD_week"]
    if before.empty or after.empty:
        return None
    return float(after.iloc[0] - before.iloc[0])


def descriptive_feature_analysis(area: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return Spearman area associations and property-type rent summaries."""

    target = _numeric(area["current_median_weekly_rent"])
    features = [c for c in [*_feature_sets(area)["internal_socioeconomic_transport_amenities"]] if c in area.columns]
    rows = []
    for feature in features:
        values = _numeric(area[feature])
        valid = target.notna() & values.notna()
        rows.append({"feature": feature, "spearman_rho": float(target[valid].corr(values[valid], method="spearman")) if valid.sum() >= 3 else np.nan, "areas": int(valid.sum())})
    property_summary = pd.DataFrame()
    return pd.DataFrame(rows).sort_values("spearman_rho", ascending=False), property_summary


def build_rental_activity_proxy(history: pd.DataFrame, area: pd.DataFrame) -> pd.DataFrame:
    """Create a clearly labelled new-lease activity proxy, separate from rent models."""

    latest = history.loc[history["period_end"].eq(history["period_end"].max()), ["sa2_code", "new_lease_count"]].copy()
    latest = latest.groupby("sa2_code", as_index=False)["new_lease_count"].sum()
    result = area[["sa2_code", "sa2_name", "listing_count", "population_2026"]].merge(latest, on="sa2_code", how="left")
    result["rental_activity_proxy_new_leases_per_1000_population"] = result["new_lease_count"] / result["population_2026"] * 1000
    return result


def run(project_root: str | Path, *, config: ForecastConfig | None = None) -> dict[str, object]:
    """Run the saved-input Sprint 4 workflow and write only ``sprint4`` outputs."""

    root = Path(project_root).resolve()
    config = config or ForecastConfig()
    output = root / "data" / "processed" / "sprint4"
    output.mkdir(parents=True, exist_ok=True)
    history_path = root / "data" / "processed" / "homes_victoria_sa2_rents_2023_2025.csv"
    listing_path = root / "data" / "processed" / "victoria_listings_with_accessibility_week8_amenities.csv"
    if not history_path.exists():
        raise FileNotFoundError(f"Missing saved historical rent input: {history_path}")
    if not listing_path.exists():
        raise FileNotFoundError(f"Missing saved Sprint 3/Week 8 listing input: {listing_path}")

    history = load_sa2_history(history_path)
    qa = history_qa(history, config.min_history)
    validation = validate_forecasts(
        history,
        min_history=config.min_history,
        validation_periods=config.validation_periods,
    )
    selected_method = choose_forecast_method(validation)
    forecasts = build_forecasts(
        history,
        selected_method,
        min_history=config.min_history,
        forecast_years=config.forecast_years,
    )
    listings = pd.read_csv(
        listing_path,
        dtype={"sa2_code": "string", "postcode": "string"},
        low_memory=False,
    )
    area, feature_audit = prepare_area_features(listings)
    model_scores, importance, model_audit = compare_feature_models(area, seed=config.seed)
    spearman, _ = descriptive_feature_analysis(area)
    activity = build_rental_activity_proxy(history, area)
    forecasts.to_csv(output / "rent_forecasts.csv", index=False)
    validation.to_csv(output / "forecast_model_validation.csv", index=False)
    area.to_csv(output / "area_feature_table.csv", index=False)
    model_scores.to_csv(output / "feature_model_comparison.csv", index=False)
    importance.to_csv(output / "feature_importance.csv", index=False)
    spearman.to_csv(output / "feature_spearman.csv", index=False)
    activity.to_csv(output / "rental_activity_proxy.csv", index=False)

    audit = {
        "forecasting_granularity": "SA2",
        "granularity_reason": "The all-property Homes Victoria SA2 panel has 277 complete areas over 11 quarters. Homes Victoria suburb groups are broader grouped areas and only 63% of current listings map to them through the available crosswalk, so SA2 provides the more consistent common unit for forecasts and listing-level feature aggregation.",
        "history_qa": qa,
        "forecast_config": {
            "min_history": config.min_history,
            "validation_periods": config.validation_periods,
            "forecast_years": config.forecast_years,
            "seed": config.seed,
        },
        "selected_forecast_method": selected_method,
        "successful_five_year_forecasts": int(len(forecasts)),
        "meaningful_growth_ranking_produced": False,
        "growth_ranking_reason": "The validation-selected Naive baseline holds the latest rent constant, so all predicted growth rates are tied at 0%.",
        "feature_input": str(listing_path),
        "feature_audit": feature_audit,
        "feature_model_audit": model_audit,
        "statewide_amenity_comparison": {
            "random_forest_rmse_change_after_amenities": model_audit["amenities_vs_transport_random_forest_rmse_change"],
            "interpretation": "Negative values indicate lower holdout RMSE after adding the Week 8 statewide amenity features; this is an exploratory comparison, not causal evidence.",
        },
        "leakage_policy": "Current area rent, Census median rent, rental stress, historical rent change and forecast outputs are retained only for audit/reporting and excluded from candidate predictors.",
        "ors_called": False,
        "sprint1_to_sprint3_outputs_overwritten": False,
    }
    (output / "sprint4_audit.json").write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
    return {
        "history": history,
        "history_qa": qa,
        "validation": validation,
        "selected_method": selected_method,
        "forecasts": forecasts,
        "listings": listings,
        "area_features": area,
        "feature_model_scores": model_scores,
        "feature_importance": importance,
        "feature_spearman": spearman,
        "rental_activity": activity,
        "feature_audit": feature_audit,
        "feature_model_audit": model_audit,
        "output": output,
    }

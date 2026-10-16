"""Reusable Sprint 6 Q2 origin-centred Ridge research implementation.

This is a project-local adaptation of the frozen Sprint 6 research protocol.
It reads the saved Homes Victoria long-history panel and performs calculations
in memory only: no forecast package, ledger, notebook, or data file is written.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .sprint5_direct_growth_model import FEATURE_COLUMNS, build_origin_feature_rows, five_year_growth_target
from .sprint5_forecasting import forecast_values
from .sprint5_model_extension import fit_theta


LONG_HISTORY_RELATIVE_PATH = Path(
    "data/processed/sprint5/long_history_validation/"
    "homes_victoria_long_history_all_properties.csv"
)
FORECAST_ORIGIN = pd.Timestamp("2025-09-30")
LABEL_EMBARGO_QUARTERS = 1  # Conservative analyst rule; not a verified release lag.
BEHAVIOUR_FEATURES = (
    "std_quarterly_changes_last_8q",
    "current_rent_vs_recent_3y_mean_pct",
    "distance_from_recent_3y_peak_pct",
)
VOLUME_FEATURES = (
    "log1p_lease_count",
    "relative_log_lease_count",
    "count_log_change_4q",
    "count_log_change_8q",
    "count_log_change_std_8q",
)
FEATURES_21 = tuple([*FEATURE_COLUMNS, *VOLUME_FEATURES])
FEATURES_18 = tuple(feature for feature in FEATURES_21 if feature not in BEHAVIOUR_FEATURES)
MODEL_SPECS = {
    "ridge21": {
        "horizon_quarters": 12,
        "candidate_id": "origin_centred_ridge21_12q",
        "feature_set": "ridge21",
        "features": FEATURES_21,
        "alpha": 100.0,
        "anchor": "project_theta_approximation",
    },
    "ridge18": {
        "horizon_quarters": 20,
        "candidate_id": "origin_centred_ridge18_20q",
        "feature_set": "ridge18",
        "features": FEATURES_18,
        "alpha": 100.0,
        "anchor": "drift",
    },
}

if len(FEATURES_21) != 21 or len(FEATURES_18) != 18:
    raise RuntimeError("Frozen Sprint 6 Q2 feature definitions have changed unexpectedly.")


def load_homes_victoria_history(project_root: str | Path) -> pd.DataFrame:
    """Load the saved aggregate rental-history input; never fetch or write data."""

    path = Path(project_root).resolve() / LONG_HISTORY_RELATIVE_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Required authorised processed rental-history input is missing: {path}")
    frame = pd.read_csv(path, low_memory=False)
    required = {
        "source_rental_market",
        "period_end",
        "moving_annual_median_rent_aud",
        "rental_count",
        "is_summary_row",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Homes Victoria history is missing required columns: {missing}")
    frame["period_end"] = pd.to_datetime(frame["period_end"], errors="coerce")
    for column in ("moving_annual_median_rent_aud", "rental_count"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def build_market_panels(
    long_history: pd.DataFrame,
    *,
    complete_only: bool = True,
    market_order: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create native-market rent/count panels without aggregating duplicate keys."""

    required = {
        "source_rental_market",
        "period_end",
        "moving_annual_median_rent_aud",
        "rental_count",
        "is_summary_row",
    }
    missing = sorted(required.difference(long_history.columns))
    if missing:
        raise ValueError(f"Long-history input is missing required columns: {missing}")
    raw = long_history.copy()
    summary = raw["is_summary_row"].astype(str).str.strip().str.lower().isin({"true", "1"})
    raw = raw.loc[~summary].copy()
    raw["period_end"] = pd.to_datetime(raw["period_end"], errors="coerce")
    raw["moving_annual_median_rent_aud"] = pd.to_numeric(
        raw["moving_annual_median_rent_aud"], errors="coerce"
    )
    raw["rental_count"] = pd.to_numeric(raw["rental_count"], errors="coerce")
    raw = raw.dropna(subset=["source_rental_market", "period_end"])
    if raw.duplicated(["source_rental_market", "period_end"]).any():
        raise ValueError("Duplicate native-market/quarter keys; refusing silent aggregation.")

    rents = raw.pivot(
        index="source_rental_market",
        columns="period_end",
        values="moving_annual_median_rent_aud",
    ).sort_index(axis=1)
    counts = raw.pivot(
        index="source_rental_market",
        columns="period_end",
        values="rental_count",
    ).reindex(index=rents.index, columns=rents.columns)
    if complete_only:
        complete = rents.notna().all(axis=1) & rents.gt(0).all(axis=1)
        rents, counts = rents.loc[complete], counts.loc[complete]
    if market_order is not None:
        missing_markets = set(market_order).difference(rents.index)
        if missing_markets:
            raise ValueError(f"Requested native markets are missing from rental history: {sorted(missing_markets)[:5]}")
        rents, counts = rents.reindex(market_order), counts.reindex(market_order)
    return rents, counts, raw


def _annual_q3_origins(periods: Sequence[pd.Timestamp]) -> list[pd.Timestamp]:
    available = {pd.Timestamp(value) for value in periods}
    return [pd.Timestamp(f"{year}-09-30") for year in range(2003, 2026) if pd.Timestamp(f"{year}-09-30") in available]


def build_growth_design(
    rents: pd.DataFrame,
    counts: pd.DataFrame,
    *,
    horizons: Iterable[int] = (12, 20),
    origins: Iterable[pd.Timestamp] | None = None,
) -> pd.DataFrame:
    """Build causal quarterly features and distinct 12Q/20Q cumulative targets."""

    if not rents.index.equals(counts.index) or not rents.columns.equals(counts.columns):
        raise ValueError("Rent and lease-count panels must have identical market and period axes.")
    periods = [pd.Timestamp(value) for value in rents.columns]
    origin_values = list(origins) if origins is not None else _annual_q3_origins(periods)
    origin_values = [pd.Timestamp(value) for value in origin_values if pd.Timestamp(value) in set(periods)]
    legacy = build_origin_feature_rows(rents, periods, origin_values, role="current_prediction")
    base_columns = [
        "source_rental_market",
        "forecast_origin_date",
        "origin_index",
        *FEATURE_COLUMNS,
    ]
    common = legacy[base_columns].rename(
        columns={"source_rental_market": "market", "forecast_origin_date": "origin"}
    ).copy()
    common["origin"] = pd.to_datetime(common["origin"])
    common["lease_count"] = np.nan
    for feature in VOLUME_FEATURES:
        common[feature] = np.nan

    for idx, row in common.iterrows():
        at = int(row["origin_index"])
        history_counts = counts.loc[row["market"]].iloc[: at + 1].to_numpy(dtype=float)
        current_count = history_counts[-1]
        cross_market = counts.iloc[:, at].to_numpy(dtype=float)
        cross_median = float(np.nanmedian(cross_market))
        common.at[idx, "lease_count"] = current_count
        last_nine = history_counts[-9:]
        if (
            len(last_nine) == 9
            and np.isfinite(last_nine).all()
            and (last_nine >= 0).all()
            and np.isfinite(cross_median)
            and cross_median >= 0
        ):
            log_counts = np.log1p(last_nine)
            common.loc[idx, list(VOLUME_FEATURES)] = [
                np.log1p(current_count),
                np.log1p(current_count) - np.log1p(cross_median),
                log_counts[-1] - log_counts[-5],
                log_counts[-1] - log_counts[-9],
                np.std(np.diff(log_counts)),
            ]

    outputs: list[pd.DataFrame] = []
    for horizon in horizons:
        if int(horizon) not in (12, 20):
            raise ValueError("The frozen Q2 growth targets are defined for 12 or 20 quarters only.")
        frame = common.copy()
        frame["horizon"] = int(horizon)
        frame["target_end"] = [
            periods[int(index) + int(horizon)]
            if int(index) + int(horizon) < len(periods)
            else pd.NaT
            for index in frame["origin_index"]
        ]
        frame["actual_growth"] = [
            five_year_growth_target(
                rents.loc[market].to_numpy(dtype=float),
                int(index),
                horizon=int(horizon),
            )
            for market, index in zip(frame["market"], frame["origin_index"])
        ]
        frame["complete_common"] = np.isfinite(
            frame[list(FEATURES_21)].to_numpy(dtype=float)
        ).all(axis=1)
        outputs.append(frame)
    return pd.concat(outputs, ignore_index=True)


def chronological_training_rows(
    design: pd.DataFrame,
    forecast_origin: pd.Timestamp,
    horizon_quarters: int,
    *,
    label_embargo_quarters: int = LABEL_EMBARGO_QUARTERS,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Select only prior-origin, complete, fully matured training labels."""

    origin = pd.Timestamp(forecast_origin)
    if label_embargo_quarters < 0:
        raise ValueError("Label embargo cannot be negative.")
    cutoff = (
        origin - pd.offsets.QuarterEnd(label_embargo_quarters)
        if label_embargo_quarters
        else origin
    )
    target_end = pd.to_datetime(design["target_end"], errors="coerce")
    row_origin = pd.to_datetime(design["origin"], errors="coerce")
    mask = (
        design["horizon"].eq(int(horizon_quarters))
        & design["complete_common"].astype(bool)
        & row_origin.lt(origin)
        & target_end.le(cutoff)
        & pd.to_numeric(design["actual_growth"], errors="coerce").notna()
    )
    train = design.loc[mask].copy()
    if not train.empty and (
        not pd.to_datetime(train["origin"]).lt(origin).all()
        or not pd.to_datetime(train["target_end"]).le(cutoff).all()
    ):
        raise AssertionError("Chronological training or label-maturity safeguard failed.")
    return train, cutoff


def _macro_growth_anchor(
    rents: pd.DataFrame,
    market: str,
    origin: pd.Timestamp,
    horizon_quarters: int,
    anchor: str,
) -> float:
    history = rents.loc[market, :pd.Timestamp(origin)].to_numpy(dtype=float)
    if not len(history) or not np.isfinite(history).all() or history[-1] <= 0:
        raise ValueError(f"Invalid rent history for macro anchor: {market}")
    if anchor == "project_theta_approximation":
        forecast, _ = fit_theta(history, horizon_quarters)
    elif anchor == "drift":
        forecast = forecast_values(history, "drift", horizon_quarters)
    else:
        raise ValueError(f"Unknown frozen macro anchor: {anchor}")
    return float((float(forecast[-1]) / history[-1] - 1.0) * 100.0)


def fit_origin_centred_ridge(
    design: pd.DataFrame,
    rents: pd.DataFrame,
    forecast_origin: pd.Timestamp,
    candidate: str,
    *,
    label_embargo_quarters: int = LABEL_EMBARGO_QUARTERS,
    _limit_threads: bool = True,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Fit one frozen candidate; held-out actual growth is never used for fitting."""

    if candidate not in MODEL_SPECS:
        raise ValueError(f"Unknown frozen Q2 candidate: {candidate}")
    spec = MODEL_SPECS[candidate]
    horizon = int(spec["horizon_quarters"])
    origin = pd.Timestamp(forecast_origin)
    train, cutoff = chronological_training_rows(
        design,
        origin,
        horizon,
        label_embargo_quarters=label_embargo_quarters,
    )
    valid = design.loc[
        pd.to_datetime(design["origin"]).eq(origin)
        & design["horizon"].eq(horizon)
        & design["complete_common"].astype(bool)
    ].copy()
    if train.empty or valid.empty:
        raise ValueError(f"Insufficient complete training or validation rows for {candidate} at {origin.date()}.")
    features = list(spec["features"])
    if not np.isfinite(train[features].to_numpy(dtype=float)).all() or not np.isfinite(
        valid[features].to_numpy(dtype=float)
    ).all():
        raise ValueError("Non-finite Q2 feature values; the frozen protocol does not impute.")

    # Cohort-centred targets and train-only scaling are both fit strictly within
    # the chronological training rows selected above.
    target = train["actual_growth"] - train.groupby("origin")["actual_growth"].transform("mean")
    estimator = make_pipeline(StandardScaler(), Ridge(alpha=float(spec["alpha"])))
    if _limit_threads:
        with threadpool_limits(limits=1):
            estimator.fit(train[features].to_numpy(dtype=float), target.to_numpy(dtype=float))
            raw_residual = estimator.predict(valid[features].to_numpy(dtype=float))
    else:
        estimator.fit(train[features].to_numpy(dtype=float), target.to_numpy(dtype=float))
        raw_residual = estimator.predict(valid[features].to_numpy(dtype=float))
    if not np.isfinite(raw_residual).all():
        raise ValueError("Ridge produced non-finite residual predictions.")

    market_anchor = [
        _macro_growth_anchor(
            rents,
            str(market),
            origin,
            horizon,
            str(spec["anchor"]),
        )
        for market in valid["market"]
    ]
    macro = float(np.mean(market_anchor))
    centred_residual = raw_residual - float(np.mean(raw_residual))
    prediction = macro + centred_residual
    output = valid[
        ["market", "origin", "target_end", "horizon", "current_rent", "lease_count", "actual_growth"]
    ].copy()
    output["candidate"] = candidate
    output["family"] = "ridge"
    output["feature_set"] = str(spec["feature_set"])
    output["pred_growth"] = prediction
    output["rank_score"] = prediction
    output["kind"] = "numerical"
    output["example_window"] = "expanding"
    output["label_lag_quarters"] = int(label_embargo_quarters)
    output["macro_growth_forecast"] = macro
    output["raw_relative_growth_prediction"] = raw_residual
    audit = {
        "candidate": candidate,
        "horizon_quarters": horizon,
        "training_examples": int(len(train)),
        "training_cohorts": int(train["origin"].nunique()),
        "training_markets_per_cohort": sorted(train.groupby("origin").size().unique().astype(int).tolist()),
        "latest_training_origin": pd.Timestamp(train["origin"].max()),
        "latest_training_target": pd.Timestamp(train["target_end"].max()),
        "information_cutoff": cutoff,
        "feature_count": len(features),
        "features": features,
        "alpha": float(spec["alpha"]),
        "macro_anchor_growth_pct": macro,
        "centred_residual_mean": float(np.mean(centred_residual)),
        "future_training_targets": int(pd.to_datetime(train["target_end"]).gt(cutoff).sum()),
        "actual_validation_mean_used": False,
        "training_rows_used_for_scaling": int(len(train)),
        "scaler_mean": estimator.named_steps["standardscaler"].mean_.astype(float).tolist(),
    }
    return output, audit


def tie_aware_top_k_membership(scores: dict[str, float], k: int = 10) -> dict[str, float]:
    """Fractional membership at a rounded-score Top-K boundary, as preregistered."""

    if not scores or k < 1 or len(scores) != len(set(scores)):
        raise ValueError("Top-K requires a nonempty unique population and positive K.")
    rounded = {key: float(np.round(float(value), 10)) for key, value in scores.items()}
    if not np.isfinite(list(rounded.values())).all():
        raise ValueError("Top-K scores must be finite.")
    slots = min(k, len(rounded))
    boundary = sorted(rounded.values(), reverse=True)[slots - 1]
    above = [value for value in rounded.values() if value > boundary]
    tied = [value for value in rounded.values() if value == boundary]
    share = (slots - len(above)) / len(tied)
    return {
        market: 1.0 if value > boundary else share if value == boundary else 0.0
        for market, value in rounded.items()
    }


def _rank_and_top10(forecast: pd.DataFrame) -> pd.DataFrame:
    result = forecast.copy()
    scores = result["rank_score"].to_numpy(dtype=float)
    rounded = np.round(scores, 10)
    result["rank_score_rounded_10dp"] = rounded
    result["growth_rank"] = pd.Series(rounded).rank(method="average", ascending=False).to_numpy(dtype=float)
    membership = tie_aware_top_k_membership(dict(zip(result["market_id"], scores)), 10)
    result["top10_membership"] = result["market_id"].map(membership).astype(float)
    result["top10_candidate"] = result["top10_membership"].gt(0.0)
    if not np.isclose(result["top10_membership"].sum(), min(10, len(result)), atol=1e-10):
        raise AssertionError("Fractional Top-10 memberships do not sum to ten slots.")
    return result


def _feature_domain_warnings(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    features: Sequence[str],
) -> dict[str, list[str]]:
    warnings_by_market = {str(market): [] for market in valid["market"]}
    indexed = valid.set_index("market", drop=False)
    for feature in features:
        values = train[feature].to_numpy(dtype=float)
        if not len(values) or not np.isfinite(values).all():
            raise ValueError(f"Invalid training range for feature {feature}.")
        low, high = float(np.min(values)), float(np.max(values))
        for market, row in indexed.iterrows():
            value = float(row[feature])
            if value < low or value > high:
                warnings_by_market[str(market)].append(feature)
    return warnings_by_market


def reproduce_registered_predictions_in_memory(
    project_root: str | Path,
    *,
    expected_markets: int = 145,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, object]]]:
    """Re-fit the frozen current candidates in memory for audit; never register/write.

    This routine is for checking consistency against the saved Stage 7 forecasts.
    It does not create a forecast package, mutate the ledger, or write project files.
    """

    history = load_homes_victoria_history(project_root)
    rents, counts, _ = build_market_panels(history, complete_only=True)
    if len(rents) != expected_markets:
        raise ValueError(f"Expected {expected_markets} complete native markets, found {len(rents)}.")
    if FORECAST_ORIGIN not in rents.columns:
        raise ValueError(f"Forecast origin {FORECAST_ORIGIN.date()} is absent from the rental panel.")
    design = build_growth_design(rents, counts)
    forecasts: dict[str, pd.DataFrame] = {}
    audits: dict[str, dict[str, object]] = {}
    for candidate in ("ridge21", "ridge18"):
        fit, audit = fit_origin_centred_ridge(design, rents, FORECAST_ORIGIN, candidate)
        horizon = int(MODEL_SPECS[candidate]["horizon_quarters"])
        if len(fit) != expected_markets or fit["market"].duplicated().any():
            raise ValueError(f"Unexpected current market population for {candidate}.")
        if fit["actual_growth"].notna().any() or audit["actual_validation_mean_used"] is not False:
            raise AssertionError("Current prediction path accessed an unobserved target.")
        train, cutoff = chronological_training_rows(design, FORECAST_ORIGIN, horizon)
        valid = design.loc[
            pd.to_datetime(design["origin"]).eq(FORECAST_ORIGIN)
            & design["horizon"].eq(horizon)
            & design["complete_common"].astype(bool)
        ].copy()
        spec = MODEL_SPECS[candidate]
        warning_map = _feature_domain_warnings(train, valid, spec["features"])
        macro = float(fit["macro_growth_forecast"].iloc[0])
        raw_residual = fit["raw_relative_growth_prediction"].to_numpy(dtype=float)
        centred_residual = raw_residual - float(np.mean(raw_residual))
        predicted = fit["pred_growth"].to_numpy(dtype=float)
        target_end = FORECAST_ORIGIN + pd.offsets.QuarterEnd(horizon)
        start_rent = fit["current_rent"].to_numpy(dtype=float)
        output = pd.DataFrame(
            {
                "candidate": candidate,
                "market_id": fit["market"].astype(str).to_numpy(),
                "origin_period_end": FORECAST_ORIGIN.date().isoformat(),
                "target_end_date": target_end.date().isoformat(),
                "horizon_quarters": horizon,
                "origin_rent_aud": start_rent,
                "predicted_growth_pct": predicted,
                "predicted_endpoint_rent_aud": start_rent * (1.0 + predicted / 100.0),
                "rank_score": predicted,
                "macro_anchor_growth_pct": macro,
                "raw_model_residual_pct": raw_residual,
                "centred_model_residual_pct": centred_residual,
                "feature_domain_warning_count": [len(warning_map[str(market)]) for market in fit["market"]],
                "feature_domain_warning_features": [
                    json.dumps(warning_map[str(market)], ensure_ascii=False)
                    for market in fit["market"]
                ],
            }
        )
        output = _rank_and_top10(output)
        forecasts[candidate] = output
        audit.update(
            {
                "information_cutoff": cutoff,
                "target_end_date": target_end,
                "future_realised_targets_used": False,
            }
        )
        audits[candidate] = audit
    return forecasts, audits


def _top_k_recovery(actual: dict[str, float], predicted: dict[str, float], k: int = 10) -> float:
    """Return expected recovered market count under fractional boundary ties."""

    if set(actual) != set(predicted):
        raise ValueError("Actual and predicted Top-K populations differ.")
    actual_membership = tie_aware_top_k_membership(actual, k)
    predicted_membership = tie_aware_top_k_membership(predicted, k)
    return float(sum(actual_membership[key] * predicted_membership[key] for key in actual))


def top_k_recovery_metrics(
    actual: dict[str, float], predicted: dict[str, float], k: int = 10
) -> dict[str, float]:
    """Report recovered market count and its fraction of min(K, population)."""

    recovery_count = _top_k_recovery(actual, predicted, k)
    return {
        "recovery_count": recovery_count,
        "recovery_fraction": recovery_count / min(k, len(actual)),
    }


def _validate_design_over_origins(
    design: pd.DataFrame,
    rents: pd.DataFrame,
    selected_origins: Sequence[pd.Timestamp],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions: list[pd.DataFrame] = []
    metrics: list[dict[str, object]] = []
    for candidate in ("ridge21", "ridge18"):
        horizon = int(MODEL_SPECS[candidate]["horizon_quarters"])
        for origin in selected_origins:
            origin = pd.Timestamp(origin)
            if origin not in rents.columns:
                raise ValueError(f"Historical origin is absent from source data: {origin.date()}.")
            fit, _ = fit_origin_centred_ridge(
                design,
                rents,
                origin,
                candidate,
                _limit_threads=False,
            )
            if len(fit) != len(rents) or fit["actual_growth"].isna().any():
                raise ValueError(f"Historical validation target is incomplete for {candidate}/{origin.date()}.")
            error = fit["pred_growth"].to_numpy(dtype=float) - fit["actual_growth"].to_numpy(dtype=float)
            actual = dict(zip(fit["market"].astype(str), fit["actual_growth"].astype(float)))
            predicted = dict(zip(fit["market"].astype(str), fit["pred_growth"].astype(float)))
            recovery = top_k_recovery_metrics(actual, predicted, 10)
            spearman = float(fit["pred_growth"].corr(fit["actual_growth"], method="spearman"))
            metrics.append(
                {
                    "candidate": candidate,
                    "horizon_quarters": horizon,
                    "origin": origin,
                    "target_end": origin + pd.offsets.QuarterEnd(horizon),
                    "markets": int(len(fit)),
                    "mae_pp": float(np.mean(np.abs(error))),
                    "rmse_pp": float(np.sqrt(np.mean(np.square(error)))),
                    "bias_pp": float(np.mean(error)),
                    "spearman": spearman,
                    "tie_aware_top10_recovery_count": recovery["recovery_count"],
                    "tie_aware_top10_recovery_fraction": recovery["recovery_fraction"],
                    "validation_role": "retrospective_descriptive_not_independent_prospective",
                }
            )
            predictions.append(fit)
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(metrics)


def run_matched_historical_validation(
    project_root: str | Path,
    *,
    origins: Iterable[pd.Timestamp] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the frozen 3Y Ridge21 and 5Y Ridge18 over matched historical Q3 origins.

    Returns predictions and descriptive per-origin metrics in memory. Top-10
    recovery has separate market-count and fraction columns. This is a
    retrospective exercise on revised history, not an independent prospective test.
    """

    history = load_homes_victoria_history(project_root)
    rents, counts, _ = build_market_panels(history, complete_only=True)
    design = build_growth_design(rents, counts)
    selected_origins = list(origins) if origins is not None else [
        pd.Timestamp(f"{year}-09-30") for year in range(2010, 2021)
    ]
    # One deterministic CPU context for the whole backtest avoids repeatedly
    # toggling native thread pools across the origin loop.
    with threadpool_limits(limits=1):
        return _validate_design_over_origins(design, rents, selected_origins)

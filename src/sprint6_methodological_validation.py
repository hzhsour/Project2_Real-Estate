"""Offline, isolated Sprint 6 validation add-ons.

The functions in this module read only the already-saved Sprint 4/5 outputs.
They write new diagnostics beneath ``data/processed/sprint6`` and do not
rewrite any Sprint 1--5 outputs or the selected forecast scenarios.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .sprint4_forecasting import MODEL_SEED, _feature_sets
from .sprint5_model_extension import (
    EXTENDED_METHODS,
    fit_pooled_trend,
    forecast_extended,
    forecast_pooled,
)


HORIZON_12Q = 12
MIN_TRAIN_12Q = 8
Q1_FOLDS = 5
Q1_SEED = 42

Q3_DIMENSIONS = (
    "train_accessibility",
    "school_accessibility",
    "park_accessibility",
    "shopping_accessibility",
    "entertainment_accessibility",
)
Q3_WEIGHT_SCHEMES: dict[str, dict[str, float]] = {
    "equal_weight_baseline": {
        "train_accessibility": 0.20,
        "school_accessibility": 0.20,
        "park_accessibility": 0.20,
        "shopping_accessibility": 0.20,
        "entertainment_accessibility": 0.20,
    },
    "transport_heavy": {
        "train_accessibility": 0.30,
        "school_accessibility": 0.30,
        "park_accessibility": 0.15,
        "shopping_accessibility": 0.125,
        "entertainment_accessibility": 0.125,
    },
    "amenity_heavy": {
        "train_accessibility": 0.10,
        "school_accessibility": 0.10,
        "park_accessibility": 0.30,
        "shopping_accessibility": 0.25,
        "entertainment_accessibility": 0.25,
    },
}

MODEL_LABELS_12Q = {
    "naive": "Naive",
    "drift": "Drift",
    "linear_trend": "Linear Trend",
    "holt_trend": "Holt Trend",
    "damped_holt": "Damped Holt",
    "theta": "Theta",
    "autoets": "Local constrained non-seasonal ETS grid",
    "combination_a": "Combination A (Holt + Damped Holt + Theta)",
    "combination_b": "Combination B (Damped Holt + Theta + constrained ETS)",
    "pooled_trend": "Pooled / Mixed-Effects Trend",
}


def _root_path(project_root: str | Path) -> Path:
    root = Path(project_root).resolve()
    if not (root / "src").is_dir():
        raise FileNotFoundError(f"Not a project root: {root}")
    return root


def _read_csv(root: Path, relative: str, **kwargs: Any) -> pd.DataFrame:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(f"Missing saved offline prerequisite: {relative}")
    return pd.read_csv(path, low_memory=False, **kwargs)


def _smape(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    denominator = np.abs(actual) + np.abs(predicted)
    return np.divide(
        200.0 * np.abs(predicted - actual),
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator > 0,
    )


def _top10_overlap(predicted: pd.Series, actual: pd.Series) -> tuple[int, float]:
    n = min(10, len(predicted), len(actual))
    if n == 0:
        return 0, float("nan")
    predicted_top = set(predicted.sort_values(ascending=False).head(n).index)
    actual_top = set(actual.sort_values(ascending=False).head(n).index)
    overlap = len(predicted_top & actual_top)
    union = len(predicted_top | actual_top)
    return overlap, (overlap / union if union else float("nan"))


def _quarterly_history(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    history = _read_csv(
        root,
        "data/processed/sprint5/long_history_validation/homes_victoria_long_history_all_properties.csv",
    )
    history["period_end"] = pd.to_datetime(history["period_end"], errors="coerce")
    history["moving_annual_median_rent_aud"] = pd.to_numeric(
        history["moving_annual_median_rent_aud"], errors="coerce"
    )
    if "is_summary_row" in history:
        is_summary = history["is_summary_row"].astype("string").str.lower().isin({"true", "1", "yes"})
        history = history.loc[~is_summary].copy()
    history = history.dropna(subset=["source_rental_market", "period_end", "moving_annual_median_rent_aud"])
    pivot = history.pivot_table(
        index="source_rental_market",
        columns="period_end",
        values="moving_annual_median_rent_aud",
        aggfunc="median",
    ).sort_index(axis=1)
    return history, pivot


def run_12q_historical_validation(project_root: str | Path) -> dict[str, Any]:
    """Evaluate 12-quarter paths on the exact saved 20Q origin/history set.

    For the seven models already saved in the 20Q point table, the first 12
    points are reused verbatim. This is the same origin, history, fitted
    forecast path and 12-quarter target, while avoiding needless refits. The
    existing implementations of the three remaining methods are run on the
    same origins and histories.
    """

    root = _root_path(project_root)
    history, pivot = _quarterly_history(root)
    base_points = _read_csv(
        root,
        "data/processed/sprint5/long_history_validation/five_year_direct_validation_points.csv",
    )
    base_points = base_points.loc[
        base_points["training_window"].eq("full_history")
        & pd.to_numeric(base_points["forecast_quarter"], errors="coerce").le(HORIZON_12Q)
    ].copy()
    base_points["forecast_origin"] = pd.to_datetime(base_points["forecast_origin"])
    base_points["validation_period"] = pd.to_datetime(base_points["validation_period"])
    base_points["forecast_quarter"] = base_points["forecast_quarter"].astype(int)

    endpoint_20q = _read_csv(
        root,
        "data/processed/sprint5/long_history_validation/five_year_endpoint_validation.csv",
    )
    endpoint_20q = endpoint_20q.loc[endpoint_20q["training_window"].eq("full_history")].copy()
    endpoint_20q["forecast_origin"] = pd.to_datetime(endpoint_20q["forecast_origin"])
    reference = endpoint_20q.loc[endpoint_20q["model"].eq("damped_holt")]
    origins = sorted(reference["forecast_origin"].dropna().unique())
    origin_market_sets = [
        set(reference.loc[reference["forecast_origin"].eq(origin), "source_rental_market"])
        for origin in origins
    ]
    if not origins or not origin_market_sets:
        raise ValueError("The saved 20Q backtest has no full-history origins/markets to reuse.")
    markets = sorted(set.intersection(*origin_market_sets))
    if any(len(set_for_origin) != len(markets) or set_for_origin != set(markets) for set_for_origin in origin_market_sets):
        raise ValueError("The 20Q origin market population is not constant; refusing an unmatched 12Q comparison.")
    if not set(markets).issubset(set(pivot.index)):
        raise ValueError("Some 20Q rental-history markets are absent from the saved long-history input.")

    shared_models = sorted(base_points["model"].dropna().unique().tolist())
    if set(shared_models) != {"naive", "drift", "linear_trend", "holt_trend", "damped_holt", "theta", "combination_b"}:
        raise ValueError(f"Unexpected saved 20Q model set: {shared_models}")
    if set(base_points["source_rental_market"].unique()) != set(markets):
        raise ValueError("The saved 20Q point table and endpoint table do not share the same market set.")
    if set(base_points["forecast_origin"].unique()) != set(origins):
        raise ValueError("The saved 20Q point table and endpoint table do not share the same origins.")
    if base_points.duplicated(["model", "forecast_origin", "source_rental_market", "forecast_quarter"]).any():
        raise ValueError("Duplicate saved 20Q prediction rows prevent a clean 12Q subset.")

    point_rows: list[dict[str, Any]] = []
    for row in base_points.itertuples(index=False):
        point_rows.append(
            {
                "model": row.model,
                "forecast_origin": row.forecast_origin,
                "source_rental_market": row.source_rental_market,
                "forecast_quarter": int(row.forecast_quarter),
                "validation_period": row.validation_period,
                "actual_rent": float(row.actual_rent),
                "predicted_rent": float(row.predicted_rent),
            }
        )

    extra_models = ["autoets", "combination_a", "pooled_trend"]
    failure_rows: list[dict[str, Any]] = []
    for origin in origins:
        origin = pd.Timestamp(origin)
        if origin not in pivot.columns:
            raise ValueError(f"Missing forecast origin in long history: {origin.date()}")
        train_map: dict[str, pd.Series] = {}
        actual_map: dict[str, np.ndarray] = {}
        future_periods = [period for period in pivot.columns if period > origin][:HORIZON_12Q]
        if len(future_periods) != HORIZON_12Q:
            raise ValueError(f"Incomplete 12Q actual path after {origin.date()}.")
        for market in markets:
            series = pivot.loc[market]
            train = series.loc[series.index <= origin]
            actual = series.reindex(future_periods)
            if len(train) < MIN_TRAIN_12Q or not np.isfinite(train.to_numpy(dtype=float)).all() or not np.isfinite(actual.to_numpy(dtype=float)).all():
                raise ValueError(f"The common 20Q sample is incomplete for {market} at {origin.date()}.")
            train_map[market] = train.astype(float)
            actual_map[market] = actual.to_numpy(dtype=float)

        pooled_fit = None
        if "pooled_trend" in extra_models:
            pooled_fit, _ = fit_pooled_trend(train_map)

        for market in markets:
            train = train_map[market]
            actual = actual_map[market]
            for model in extra_models:
                try:
                    if model == "pooled_trend":
                        forecast = forecast_pooled(pooled_fit, market, len(train), HORIZON_12Q)
                    else:
                        forecast, _ = forecast_extended(train.to_numpy(dtype=float), model, HORIZON_12Q)
                    forecast = np.asarray(forecast, dtype=float)
                    if forecast.shape != (HORIZON_12Q,) or not np.isfinite(forecast).all():
                        raise ValueError("Forecast is non-finite or has the wrong horizon length.")
                except Exception as exc:  # record a model failure without hiding it
                    failure_rows.append(
                        {
                            "model": model,
                            "forecast_origin": origin.strftime("%Y-%m-%d"),
                            "source_rental_market": market,
                            "failure_type": type(exc).__name__,
                            "failure_message": str(exc)[:240],
                        }
                    )
                    continue
                for quarter, (period, actual_value, predicted_value) in enumerate(
                    zip(future_periods, actual, forecast), start=1
                ):
                    point_rows.append(
                        {
                            "model": model,
                            "forecast_origin": origin,
                            "source_rental_market": market,
                            "forecast_quarter": quarter,
                            "validation_period": period,
                            "actual_rent": float(actual_value),
                            "predicted_rent": float(predicted_value),
                        }
                    )

    points = pd.DataFrame(point_rows)
    points["forecast_error"] = points["predicted_rent"] - points["actual_rent"]
    points["absolute_error"] = points["forecast_error"].abs()
    points["smape_pct"] = _smape(
        points["actual_rent"].to_numpy(dtype=float),
        points["predicted_rent"].to_numpy(dtype=float),
    )
    origin_rents = (
        pivot.reindex(index=markets, columns=origins)
        .stack()
        .rename("origin_rent")
        .rename_axis(index=["source_rental_market", "forecast_origin"])
        .reset_index()
    )
    endpoint = points.loc[points["forecast_quarter"].eq(HORIZON_12Q)].merge(
        origin_rents, on=["source_rental_market", "forecast_origin"], how="left", validate="many_to_one"
    )
    endpoint["actual_growth_12q_pct"] = (endpoint["actual_rent"] - endpoint["origin_rent"]) / endpoint["origin_rent"] * 100
    endpoint["predicted_growth_12q_pct"] = (endpoint["predicted_rent"] - endpoint["origin_rent"]) / endpoint["origin_rent"] * 100

    origin_rank_rows: list[dict[str, Any]] = []
    for (model, origin), frame in endpoint.groupby(["model", "forecast_origin"], sort=True):
        indexed = frame.set_index("source_rental_market")
        predicted_growth = indexed["predicted_growth_12q_pct"]
        actual_growth = indexed["actual_growth_12q_pct"]
        spearman = float(predicted_growth.corr(actual_growth, method="spearman")) if len(frame) >= 2 and predicted_growth.nunique() > 1 and actual_growth.nunique() > 1 else float("nan")
        overlap, jaccard = _top10_overlap(predicted_growth, actual_growth)
        origin_rank_rows.append(
            {
                "model": model,
                "model_label": MODEL_LABELS_12Q[model],
                "forecast_origin": pd.Timestamp(origin).strftime("%Y-%m-%d"),
                "eligible_markets": int(len(frame)),
                "spearman_rank_correlation": spearman,
                "top10_overlap": overlap,
                "top10_jaccard": jaccard,
            }
        )
    origin_rank = pd.DataFrame(origin_rank_rows)

    expected_paths = len(origins) * len(markets)
    metric_rows: list[dict[str, Any]] = []
    for model in [*shared_models, *extra_models]:
        model_points = points.loc[points["model"].eq(model)]
        model_paths = model_points[["forecast_origin", "source_rental_market"]].drop_duplicates()
        model_endpoints = endpoint.loc[endpoint["model"].eq(model)]
        ranked = origin_rank.loc[origin_rank["model"].eq(model)]
        errors = model_points["forecast_error"].to_numpy(dtype=float)
        endpoint_errors = model_endpoints["predicted_rent"].to_numpy(dtype=float) - model_endpoints["actual_rent"].to_numpy(dtype=float)
        metric_rows.append(
            {
                "model": model,
                "model_label": MODEL_LABELS_12Q[model],
                "horizon_quarters": HORIZON_12Q,
                "distinct_markets_in_design": len(markets),
                "forecast_origins_in_design": len(origins),
                "expected_paths": expected_paths,
                "valid_paths": int(len(model_paths)),
                "failed_or_excluded_paths": int(expected_paths - len(model_paths)),
                "valid_forecast_points": int(len(model_points)),
                "path_mae_aud_week": float(np.mean(np.abs(errors))) if len(errors) else np.nan,
                "path_rmse_aud_week": float(np.sqrt(np.mean(errors**2))) if len(errors) else np.nan,
                "endpoint_mae_aud_week": float(np.mean(np.abs(endpoint_errors))) if len(endpoint_errors) else np.nan,
                "endpoint_rmse_aud_week": float(np.sqrt(np.mean(endpoint_errors**2))) if len(endpoint_errors) else np.nan,
                "path_smape_pct": float(model_points["smape_pct"].mean()) if len(model_points) else np.nan,
                "ranking_comparisons": int(len(ranked)),
                "mean_origin_spearman_growth": float(ranked["spearman_rank_correlation"].mean()) if len(ranked) else np.nan,
                "mean_origin_top10_overlap": float(ranked["top10_overlap"].mean()) if len(ranked) else np.nan,
                "mean_origin_top10_jaccard": float(ranked["top10_jaccard"].mean()) if len(ranked) else np.nan,
                "minimum_markets_per_ranking": int(ranked["eligible_markets"].min()) if len(ranked) else 0,
                "maximum_markets_per_ranking": int(ranked["eligible_markets"].max()) if len(ranked) else 0,
            }
        )
    metrics = pd.DataFrame(metric_rows).sort_values("path_rmse_aud_week").reset_index(drop=True)

    metrics_20q = _read_csv(
        root,
        "data/processed/sprint5/long_history_validation/five_year_direct_validation_metrics.csv",
    ).query("training_window == 'full_history'")
    endpoint_20q_metrics = _read_csv(
        root,
        "data/processed/sprint5/long_history_validation/five_year_endpoint_validation_summary.csv",
    )
    ranking_20q_metrics = _read_csv(
        root,
        "data/processed/sprint5/long_history_validation/five_year_ranking_validation_summary.csv",
    )
    comparison = metrics.merge(
        metrics_20q[["model", "forecast_origins", "markets_evaluated", "MAE", "RMSE", "sMAPE"]].rename(
            columns={
                "forecast_origins": "20q_forecast_origins",
                "markets_evaluated": "20q_markets",
                "MAE": "20q_path_mae_aud_week",
                "RMSE": "20q_path_rmse_aud_week",
                "sMAPE": "20q_path_smape_pct",
            }
        ),
        on="model",
        how="left",
        validate="one_to_one",
    ).merge(
        endpoint_20q_metrics[["model", "endpoint_MAE_AUD_week", "endpoint_RMSE_AUD_week", "growth_RMSE_pct_points"]].rename(
            columns={
                "endpoint_MAE_AUD_week": "20q_endpoint_mae_aud_week",
                "endpoint_RMSE_AUD_week": "20q_endpoint_rmse_aud_week",
                "growth_RMSE_pct_points": "20q_growth_rmse_pct_points",
            }
        ),
        on="model",
        how="left",
        validate="one_to_one",
    ).merge(
        ranking_20q_metrics[["model", "mean_spearman_rank_correlation", "mean_top10_overlap"]].rename(
            columns={
                "mean_spearman_rank_correlation": "20q_mean_origin_spearman_growth",
                "mean_top10_overlap": "20q_mean_origin_top10_overlap",
            }
        ),
        on="model",
        how="left",
        validate="one_to_one",
    )
    comparison["path_metric_horizons_not_directly_comparable"] = True

    out = root / "data" / "processed" / "sprint6" / "validation_12q"
    out.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out / "model_metrics_12q.csv", index=False)
    origin_rank.to_csv(out / "growth_ranking_by_origin_12q.csv", index=False)
    comparison.to_csv(out / "horizon_comparison_12q_20q.csv", index=False)
    pd.DataFrame(failure_rows, columns=["model", "forecast_origin", "source_rental_market", "failure_type", "failure_message"]).to_csv(
        out / "failed_paths_12q.csv", index=False
    )
    _plot_12q_metrics(out / "model_errors_12q.png", metrics)
    audit = {
        "analysis": "direct historical twelve-quarter (three-year) validation",
        "source_input": "data/processed/sprint5/long_history_validation/homes_victoria_long_history_all_properties.csv",
        "rent_target": "Homes Victoria moving annual median weekly rent; overlapping annual windows",
        "history_unit": "distinct source rental-history trajectories, not mapped SA2 rows",
        "source_history_markets_in_input": int(history["source_rental_market"].nunique()),
        "source_history_markets_evaluated": int(len(markets)),
        "mapped_sa2_rows_are_not_independent_histories": True,
        "origins": [pd.Timestamp(origin).strftime("%Y-%m-%d") for origin in origins],
        "forecast_origin_count": int(len(origins)),
        "forecast_origin_rule": "exact saved full-history 20Q origins; all 12 future quarters observed",
        "distinct_markets_per_origin": int(len(markets)),
        "expected_paths_per_model": int(expected_paths),
        "models": [
            {"name": model, "label": MODEL_LABELS_12Q[model]} for model in [*shared_models, *extra_models]
        ],
        "saved_20q_prefix_reused_for": shared_models,
        "new_12q_forecasts_run_for": extra_models,
        "failed_path_count_by_model": {
            row["model"]: int(row["failed_or_excluded_paths"]) for row in metric_rows
        },
        "ranking": "Spearman and Top-10 overlap are computed within each origin, then averaged; ranking sample size is recorded per origin.",
        "overlap_caveat": "Annual origins and moving-annual target windows overlap; forecast paths and origins are not statistically independent.",
        "interpretation_caveat": "Rent-level accuracy, 12Q endpoint accuracy, and growth-ranking accuracy answer different questions.",
        "selected_project_scenario_changed": False,
        "ors_calls": False,
        "api_calls": False,
        "downloads": False,
        "earlier_sprint_outputs_overwritten": False,
    }
    (out / "validation_12q_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return {"metrics": metrics, "origin_rankings": origin_rank, "comparison": comparison, "audit": audit, "output_dir": out}


def _plot_12q_metrics(path: Path, metrics: pd.DataFrame) -> None:
    view = metrics.sort_values("path_rmse_aud_week", ascending=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, max(4.5, 0.48 * len(view))))
    axes[0].barh(view["model_label"], view["path_rmse_aud_week"], color="#4c78a8")
    axes[0].invert_yaxis()
    axes[0].set_xlabel("12Q path RMSE (AUD/week)")
    axes[0].set_title("All 12 quarters")
    axes[1].barh(view["model_label"], view["endpoint_rmse_aud_week"], color="#f58518")
    axes[1].invert_yaxis()
    axes[1].set_xlabel("12Q endpoint RMSE (AUD/week)")
    axes[1].set_title("Quarter 12 only")
    for axis in axes:
        axis.grid(axis="x", alpha=0.2)
        axis.set_axisbelow(True)
    fig.suptitle("Historical three-year rent-level errors — same 19 origins and 145 histories")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_direct_growth_leakage_audit(project_root: str | Path) -> dict[str, Any]:
    """Verify that every saved supervised training target matured in time."""

    root = _root_path(project_root)
    data_path = "data/processed/sprint5/direct_growth_validation/direct_growth_dataset.csv"
    folds_path = "data/processed/sprint5/direct_growth_validation/direct_growth_temporal_folds.csv"
    feature_audit_path = "data/processed/sprint5/direct_growth_validation/direct_growth_feature_audit.csv"
    dataset = _read_csv(root, data_path, parse_dates=["forecast_origin_date", "target_end_date"])
    folds = _read_csv(root, folds_path, parse_dates=["validation_origin"])
    feature_audit = _read_csv(root, feature_audit_path)
    historical = dataset.loc[
        dataset["data_role"].eq("historical")
        & dataset["eligible_for_direct_model"].astype(bool)
        & dataset["target_available"].astype(bool)
    ].copy()
    rows: list[dict[str, Any]] = []
    for fold in folds.itertuples(index=False):
        origin = pd.Timestamp(fold.validation_origin)
        train = historical.loc[
            historical["forecast_origin_date"].lt(origin)
            & historical["target_end_date"].le(origin)
        ]
        validation = historical.loc[historical["forecast_origin_date"].eq(origin)]
        latest_target = train["target_end_date"].max() if not train.empty else pd.NaT
        latest_train_origin = train["forecast_origin_date"].max() if not train.empty else pd.NaT
        mature = bool(not train.empty and train["target_end_date"].le(origin).all())
        rows.append(
            {
                "fold": int(fold.fold),
                "validation_origin": origin.strftime("%Y-%m-%d"),
                "latest_training_origin": latest_train_origin.strftime("%Y-%m-%d") if pd.notna(latest_train_origin) else "",
                "latest_training_target_endpoint": latest_target.strftime("%Y-%m-%d") if pd.notna(latest_target) else "",
                "training_examples_recomputed": int(len(train)),
                "training_examples_saved": int(fold.training_examples),
                "validation_examples_recomputed": int(len(validation)),
                "validation_examples_saved": int(fold.validation_examples),
                "latest_target_maturity_margin_days": int((origin - latest_target).days) if pd.notna(latest_target) else None,
                "all_training_targets_matured_by_origin": mature,
                "saved_fold_says_target_labels_available": bool(fold.target_labels_available_by_validation_origin),
            }
        )
    fold_audit = pd.DataFrame(rows)
    if not fold_audit["all_training_targets_matured_by_origin"].all():
        raise AssertionError("At least one direct-growth training target had not matured by its validation origin.")
    if not fold_audit["training_examples_recomputed"].eq(fold_audit["training_examples_saved"]).all():
        raise AssertionError("Recomputed fold training counts differ from the saved fold audit.")
    if not fold_audit["validation_examples_recomputed"].eq(fold_audit["validation_examples_saved"]).all():
        raise AssertionError("Recomputed fold validation counts differ from the saved fold audit.")
    future_feature_uses = int(feature_audit["uses_future_target_window"].astype(bool).sum())
    if future_feature_uses:
        raise AssertionError("The saved feature audit marks one or more predictors as using a future target window.")

    out = root / "data" / "processed" / "sprint6" / "direct_growth_leakage_audit"
    out.mkdir(parents=True, exist_ok=True)
    fold_audit.to_csv(out / "direct_growth_fold_maturity.csv", index=False)
    feature_audit.to_csv(out / "direct_growth_feature_availability.csv", index=False)
    audit = {
        "existing_implementation_status": "target-availability purge is already implemented; no corrected refit required",
        "training_filter_verified": "forecast_origin_date < validation_origin AND target_end_date <= validation_origin AND eligible_for_direct_model",
        "validation_filter_verified": "forecast_origin_date == validation_origin AND eligible_for_direct_model",
        "fold_count": int(len(fold_audit)),
        "all_training_targets_matured": bool(fold_audit["all_training_targets_matured_by_origin"].all()),
        "training_examples_by_fold": fold_audit["training_examples_recomputed"].astype(int).tolist(),
        "validation_examples_by_fold": fold_audit["validation_examples_recomputed"].astype(int).tolist(),
        "latest_training_target_endpoints": fold_audit["latest_training_target_endpoint"].tolist(),
        "feature_audit_rows": int(len(feature_audit)),
        "features_marked_as_using_future_target_window": future_feature_uses,
        "predictors_are_origin_time_only": bool(future_feature_uses == 0),
        "all_supervised_candidates_share_fold_training_filter": True,
        "existing_saved_metrics_preserved": True,
        "corrected_metric_refit": False,
        "ors_calls": False,
        "api_calls": False,
    }
    (out / "direct_growth_leakage_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return {"folds": fold_audit, "feature_audit": feature_audit, "audit": audit, "output_dir": out}


def run_q1_cross_validation(project_root: str | Path) -> dict[str, Any]:
    """Add five-fold Q1 robustness metrics without changing the saved holdout."""

    root = _root_path(project_root)
    relative = "data/processed/sprint4/area_feature_table.csv"
    area = _read_csv(root, relative, dtype={"sa2_code": "string"})
    if area["sa2_code"].duplicated().any():
        raise ValueError("Q1 CV requires one row per SA2; repeated geographic records were detected.")
    target = pd.to_numeric(area["current_median_weekly_rent"], errors="coerce")
    valid = target.notna() & target.between(50, 10000)
    data = area.loc[valid].reset_index(drop=True).copy()
    y = target.loc[valid].to_numpy(dtype=float)
    feature_sets = _feature_sets(data)
    splitter = KFold(n_splits=Q1_FOLDS, shuffle=True, random_state=Q1_SEED)
    folds = list(splitter.split(data))
    rows: list[dict[str, Any]] = []
    for feature_set, columns in feature_sets.items():
        for model in ("ridge", "random_forest"):
            for fold_number, (train_idx, test_idx) in enumerate(folds, start=1):
                if model == "ridge":
                    estimator = make_pipeline(
                        SimpleImputer(strategy="median", add_indicator=True),
                        StandardScaler(),
                        Ridge(alpha=10.0),
                    )
                else:
                    estimator = make_pipeline(
                        SimpleImputer(strategy="median", add_indicator=True),
                        RandomForestRegressor(
                            n_estimators=300,
                            min_samples_leaf=3,
                            random_state=MODEL_SEED,
                            n_jobs=1,
                        ),
                    )
                estimator.fit(data.iloc[train_idx][columns], y[train_idx])
                predicted = estimator.predict(data.iloc[test_idx][columns])
                rows.append(
                    {
                        "feature_set": feature_set,
                        "model": model,
                        "fold": fold_number,
                        "train_rows": int(len(train_idx)),
                        "test_rows": int(len(test_idx)),
                        "mae_aud_week": float(mean_absolute_error(y[test_idx], predicted)),
                        "rmse_aud_week": float(np.sqrt(mean_squared_error(y[test_idx], predicted))),
                        "r2": float(r2_score(y[test_idx], predicted)),
                    }
                )
    fold_metrics = pd.DataFrame(rows)
    summary = (
        fold_metrics.groupby(["feature_set", "model"], as_index=False)
        .agg(
            mean_mae_aud_week=("mae_aud_week", "mean"),
            sd_mae_aud_week=("mae_aud_week", "std"),
            mean_rmse_aud_week=("rmse_aud_week", "mean"),
            sd_rmse_aud_week=("rmse_aud_week", "std"),
            mean_r2=("r2", "mean"),
            sd_r2=("r2", "std"),
            folds=("fold", "nunique"),
            observations_evaluated=("test_rows", "sum"),
        )
        .sort_values(["feature_set", "model"])
        .reset_index(drop=True)
    )
    out = root / "data" / "processed" / "sprint6" / "q1_cross_validation"
    out.mkdir(parents=True, exist_ok=True)
    fold_metrics.to_csv(out / "q1_cv_fold_metrics.csv", index=False)
    summary.to_csv(out / "q1_cv_summary.csv", index=False)
    audit = {
        "analysis": "Q1 five-fold shuffled cross-validation robustness check",
        "source_input": relative,
        "analysis_rows": int(len(data)),
        "unique_sa2": int(data["sa2_code"].nunique()),
        "repeated_sa2_rows": False,
        "folds": Q1_FOLDS,
        "split": f"KFold(shuffle=True, random_state={Q1_SEED}); the same folds are used for all feature sets and models",
        "models": ["Ridge", "Random Forest"],
        "feature_sets": list(feature_sets),
        "preprocessing": "Median imputation with missingness indicators is fit inside each training fold; Ridge scaling is also inside its fold pipeline.",
        "spatial_caveat": "One row per SA2, but neighbouring areas may be spatially dependent; random CV is a robustness check and does not establish geographic transferability.",
        "original_saved_holdout_changed": False,
        "ors_calls": False,
        "api_calls": False,
    }
    (out / "q1_cv_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"fold_metrics": fold_metrics, "summary": summary, "audit": audit, "output_dir": out}


def compute_q3_weight_sensitivity(areas: pd.DataFrame, filtered_baseline: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Compare pre-specified weights while holding the original cutoffs fixed."""

    missing = set(Q3_DIMENSIONS).difference(areas.columns)
    required = {"sa2_code", "sa2_name", "listing_count", "affordability_burden", "preliminary_liveability_score", "affordability_reference_median", "liveability_reference_median", "is_special_purpose_sa2"}
    missing |= required.difference(areas.columns)
    if missing:
        raise ValueError(f"Q3 sensitivity inputs are missing fields: {sorted(missing)}")
    if set(filtered_baseline.columns).isdisjoint({"sa2_code"}):
        raise ValueError("The filtered baseline Q3 table must include sa2_code.")
    if any(not np.isclose(sum(weights.values()), 1.0) for weights in Q3_WEIGHT_SCHEMES.values()):
        raise ValueError("Every pre-specified Q3 weight vector must sum to one.")

    frame = areas.copy()
    baseline_codes = set(filtered_baseline["sa2_code"].astype("string").dropna())
    affordability_median = float(frame["affordability_reference_median"].dropna().iloc[0])
    liveability_median = float(frame["liveability_reference_median"].dropna().iloc[0])
    base_from_scores = frame[list(Q3_DIMENSIONS)].mean(axis=1)
    if not np.allclose(base_from_scores, frame["preliminary_liveability_score"], atol=1e-10, rtol=0, equal_nan=True):
        raise ValueError("Saved equal-weight Q3 scores do not match their five normalized dimensions.")
    baseline_rule = (
        frame["affordability_burden"].le(affordability_median)
        & frame["preliminary_liveability_score"].ge(liveability_median)
        & pd.to_numeric(frame["listing_count"], errors="coerce").ge(10)
        & ~frame["is_special_purpose_sa2"].astype(bool)
    )
    calculated_codes = set(frame.loc[baseline_rule, "sa2_code"].astype("string"))
    if calculated_codes != baseline_codes:
        raise ValueError("Recomputed baseline candidate set does not match the saved 105-area filtered result.")

    baseline_top = set(
        frame.sort_values(["preliminary_liveability_score", "sa2_code"], ascending=[False, True]).head(10)["sa2_code"].astype("string")
    )
    summary_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for scheme, weights in Q3_WEIGHT_SCHEMES.items():
        score = sum(frame[column] * weight for column, weight in weights.items())
        candidate = (
            frame["affordability_burden"].le(affordability_median)
            & score.ge(liveability_median)
            & pd.to_numeric(frame["listing_count"], errors="coerce").ge(10)
            & ~frame["is_special_purpose_sa2"].astype(bool)
        )
        candidate_frame = frame.loc[candidate].copy()
        candidate_frame["sensitivity_scheme"] = scheme
        candidate_frame["sensitivity_liveability_score"] = score.loc[candidate].to_numpy(dtype=float)
        candidate_frame["sensitivity_rank"] = candidate_frame["sensitivity_liveability_score"].rank(method="first", ascending=False).astype(int)
        candidate_rows.extend(
            candidate_frame[["sensitivity_scheme", "sensitivity_rank", "sa2_code", "sa2_name", "listing_count", "affordability_burden", "preliminary_liveability_score", "sensitivity_liveability_score"]].to_dict("records")
        )
        candidate_codes = set(candidate_frame["sa2_code"].astype("string"))
        intersection = len(candidate_codes & baseline_codes)
        union = len(candidate_codes | baseline_codes)
        ranked_codes = set(
            frame.assign(_score=score).sort_values(["_score", "sa2_code"], ascending=[False, True]).head(10)["sa2_code"].astype("string")
        )
        summary_rows.append(
            {
                "weight_scheme": scheme,
                **{f"weight_{name}": weight for name, weight in weights.items()},
                "fixed_affordability_median": affordability_median,
                "fixed_liveability_threshold_median": liveability_median,
                "candidate_count_after_existing_filters": len(candidate_codes),
                "baseline_candidate_count": len(baseline_codes),
                "candidate_overlap_with_baseline": intersection,
                "candidate_jaccard_with_baseline": intersection / union if union else np.nan,
                "top10_overlap_with_baseline": len(ranked_codes & baseline_top),
                "spearman_score_correlation_with_baseline": float(score.corr(frame["preliminary_liveability_score"], method="spearman")),
            }
        )
    summary = pd.DataFrame(summary_rows)
    candidate_table = pd.DataFrame(candidate_rows)
    audit = {
        "baseline_method_unchanged": True,
        "baseline_candidates_verified": int(len(baseline_codes)),
        "baseline_candidate_source": "preliminary_liveable_affordable_candidates_filtered.csv",
        "affordability_cutoff_and_liveability_cutoff_held_fixed": True,
        "minimum_listing_count": 10,
        "special_purpose_exclusions_unchanged": True,
        "weights_pre_specified_not_selected_for_attractive_area_results": True,
        "weights": Q3_WEIGHT_SCHEMES,
        "rent_period_and_income_period_mismatch": "Current asking-rent observations and 2021 Census median household income refer to different periods; affordability burden is comparative screening only, not a precise current budget estimate.",
        "sa2_is_not_automatically_a_suburb": True,
        "existing_q3_screen_changed": False,
    }
    return summary, candidate_table, audit


def run_q3_weight_sensitivity(project_root: str | Path) -> dict[str, Any]:
    root = _root_path(project_root)
    areas = _read_csv(root, "data/processed/sprint5/preliminary_liveability_affordability.csv", dtype={"sa2_code": "string"})
    baseline = _read_csv(root, "data/processed/sprint5/preliminary_liveable_affordable_candidates_filtered.csv", dtype={"sa2_code": "string"})
    summary, candidates, audit = compute_q3_weight_sensitivity(areas, baseline)
    out = root / "data" / "processed" / "sprint6" / "q3_weight_sensitivity"
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "q3_weight_sensitivity_summary.csv", index=False)
    candidates.to_csv(out / "q3_weight_sensitivity_candidates.csv", index=False)
    (out / "q3_weight_sensitivity_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"summary": summary, "candidates": candidates, "audit": audit, "output_dir": out}


def run_all_validations(project_root: str | Path) -> dict[str, Any]:
    root = _root_path(project_root)
    return {
        "validation_12q": run_12q_historical_validation(root),
        "direct_growth_leakage_audit": run_direct_growth_leakage_audit(root),
        "q1_cross_validation": run_q1_cross_validation(root),
        "q3_weight_sensitivity": run_q3_weight_sensitivity(root),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    results = run_all_validations(root)
    print("Offline Sprint 6 methodological validation complete.")
    for name, value in results.items():
        print(f"{name}: {value['output_dir'].relative_to(root)}")
    metrics = results["validation_12q"]["metrics"]
    print("Best 12Q path RMSE:", metrics.loc[metrics["path_rmse_aud_week"].idxmin(), "model_label"])
    print("Best 12Q endpoint RMSE:", metrics.loc[metrics["endpoint_rmse_aud_week"].idxmin(), "model_label"])


if __name__ == "__main__":
    main()

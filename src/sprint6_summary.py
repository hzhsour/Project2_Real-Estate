"""Offline Sprint 6 synthesis from saved Sprint 1–5 outputs.

This module performs deterministic table joins, validation, ranking, and plotting
from existing processed files. It does not fit a model, download data, call an
API, rerun ORS, or modify any Sprint 1–5 file.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUTS = {
    "sprint3_audit": "data/processed/sprint3/audit.json",
    "sprint3_saved_notebook_qa": "notebooks/03_sprint3_accessibility.ipynb",
    "sprint4_model_comparison": "data/processed/sprint4/feature_model_comparison.csv",
    "sprint4_feature_importance": "data/processed/sprint4/feature_importance.csv",
    "q1_feature_summary": "data/processed/sprint5/final_integration/q1_final_feature_summary.csv",
    "feature_catalogue": "data/processed/sprint5/final_integration/project_feature_catalogue.csv",
    "feature_crosswalk": "data/processed/sprint5/final_integration/q1_q2_q3_feature_crosswalk.csv",
    "sa2_forecasts": "data/processed/sprint5/sprint5_rent_forecasts_audited.csv",
    "history_group_audit": "data/processed/sprint5/rental_history_group_audit.csv",
    "distinct_history_groups": "data/processed/sprint5/model_extension/distinct_history_groups.csv",
    "saved_five_year_top10": "data/processed/sprint5/model_refinement/final_distinct_top10_with_stability.csv",
    "model_decision": "data/processed/sprint5/model_refinement/final_model_decision.json",
    "scale_free_validation": "data/processed/sprint5/model_refinement/scale_free_validation.csv",
    "five_year_model_comparison": "data/processed/sprint5/q2_final_summary/q2_model_comparison.csv",
    "five_year_backtest_audit": "data/processed/sprint5/long_history_validation/five_year_validation_audit.json",
    "archive_matching_audit": "data/processed/sprint5/long_history_validation/source_market_matching_audit.csv",
    "history_integration_audit": "data/processed/sprint5/long_history_validation/sa2_history_integration_experiment/integration_audit.json",
    "direct_growth_metrics": "data/processed/sprint5/direct_growth_validation/direct_growth_model_metrics.csv",
    "direct_growth_audit": "data/processed/sprint5/direct_growth_validation/direct_growth_validation_audit.json",
    "q2_q3_geography": "data/processed/sprint5/final_integration/q2_q3_geography_matching.csv",
    "balanced_candidates": "data/processed/sprint5/final_integration/balanced_candidate_summary.csv",
    "integrated_takeaways": "data/processed/sprint5/final_integration/final_project_takeaways.json",
    "q3_all_areas": "data/processed/sprint5/preliminary_liveability_affordability.csv",
    "q3_candidates": "data/processed/sprint5/preliminary_liveable_affordable_candidates_filtered.csv",
    "q3_audit": "data/processed/sprint5/preliminary_q3_audit.json",
}

OUTPUT_FILES = [
    "sprint6_5year_forecasts.csv",
    "sprint6_5year_top10.csv",
    "sprint6_3year_forecasts.csv",
    "sprint6_3year_top10.csv",
    "sprint6_horizon_comparison.csv",
    "sprint6_q2_group_eligibility_audit.csv",
    "sprint6_q1_summary.csv",
    "sprint6_q1_feature_importance.csv",
    "sprint6_q3_summary.csv",
    "sprint6_final_recommendations.csv",
    "sprint6_final_takeaways.json",
    "sprint6_audit.json",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping_key(value: object) -> str:
    if pd.isna(value):
        return ""
    parts = sorted(part.strip().casefold() for part in str(value).split(";") if part.strip())
    return " | ".join(parts)


def _as_bool(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    return values.astype("string").str.strip().str.lower().isin({"true", "1", "yes", "y"})


def _require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _quarter_distance(start: str, end: str) -> int:
    first = pd.Period(pd.Timestamp(start), freq="Q-DEC")
    last = pd.Period(pd.Timestamp(end), freq="Q-DEC")
    return int(last.ordinal - first.ordinal)


def _save_workflow_figure(path: Path) -> None:
    labels = [
        "Listing sources\nDomain + Kaggle",
        "Cleaning +\nprovenance",
        "ABS 2021\nSA2 assignment",
        "Socioeconomic,\ntransport + amenities",
        "Q1: current-rent\nfeature evidence",
        "Q2: historical rent\nforecast scenarios",
        "Q3: affordability +\nliveability",
        "Stakeholder\nrecommendations",
    ]
    fig, ax = plt.subplots(figsize=(16, 3.1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    centers = np.linspace(0.065, 0.935, len(labels))
    width = 0.105
    height = 0.43
    for idx, (x, label) in enumerate(zip(centers, labels, strict=True)):
        color = "#EAF2F8" if idx < 4 else "#EAF4EA"
        rect = plt.Rectangle(
            (x - width / 2, 0.31), width, height,
            facecolor=color, edgecolor="#486581", linewidth=1.2,
            transform=ax.transAxes,
        )
        ax.add_patch(rect)
        ax.text(
            x, 0.525, label, ha="center", va="center",
            fontsize=8.6, color="#243B53", transform=ax.transAxes,
        )
        if idx < len(labels) - 1:
            ax.annotate(
                "", xy=(centers[idx + 1] - width / 2 - 0.006, 0.525),
                xytext=(x + width / 2 + 0.006, 0.525),
                xycoords=ax.transAxes, textcoords=ax.transAxes,
                arrowprops={"arrowstyle": "->", "color": "#627D98", "lw": 1.3},
            )
    ax.text(
        0.5, 0.91,
        "Project workflow: one integrated data foundation, three question-specific analyses",
        ha="center", va="center", fontsize=12, weight="bold",
        color="#102A43", transform=ax.transAxes,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _save_q1_figure(
    path: Path,
    model_summary: pd.DataFrame,
    importance: pd.DataFrame,
) -> None:
    colors = {
        "Internal property characteristics": "#D97706",
        "Socioeconomic context": "#3B82F6",
        "Transport accessibility": "#16A34A",
        "Amenities": "#8B5CF6",
    }
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), gridspec_kw={"width_ratios": [1.1, 1.35]})

    perf = model_summary.sort_values("rmse_aud_week", ascending=True)
    axes[0].barh(
        perf["feature_set_label"], perf["rmse_aud_week"],
        color="#4C78A8", edgecolor="white",
    )
    axes[0].set_title("Random Forest holdout error")
    axes[0].set_xlabel("RMSE (AUD/week; lower is better)")
    axes[0].grid(axis="x", alpha=0.22)
    axes[0].invert_yaxis()
    for y, value in enumerate(perf["rmse_aud_week"]):
        axes[0].text(value + 1.0, y, f"{value:.1f}", va="center", fontsize=8.5)

    imp = importance.sort_values("mae_increase_when_shuffled", ascending=True)
    bar_colors = [colors.get(category, "#94A3B8") for category in imp["feature_category"]]
    axes[1].barh(
        imp["feature"], imp["mae_increase_when_shuffled"],
        color=bar_colors, edgecolor="white",
    )
    axes[1].set_title("Permutation importance in the full-feature model")
    axes[1].set_xlabel("Increase in holdout MAE when shuffled (AUD/week)")
    axes[1].grid(axis="x", alpha=0.22)
    axes[1].invert_yaxis()
    for y, value in enumerate(imp["mae_increase_when_shuffled"]):
        axes[1].text(value + 0.35, y, f"{value:.1f}", va="center", fontsize=8)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=color, label=label)
        for label, color in colors.items()
        if label in set(imp["feature_category"])
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.91),
               ncol=4, frameon=False, fontsize=8)
    fig.suptitle("Q1 — Current asking-rent feature evidence", fontsize=15, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _save_five_year_figure(path: Path, top10: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    growth = "predicted_growth_5y_pct"
    view = top10.sort_values(growth, ascending=True)
    bars = ax.barh(
        view["rental_market"], view[growth],
        color="#3977A8", edgecolor="white", height=0.72,
    )
    ax.set_title("Predicted Five-Year Rental Growth by Rental Market, 2025–2030", fontsize=14, weight="bold", pad=15)
    ax.text(
        0, 1.015,
        "Damped Holt scenario — not a definitive prediction of future rankings",
        transform=ax.transAxes, fontsize=9, color="#52606D",
    )
    ax.set_xlabel("Predicted change in moving annual median rent (%)")
    ax.set_ylabel("Homes Victoria rental-history group")
    ax.grid(axis="x", alpha=0.22)
    ax.set_axisbelow(True)
    ax.bar_label(bars, labels=[f"{v:.1f}%" for v in view[growth]], padding=4, fontsize=8.5)
    ax.set_xlim(0, max(view[growth]) * 1.22)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _save_horizon_figure(path: Path, comparison: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 6.3))
    top = comparison["top10_5y"]
    ax.scatter(
        comparison.loc[~top, "growth_3y_pct"],
        comparison.loc[~top, "growth_5y_pct"],
        s=26, alpha=0.42, color="#94A3B8", edgecolors="none",
        label="Other eligible history groups",
    )
    ax.scatter(
        comparison.loc[top, "growth_3y_pct"],
        comparison.loc[top, "growth_5y_pct"],
        s=48, alpha=0.9, color="#2F855A", edgecolors="white", linewidths=0.6,
        label="Damped Holt five-year Top 10",
    )
    maximum = float(max(comparison["growth_3y_pct"].max(), comparison["growth_5y_pct"].max()))
    ax.plot([0, maximum * 1.04], [0, maximum * 1.04], linestyle="--", linewidth=1, color="#64748B", label="Equal growth rate")
    label_offsets = [(14, 13), (22, -19), (-155, -13)]
    for (_, row), offset in zip(
        comparison.loc[top].sort_values("rank_5y").head(3).iterrows(),
        label_offsets,
        strict=False,
    ):
        ax.annotate(
            str(row["rental_market"]),
            (row["growth_3y_pct"], row["growth_5y_pct"]),
            xytext=offset, textcoords="offset points", fontsize=7,
            arrowprops={"arrowstyle": "-", "color": "#64748B", "lw": 0.7},
        )
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Three-year predicted rent growth, 2025–2028 (%)")
    ax.set_ylabel("Five-year predicted rent growth, 2025–2030 (%)")
    ax.set_title("Damped Holt growth scenarios across eligible markets", fontsize=13, weight="bold")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _save_q3_figure(path: Path, all_areas: pd.DataFrame, candidates: pd.DataFrame, q3_audit: dict[str, Any]) -> None:
    candidate_codes = set(candidates["sa2_code"].astype("string"))
    shown = all_areas.copy()
    shown["sa2_code"] = shown["sa2_code"].astype("string")
    marked = shown["sa2_code"].isin(candidate_codes)
    affordability_median = float(q3_audit["affordability_median"])
    liveability_median = float(q3_audit["liveability_median"])

    fig, ax = plt.subplots(figsize=(9.0, 6.3))
    ax.scatter(
        shown.loc[~marked, "affordability_burden"],
        shown.loc[~marked, "preliminary_liveability_score"],
        s=22, color="#9AA5B1", alpha=0.42, label="Other complete SA2s",
    )
    ax.scatter(
        shown.loc[marked, "affordability_burden"],
        shown.loc[marked, "preliminary_liveability_score"],
        s=35, color="#2F9E44", alpha=0.82, edgecolors="white", linewidths=0.45,
        label="105 screened candidates",
    )
    ax.axvline(
        affordability_median, color="#475569", linestyle="--", linewidth=1.1,
        label="Median affordability burden",
    )
    ax.axhline(
        liveability_median, color="#475569", linestyle=":", linewidth=1.2,
        label="Median liveability score",
    )
    label_rows = candidates.nsmallest(2, "candidate_rank")
    label_offsets = [(-112, -27), (18, 13)]
    for (_, row), offset in zip(label_rows.iterrows(), label_offsets, strict=False):
        ax.annotate(
            str(row["sa2_name"]),
            (row["affordability_burden"], row["preliminary_liveability_score"]),
            xytext=offset, textcoords="offset points", fontsize=7,
            arrowprops={"arrowstyle": "-", "color": "#475569", "lw": 0.65},
        )
    ax.set_xlabel("Affordability burden: median rent ÷ household income (lower is smaller rent share)")
    ax.set_ylabel("Preliminary liveability score (0–1)")
    ax.set_title("Q3 — Preliminary affordability and liveability across SA2s", fontsize=13, weight="bold")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _answer_q1(model_summary: pd.DataFrame, importance: pd.DataFrame) -> str:
    by_set = model_summary.set_index("feature_set")
    property_rmse = float(by_set.loc["internal_property", "rmse_aud_week"])
    full_rmse = float(by_set.loc["internal_socioeconomic_transport_amenities", "rmse_aud_week"])
    socio_rmse = float(by_set.loc["internal_socioeconomic", "rmse_aud_week"])
    sample = int(model_summary["test_rows"].iloc[0])
    irsad = importance.loc[importance["feature"].eq("irsad_score")].iloc[0]
    return (
        f"On the saved Random Forest holdout, the full property, socioeconomic, transport and amenity "
        f"feature set produced an RMSE of AUD {full_rmse:.2f} per week, compared with AUD "
        f"{property_rmse:.2f} for property characteristics alone and AUD {socio_rmse:.2f} for the "
        f"socioeconomic-only set. The comparison covers {sample} held-out SA2 feature rows, not "
        f"{sample} individual listings. IRSAD had the largest saved permutation-importance value: "
        f"shuffling it increased holdout MAE by about AUD {float(irsad['mae_increase_when_shuffled']):.2f} "
        f"per week (±{float(irsad['shuffle_std']):.2f}). This indicates model reliance in this split; "
        "it does not show that IRSAD causes rent differences or indicate a causal effect. Adding "
        "transport alone to the socioeconomic set slightly increased RMSE, while the complete set "
        "had the lowest error. Overall, the wider context provides useful predictive information in "
        "this saved comparison, but one holdout is limited evidence—not proof of universal importance."
    )


def _answer_q3(candidate_count: int, initial_count: int, minimum_listings: int) -> str:
    return (
        "Affordability burden is current SA2 median weekly asking rent divided by 2021 Census median "
        "household weekly income; lower values mean rent is a smaller share of income. Preliminary "
        "liveability averages five equally weighted, percentile-scaled dimensions: nearest train, "
        "nearest school, and nearby park, shopping and entertainment provision. The Victoria-median "
        f"thresholds first identified {initial_count} SA2s; the existing minimum of {minimum_listings} "
        f"listings and special-purpose exclusion leave {candidate_count} illustrative candidates. "
        "This is a transparent comparison, not a complete housing-affordability measure or definitive "
        "liveability index. It omits factors such as housing quality, safety, household preferences and "
        "many individual circumstances; the dimensions and equal weights are provisional. The analysis "
        "unit is SA2, which should not automatically be described as a suburb. These current candidates "
        "are separate from Q2's future-rent-growth scenarios and are not a direct measure of demand."
    )


def build_sprint6_outputs(project_root: str | Path) -> dict[str, Any]:
    """Validate and write Sprint 6 tables/figures using only saved local inputs."""

    root = Path(project_root).resolve()
    missing = [relative for relative in INPUTS.values() if not (root / relative).is_file()]
    if missing:
        raise FileNotFoundError(
            "Sprint 6 reads saved Sprint 1–5 outputs only. Missing prerequisites:\n- "
            + "\n- ".join(missing)
            + "\nGenerate the documented upstream outputs first; this function will not rebuild them."
        )

    input_hashes = {relative: _sha256(root / relative) for relative in INPUTS.values()}

    def read_csv(key: str, **kwargs: Any) -> pd.DataFrame:
        return pd.read_csv(root / INPUTS[key], low_memory=False, **kwargs)

    def read_json(key: str) -> dict[str, Any]:
        return json.loads((root / INPUTS[key]).read_text(encoding="utf-8"))

    q1_models_raw = read_csv("sprint4_model_comparison")
    q1_importance_raw = read_csv("sprint4_feature_importance")
    q1_features_raw = read_csv("q1_feature_summary")
    feature_crosswalk = read_csv("feature_crosswalk")
    q1_models = q1_models_raw.loc[q1_models_raw["model"].eq("random_forest")].copy()
    _require_columns(
        q1_models,
        ["feature_set", "test_rows", "MAE_AUD_week", "RMSE_AUD_week", "R2"],
        "Q1 model comparison",
    )
    if q1_models["feature_set"].duplicated().any():
        raise ValueError("Q1 Random Forest feature-set rows are not unique.")
    labels = {
        "internal_property": "Property only",
        "internal_socioeconomic": "Property + socioeconomic",
        "internal_socioeconomic_transport": "Property + socioeconomic + transport",
        "internal_socioeconomic_transport_amenities": "Property + socioeconomic + transport + amenities",
    }
    q1_models["feature_set_label"] = q1_models["feature_set"].map(labels).fillna(q1_models["feature_set"])
    q1_summary = q1_models.rename(
        columns={
            "test_rows": "test_rows",
            "MAE_AUD_week": "mae_aud_week",
            "RMSE_AUD_week": "rmse_aud_week",
            "R2": "r2",
        }
    )[
        ["feature_set", "feature_set_label", "model", "test_rows", "mae_aud_week", "rmse_aud_week", "r2"]
    ].sort_values("rmse_aud_week", ascending=True).reset_index(drop=True)

    q1_features = q1_importance_raw.loc[
        q1_importance_raw["feature_set"].eq("internal_socioeconomic_transport_amenities")
        & q1_importance_raw["model"].eq("random_forest")
    ].copy()
    q1_features = q1_features.merge(
        q1_features_raw[["Feature", "feature_category", "direction_relationship", "interpretation"]].rename(
            columns={"Feature": "feature"}
        ),
        on="feature",
        # The saved integrated interpretation catalogue intentionally covers
        # its curated top-10 importance features, not every fitted predictor.
        how="inner",
        validate="one_to_one",
    )
    if len(q1_features) != 10 or q1_features["feature_category"].isna().any():
        raise ValueError("The saved Q1 interpretation table should classify its ten displayed importance features.")
    q1_features = q1_features.sort_values(
        "MAE_increase_when_shuffled", ascending=False
    ).head(10).rename(
        columns={
            "MAE_increase_when_shuffled": "mae_increase_when_shuffled",
            "shuffle_std": "shuffle_std",
        }
    ).reset_index(drop=True)
    q1_answer = _answer_q1(q1_summary, q1_features)

    history_audit = read_csv("history_group_audit", dtype={"history_fingerprint": "string"})
    distinct_groups = read_csv("distinct_history_groups", dtype={"history_fingerprint": "string"})
    raw_forecasts = read_csv(
        "sa2_forecasts",
        dtype={"sa2_code": "string", "history_fingerprint": "string"},
    )
    saved_top10 = read_csv("saved_five_year_top10", dtype={"source_rental_history_group": "string"})
    archive_matching = read_csv("archive_matching_audit")
    q2_q3_geo = read_csv("q2_q3_geography")
    model_decision = read_json("model_decision")
    if history_audit["history_fingerprint"].duplicated().any():
        raise ValueError("The history-group audit has duplicate fingerprints.")
    fingerprints = set(history_audit["history_fingerprint"].dropna().astype(str))
    if len(fingerprints) != 145 or len(distinct_groups) != 145:
        raise ValueError(
            f"Expected the saved 145 distinct trajectories; found {len(fingerprints)} in the audit "
            f"and {len(distinct_groups)} in the distinct-history table."
        )
    if set(distinct_groups["history_fingerprint"].dropna().astype(str)) != fingerprints:
        raise ValueError("The two saved distinct-history audits disagree on trajectory fingerprints.")
    _require_columns(
        history_audit,
        [
            "source_rental_history_group", "history_fingerprint", "number_of_mapped_sa2s",
            "mapped_sa2_names", "representative_residential_sa2", "other_mapped_sa2s",
            "special_purpose_mapped_sa2s", "latest_rent", "forecast_3y_rent",
            "forecast_5y_rent", "predicted_growth_5y_pct", "representative_listing_count",
            "representative_is_special_purpose_sa2", "main_q2_eligible",
        ],
        "Q2 history-group audit",
    )
    history_audit["main_q2_eligible"] = _as_bool(history_audit["main_q2_eligible"])
    history_audit["representative_is_special_purpose_sa2"] = _as_bool(
        history_audit["representative_is_special_purpose_sa2"]
    )

    raw_group_columns = [
        "latest_observed_period", "forecast_3y_period", "forecast_5y_period",
        "latest_observed_rent", "forecast_3y_rent", "forecast_5y_rent",
        "forecast_sanity_status",
    ]
    _require_columns(
        raw_forecasts,
        ["history_fingerprint", "sa2_code", "sa2_name", *raw_group_columns],
        "Saved SA2 forecast table",
    )
    raw_nunique = raw_forecasts.groupby("history_fingerprint", dropna=False)[raw_group_columns].nunique(dropna=False)
    if raw_nunique.gt(1).any().any():
        raise ValueError("Mapped SA2 rows sharing a history fingerprint have inconsistent forecasts or dates.")
    if not raw_forecasts["forecast_sanity_status"].eq("included").all():
        raise ValueError("The saved selected-model forecast table includes a non-included row.")
    raw_meta = (
        raw_forecasts.sort_values(["sa2_name", "sa2_code"])
        .drop_duplicates("history_fingerprint")
        [["history_fingerprint", *raw_group_columns]]
        .rename(
            columns={
                "latest_observed_period": "latest_observed_period",
                "forecast_3y_period": "forecast_3y_period",
                "forecast_5y_period": "forecast_5y_period",
                "latest_observed_rent": "raw_latest_rent",
                "forecast_3y_rent": "raw_forecast_3y_rent",
                "forecast_5y_rent": "raw_forecast_5y_rent",
                "forecast_sanity_status": "forecast_sanity_status",
            }
        )
    )
    if set(raw_meta["history_fingerprint"].dropna().astype(str)) != fingerprints:
        raise ValueError("The saved SA2 forecasts and history-group audit do not cover the same trajectories.")

    groups = history_audit.merge(
        raw_meta, on="history_fingerprint", how="left", validate="one_to_one"
    )
    for saved, raw_name in [
        ("latest_rent", "raw_latest_rent"),
        ("forecast_3y_rent", "raw_forecast_3y_rent"),
        ("forecast_5y_rent", "raw_forecast_5y_rent"),
    ]:
        if not np.allclose(groups[saved], groups[raw_name], rtol=0, atol=1e-9, equal_nan=False):
            raise ValueError(f"Group-audit {saved} differs from the saved SA2 forecast values.")
    if groups[["latest_observed_period", "forecast_3y_period", "forecast_5y_period"]].nunique().max() != 1:
        raise ValueError("The saved market forecasts do not share one consistent set of endpoint dates.")
    period_row = groups.iloc[0]
    horizon_3y = _quarter_distance(period_row["latest_observed_period"], period_row["forecast_3y_period"])
    horizon_5y = _quarter_distance(period_row["latest_observed_period"], period_row["forecast_5y_period"])
    if (horizon_3y, horizon_5y) != (12, 20):
        raise ValueError(f"Forecast endpoints imply {horizon_3y} and {horizon_5y} quarters, not 12 and 20.")

    groups["growth_3y_pct"] = (groups["forecast_3y_rent"] - groups["latest_rent"]) / groups["latest_rent"] * 100.0
    formula_5y = (groups["forecast_5y_rent"] - groups["latest_rent"]) / groups["latest_rent"] * 100.0
    if not np.allclose(groups["predicted_growth_5y_pct"], formula_5y, rtol=0, atol=1e-8):
        raise ValueError("Saved five-year growth percentages do not agree with the saved rent forecasts.")
    groups["growth_5y_pct"] = groups["predicted_growth_5y_pct"]
    groups["rank_3y"] = groups["growth_3y_pct"].rank(method="min", ascending=False).astype(int)
    groups["rank_5y"] = groups["growth_5y_pct"].rank(method="min", ascending=False).astype(int)

    archive_matching["_mapping_key"] = archive_matching["all_mapped_sa2s"].map(_mapping_key)
    q2_q3_geo["_mapping_key"] = q2_q3_geo["mapped_sa2s"].map(_mapping_key)
    groups["_mapping_key"] = groups["mapped_sa2_names"].map(_mapping_key)
    if archive_matching["_mapping_key"].duplicated().any() or q2_q3_geo["_mapping_key"].duplicated().any():
        raise ValueError("Saved geographic audits do not have unique mapped-SA2 keys.")
    archive_cols = [
        "_mapping_key", "match_type", "matched_source_markets",
        "unmatched_group_components", "ambiguous_group_components", "match_methods",
    ]
    q2_geo_cols = [
        "_mapping_key", "geography_mapping_class", "geography_confidence",
        "q3_candidate_sa2s", "q3_candidate_count", "q3_candidate_rule_status",
    ]
    groups = groups.merge(
        archive_matching[archive_cols].rename(
            columns={
                "match_type": "archive_match_type",
                "matched_source_markets": "archive_matched_source_markets",
                "unmatched_group_components": "archive_unmatched_components",
                "ambiguous_group_components": "archive_ambiguous_components",
                "match_methods": "archive_match_methods",
            }
        ),
        on="_mapping_key", how="left", validate="one_to_one", indicator="_archive_merge",
    )
    groups = groups.merge(
        q2_q3_geo[q2_geo_cols],
        on="_mapping_key", how="left", validate="one_to_one", indicator="_geo_merge",
    )
    if not groups["_archive_merge"].eq("both").all() or not groups["_geo_merge"].eq("both").all():
        raise ValueError("Some forecast groups do not resolve through the saved geographic audits.")
    groups = groups.drop(columns=["_mapping_key", "_archive_merge", "_geo_merge"])
    groups["eligibility_status"] = np.where(
        groups["main_q2_eligible"],
        "Eligible for the primary Q2 ranking",
        "Excluded from the primary Q2 ranking by the existing special-purpose rule",
    )
    groups["geography_caveat"] = groups.apply(
        lambda row: (
            "One-to-one Q2–Q3 SA2 match; SA2 is still a statistical area, not automatically a suburb."
            if row["geography_mapping_class"] == "one-to-one"
            else (
                f"Maps to {int(row['q3_candidate_count']) if pd.notna(row['q3_candidate_count']) else int(row['number_of_mapped_sa2s'])} "
                "SA2 area(s); one rent-history trajectory is counted once."
                if row["geography_mapping_class"] == "one-to-many"
                else "Ambiguous multi-source/geography match; do not interpret as one independently observed suburb."
            )
        ),
        axis=1,
    )

    eligible = groups.loc[groups["main_q2_eligible"]].copy()
    if len(eligible) != 144:
        raise ValueError(f"Expected 144 groups under the saved Q2 eligibility rule; found {len(eligible)}.")
    if eligible["history_fingerprint"].duplicated().any():
        raise ValueError("The eligible Q2 table contains duplicate historical trajectories.")

    saved_top10 = saved_top10.sort_values("rank").copy()
    computed_top5 = eligible.sort_values(
        ["growth_5y_pct", "source_rental_history_group"],
        ascending=[False, True],
    ).head(10)
    computed_top3 = eligible.sort_values(
        ["growth_3y_pct", "source_rental_history_group"],
        ascending=[False, True],
    ).head(10)
    if len(saved_top10) != 10 or len(computed_top5) != 10 or len(computed_top3) != 10:
        raise ValueError("A selected Q2 Top 10 output does not contain exactly ten distinct groups.")
    if saved_top10["source_rental_history_group"].tolist() != computed_top5["source_rental_history_group"].tolist():
        raise ValueError("The saved five-year Top 10 does not match the eligible distinct-history ranking.")
    saved_enriched = saved_top10.merge(
        groups,
        on="source_rental_history_group",
        how="left",
        validate="one_to_one",
        suffixes=("_saved", ""),
    )
    if saved_enriched["history_fingerprint"].isna().any():
        raise ValueError("A saved five-year Top 10 market did not match its audited history group.")
    if not np.allclose(
        saved_enriched["forecast_5y_rent"],
        saved_enriched["forecast_5y_rent_saved"],
        rtol=0, atol=1e-9,
    ):
        raise ValueError("A saved five-year Top 10 forecast differs from the history-group audit.")
    if not np.allclose(
        saved_enriched["damped_holt_3y_growth_pct"],
        saved_enriched["growth_3y_pct"],
        rtol=0, atol=1e-8,
    ):
        raise ValueError("A saved Top 10 three-year growth value differs from the saved forecast.")
    if not np.allclose(
        saved_enriched["damped_holt_5y_growth_pct"],
        saved_enriched["growth_5y_pct"],
        rtol=0, atol=1e-8,
    ):
        raise ValueError("A saved Top 10 five-year growth value differs from the saved forecast.")

    top5 = saved_enriched.sort_values("rank").copy()
    top5["sensitivity_status"] = np.where(
        _as_bool(top5["stable_growth_candidate"]),
        "Stable within the saved path-model sensitivity rule",
        "Model-sensitive within the saved path-model comparison",
    )
    top5_output = top5[
        [
            "rank", "source_rental_history_group", "representative_sa2",
            "other_mapped_sa2s", "mapped_sa2_names", "latest_observed_period",
            "latest_rent", "forecast_5y_period", "forecast_5y_rent",
            "growth_5y_pct", "sensitivity_status", "models_in_top10",
            "stable_growth_candidate", "eligibility_status",
            "archive_match_type", "archive_matched_source_markets",
            "archive_unmatched_components", "archive_ambiguous_components",
            "geography_mapping_class", "geography_confidence", "geography_caveat",
        ]
    ].rename(
        columns={
            "source_rental_history_group": "rental_market",
            "mapped_sa2_names": "all_mapped_sa2s",
            "latest_observed_period": "current_period",
            "latest_rent": "current_rent_aud_week",
            "forecast_5y_period": "forecast_period",
            "forecast_5y_rent": "forecast_2030_aud_week",
            "growth_5y_pct": "predicted_growth_5y_pct",
        }
    )
    top3 = computed_top3.copy()
    top3.insert(0, "rank", range(1, len(top3) + 1))
    top3_output = top3[
        [
            "rank", "source_rental_history_group", "representative_residential_sa2",
            "other_mapped_sa2s", "mapped_sa2_names", "latest_observed_period",
            "latest_rent", "forecast_3y_period", "forecast_3y_rent",
            "growth_3y_pct", "rank_5y", "eligibility_status",
            "archive_match_type", "archive_matched_source_markets",
            "archive_unmatched_components", "archive_ambiguous_components",
            "geography_mapping_class", "geography_confidence", "geography_caveat",
        ]
    ].rename(
        columns={
            "source_rental_history_group": "rental_market",
            "representative_residential_sa2": "representative_sa2",
            "mapped_sa2_names": "all_mapped_sa2s",
            "latest_observed_period": "current_period",
            "latest_rent": "current_rent_aud_week",
            "forecast_3y_period": "forecast_period",
            "forecast_3y_rent": "forecast_2028_aud_week",
            "growth_3y_pct": "predicted_growth_3y_pct",
            "rank_5y": "rank_5y",
        }
    )

    eligible_output = eligible.copy()
    five_year_full = eligible_output[
        [
            "rank_5y", "source_rental_history_group", "history_fingerprint",
            "representative_residential_sa2", "other_mapped_sa2s", "mapped_sa2_names",
            "number_of_mapped_sa2s", "special_purpose_mapped_sa2s",
            "representative_listing_count", "eligibility_status",
            "latest_observed_period", "latest_rent", "forecast_5y_period",
            "forecast_5y_rent", "growth_5y_pct", "archive_match_type",
            "archive_matched_source_markets", "archive_unmatched_components",
            "archive_ambiguous_components", "geography_mapping_class",
            "geography_confidence", "geography_caveat",
        ]
    ].rename(
        columns={
            "source_rental_history_group": "rental_market",
            "representative_residential_sa2": "representative_sa2",
            "mapped_sa2_names": "all_mapped_sa2s",
            "number_of_mapped_sa2s": "mapped_sa2_count",
            "representative_listing_count": "representative_listing_count",
            "latest_observed_period": "current_period",
            "latest_rent": "current_rent_aud_week",
            "forecast_5y_period": "forecast_period",
            "forecast_5y_rent": "forecast_2030_aud_week",
            "growth_5y_pct": "predicted_growth_5y_pct",
        }
    ).sort_values("rank_5y")

    three_year_full = eligible.copy()
    three_year_full = three_year_full[
        [
            "rank_3y", "source_rental_history_group", "history_fingerprint",
            "representative_residential_sa2", "other_mapped_sa2s", "mapped_sa2_names",
            "number_of_mapped_sa2s", "special_purpose_mapped_sa2s",
            "representative_listing_count", "eligibility_status",
            "latest_observed_period", "latest_rent", "forecast_3y_period",
            "forecast_3y_rent", "growth_3y_pct", "archive_match_type",
            "archive_matched_source_markets", "archive_unmatched_components",
            "archive_ambiguous_components", "geography_mapping_class",
            "geography_confidence", "geography_caveat",
        ]
    ].rename(
        columns={
            "source_rental_history_group": "rental_market",
            "representative_residential_sa2": "representative_sa2",
            "mapped_sa2_names": "all_mapped_sa2s",
            "number_of_mapped_sa2s": "mapped_sa2_count",
            "latest_observed_period": "current_period",
            "latest_rent": "current_rent_aud_week",
            "forecast_3y_period": "forecast_period",
            "forecast_3y_rent": "forecast_2028_aud_week",
            "growth_3y_pct": "predicted_growth_3y_pct",
        }
    ).sort_values("rank_3y")

    horizon_comparison = eligible[
        [
            "history_fingerprint", "source_rental_history_group",
            "representative_residential_sa2", "latest_rent",
            "forecast_3y_rent", "growth_3y_pct", "rank_3y",
            "forecast_5y_rent", "growth_5y_pct", "rank_5y",
        ]
    ].copy()
    horizon_comparison["top10_3y"] = horizon_comparison["rank_3y"].le(10)
    horizon_comparison["top10_5y"] = horizon_comparison["rank_5y"].le(10)
    horizon_comparison = horizon_comparison.rename(
        columns={
            "source_rental_history_group": "rental_market",
            "representative_residential_sa2": "representative_sa2",
            "latest_rent": "current_rent_aud_week",
            "forecast_3y_rent": "forecast_2028_aud_week",
            "growth_3y_pct": "growth_3y_pct",
            "rank_3y": "rank_3y",
            "forecast_5y_rent": "forecast_2030_aud_week",
            "growth_5y_pct": "growth_5y_pct",
            "rank_5y": "rank_5y",
        }
    ).sort_values("rank_5y")
    top3_ids = set(horizon_comparison.loc[horizon_comparison["top10_3y"], "history_fingerprint"])
    top5_ids = set(horizon_comparison.loc[horizon_comparison["top10_5y"], "history_fingerprint"])
    overlap_count = len(top3_ids & top5_ids)
    jaccard = overlap_count / len(top3_ids | top5_ids) if top3_ids | top5_ids else float("nan")
    spearman = float(
        horizon_comparison["growth_3y_pct"].corr(horizon_comparison["growth_5y_pct"], method="spearman")
    )
    horizon_summary = pd.DataFrame(
        [
            {
                "eligible_distinct_history_groups": len(eligible),
                "three_year_horizon_quarters": horizon_3y,
                "five_year_horizon_quarters": horizon_5y,
                "top10_overlap_count": overlap_count,
                "top10_jaccard_similarity": jaccard,
                "spearman_growth_rank_correlation": spearman,
            }
        ]
    )
    if overlap_count != 10:
        raise ValueError(f"Expected the saved Damped Holt horizons to overlap in 10 Top-10 groups; found {overlap_count}.")

    q1_importance_out = q1_features[
        [
            "feature", "feature_category", "direction_relationship",
            "mae_increase_when_shuffled", "shuffle_std", "interpretation",
        ]
    ].copy()

    short_medium = read_csv("scale_free_validation")
    _require_columns(short_medium, ["model", "horizon_quarters", "rmse", "mae", "mase", "smape"], "Scale-free validation")
    short_medium = short_medium.loc[short_medium["model"].isin(["damped_holt", "theta"])].copy()
    long_backtest = read_csv("five_year_model_comparison")
    direct_growth = read_csv("direct_growth_metrics")
    direct_growth_audit = read_json("direct_growth_audit")
    long_audit = read_json("five_year_backtest_audit")
    integration_audit = read_json("history_integration_audit")
    final_takeaways = read_json("integrated_takeaways")
    q3_all = read_csv("q3_all_areas", dtype={"sa2_code": "string"})
    q3_candidates = read_csv("q3_candidates", dtype={"sa2_code": "string"})
    q3_audit = read_json("q3_audit")
    balanced_candidates = read_csv("balanced_candidates")
    if len(balanced_candidates) != 0:
        raise ValueError("The saved final integration no longer reports zero balanced candidates.")
    if len(q3_candidates) != 105:
        raise ValueError(f"Expected 105 saved filtered Q3 candidates; found {len(q3_candidates)}.")
    if int(q3_audit["candidate_count"]) != 124:
        raise ValueError("The saved Q3 threshold audit no longer reports 124 initial candidates.")
    if q3_candidates["sa2_code"].duplicated().any():
        raise ValueError("The filtered Q3 candidate table contains duplicate SA2 codes.")

    q3_columns = [
        "candidate_rank", "sa2_code", "sa2_name", "listing_count",
        "current_median_weekly_rent", "median_household_weekly_income",
        "affordability_burden", "preliminary_liveability_score",
        "train_accessibility", "school_accessibility", "park_accessibility",
        "shopping_accessibility", "entertainment_accessibility",
        "affordability_reference_median", "liveability_reference_median",
    ]
    _require_columns(q3_candidates, q3_columns, "Filtered Q3 candidates")
    q3_summary = q3_candidates[q3_columns].sort_values("candidate_rank").reset_index(drop=True)
    q3_top10 = q3_summary.head(10).copy()
    minimum_listing_count = int(q3_candidates["minimum_listing_count"].max())

    route_audit = read_json("sprint3_audit")
    route_status = route_audit["routes"]["statuses"]
    sprint3_notebook = json.loads(
        (root / INPUTS["sprint3_saved_notebook_qa"]).read_text(encoding="utf-8")
    )
    cached_amenity_total = None
    for cell in sprint3_notebook.get("cells", []):
        if cell.get("cell_type") != "code" or "amenity_counts = pd.DataFrame" not in "".join(cell.get("source", [])):
            continue
        for cell_output in cell.get("outputs", []):
            if cell_output.get("output_type") != "display_data":
                continue
            cached_text = cell_output.get("data", {}).get("text/plain", "")
            if isinstance(cached_text, list):
                cached_text = "".join(cached_text)
            match = re.search(r"(?m)^\s*\d+\s+Total\s+(\d+)\s*$", cached_text)
            if match:
                cached_amenity_total = int(match.group(1))
                break
        if cached_amenity_total is not None:
            break
    if cached_amenity_total is None:
        raise ValueError("The saved Notebook 3 QA output has no cached statewide amenity total.")
    if cached_amenity_total != 35104:
        raise ValueError(f"Saved Notebook 3 QA reports {cached_amenity_total} amenities, not the reviewed 35,104.")
    q2_q3_counts = q2_q3_geo["geography_mapping_class"].value_counts().to_dict()
    mapping_counts = {
        name: int(q2_q3_counts.get(name, 0))
        for name in ["one-to-one", "one-to-many", "ambiguous", "unmatched"]
    }
    if mapping_counts != {"one-to-one": 54, "one-to-many": 65, "ambiguous": 26, "unmatched": 0}:
        raise ValueError(f"Saved Q2–Q3 mapping counts differ from the audited result: {mapping_counts}")

    # Mark the sole group that the existing Q2 special-purpose rule excludes.
    eligibility_audit = groups[
        [
            "source_rental_history_group", "history_fingerprint",
            "representative_residential_sa2", "mapped_sa2_names",
            "special_purpose_mapped_sa2s", "representative_is_special_purpose_sa2",
            "special_purpose_reason", "main_q2_eligible", "eligibility_status",
        ]
    ].rename(
        columns={
            "source_rental_history_group": "rental_market",
            "representative_residential_sa2": "representative_sa2",
            "mapped_sa2_names": "all_mapped_sa2s",
            "main_q2_eligible": "eligible_for_primary_q2_ranking",
        }
    )

    q3_answer = _answer_q3(len(q3_summary), int(q3_audit["candidate_count"]), minimum_listing_count)
    full_feature_rmse = float(
        q1_summary.loc[
            q1_summary["feature_set"].eq("internal_socioeconomic_transport_amenities"),
            "rmse_aud_week",
        ].iloc[0]
    )
    property_rmse = float(
        q1_summary.loc[q1_summary["feature_set"].eq("internal_property"), "rmse_aud_week"].iloc[0]
    )
    q1_test_rows = int(q1_summary["test_rows"].iloc[0])
    q2_model = str(model_decision["selected_model"])
    if q2_model != "damped_holt":
        raise ValueError(f"The saved selected Q2 path scenario is {q2_model}, not Damped Holt.")
    history_start = str(long_audit["source"]["date_start"])
    history_end = str(long_audit["source"]["date_end"])
    origins = int(long_audit["backtest"]["number_of_origins"])
    historical_markets = int(long_audit["backtest"]["eligible_markets"])
    model_count = len(long_audit["models"])
    per_model_paths = origins * historical_markets
    total_paths_all_models = per_model_paths * model_count
    if int(long_audit["backtest"]["total_forecast_paths"]) != total_paths_all_models:
        raise ValueError("The five-year audit's total path count is inconsistent with origins × markets × models.")
    q2_answer = (
        f"The overall project retains the saved Damped Holt five-year path scenario from "
        f"{period_row['latest_observed_period']} to {period_row['forecast_5y_period']}, ranked once per "
        f"eligible Homes Victoria history group ({len(eligible)} of {len(groups)} groups). The top ten "
        f"markets have predicted five-year growth from {top5_output['predicted_growth_5y_pct'].min():.1f}% "
        f"to {top5_output['predicted_growth_5y_pct'].max():.1f}%. This is a retained project scenario, "
        "not a claim that Damped Holt wins every horizon: in the separate direct 20-quarter historical "
        "backtest, Drift had lower rent-level error. Direct-growth models also show that exact high-growth "
        "rankings are difficult. Sprint 6 additionally requires an explicit three-year Damped Holt forecast "
        "output; it shares the same ten Top-10 groups here, but that ranking agreement does not validate either horizon."
    )

    recommendations = pd.DataFrame(
        [
            {
                "market_or_sa2": "SA2-level current-rent benchmarking",
                "relevant_question": "Q1",
                "scenario_or_screening_evidence": (
                    f"Full feature set RMSE AUD {full_feature_rmse:.2f}/week vs property-only "
                    f"AUD {property_rmse:.2f}/week on {q1_test_rows} held-out SA2 rows."
                ),
                "geography_confidence": "SA2 aggregate predictive comparison; one saved holdout.",
                "recommendation_category": "Contextual price benchmarking",
                "important_limitation": "Not a causal model or a precise property-specific valuation.",
            },
            {
                "market_or_sa2": "Eligible Homes Victoria rental-history groups",
                "relevant_question": "Q2",
                "scenario_or_screening_evidence": (
                    f"Selected Damped Holt 2025–2030 scenario; {len(top5_output)} distinct groups in the "
                    "saved five-year Top 10."
                ),
                "geography_confidence": "Varies by audited market-to-SA2 mapping; source groups are not automatically suburbs.",
                "recommendation_category": "Growth-scenario monitoring",
                "important_limitation": "Model-based growth is uncertain, not investment return or a guaranteed ranking.",
            },
            {
                "market_or_sa2": f"{len(q3_summary)} filtered SA2 candidates",
                "relevant_question": "Q3",
                "scenario_or_screening_evidence": (
                    "Existing rent/income and provisional liveability rules; minimum listing threshold "
                    f"{minimum_listing_count}; special-purpose areas excluded."
                ),
                "geography_confidence": "SA2 analysis unit; candidate-level screening.",
                "recommendation_category": "Affordability/access comparison",
                "important_limitation": "The score is provisional and omits many household priorities and housing costs.",
            },
            {
                "market_or_sa2": "No confirmed Q2–Q3 balanced candidate",
                "relevant_question": "Q2 + Q3",
                "scenario_or_screening_evidence": (
                    "The saved final integration found zero candidates under the unchanged Q3 rule "
                    "and defensible one-to-one geography requirement."
                ),
                "geography_confidence": (
                    "145-group mapping audit: 54 one-to-one, 65 one-to-many, 26 ambiguous, 0 unmatched."
                ),
                "recommendation_category": "Keep objectives separate",
                "important_limitation": "This does not prove that no suitable area exists; some mappings are not one-to-one.",
            },
        ]
    )

    q1_answer = q1_answer
    takeaways = {
        "project_scope": {
            "overall_project_horizon_years": 5,
            "sprint6_required_horizon_years": 3,
            "primary_model_scenario": "Damped Holt",
            "horizon_alignment_note": (
                "The overall project objective is a five-year rental-growth scenario. Sprint 6 additionally "
                "requires explicit three-year predictions; the three-year output is not merely supplementary."
            ),
        },
        "q1_answer": q1_answer,
        "q2_answer": q2_answer,
        "q3_answer": q3_answer,
        "q2_horizon_comparison": horizon_summary.iloc[0].to_dict(),
        "q2_five_year_top10": top5_output[
            ["rank", "rental_market", "representative_sa2", "current_rent_aud_week", "forecast_2030_aud_week", "predicted_growth_5y_pct"]
        ].to_dict("records"),
        "q2_three_year_top10": top3_output[
            ["rank", "rental_market", "representative_sa2", "current_rent_aud_week", "forecast_2028_aud_week", "predicted_growth_3y_pct"]
        ].to_dict("records"),
        "q3_candidate_count": int(len(q3_summary)),
        "q3_initial_threshold_candidate_count": int(q3_audit["candidate_count"]),
        "balanced_candidate_count": int(len(balanced_candidates)),
        "balanced_candidate_qualification": (
            "No market was confirmed under the chosen criteria and one-to-one geography condition; "
            "this is not evidence that no such area exists anywhere in Victoria."
        ),
        "no_model_retraining": True,
        "no_api_calls": True,
        "no_ors_calls": True,
        "no_downloads": True,
        "previous_outputs_overwritten": False,
    }

    output = root / "data" / "processed" / "sprint6"
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    _save_workflow_figure(figures / "sprint6_project_workflow.png")
    _save_q1_figure(figures / "sprint6_q1_feature_evidence.png", q1_summary, q1_features)
    _save_five_year_figure(figures / "sprint6_5year_top10.png", top5_output)
    _save_horizon_figure(figures / "sprint6_3y_vs_5y_growth.png", horizon_comparison)
    _save_q3_figure(figures / "sprint6_q3_affordability_liveability.png", q3_all, q3_summary, q3_audit)

    five_year_full.to_csv(output / "sprint6_5year_forecasts.csv", index=False)
    top5_output.to_csv(output / "sprint6_5year_top10.csv", index=False)
    three_year_full.to_csv(output / "sprint6_3year_forecasts.csv", index=False)
    top3_output.to_csv(output / "sprint6_3year_top10.csv", index=False)
    horizon_comparison.to_csv(output / "sprint6_horizon_comparison.csv", index=False)
    eligibility_audit.to_csv(output / "sprint6_q2_group_eligibility_audit.csv", index=False)
    q1_summary.to_csv(output / "sprint6_q1_summary.csv", index=False)
    q1_importance_out.to_csv(output / "sprint6_q1_feature_importance.csv", index=False)
    q3_summary.to_csv(output / "sprint6_q3_summary.csv", index=False)
    recommendations.to_csv(output / "sprint6_final_recommendations.csv", index=False)
    (output / "sprint6_final_takeaways.json").write_text(
        json.dumps(takeaways, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )

    short_dh = short_medium.loc[short_medium["model"].eq("damped_holt")]
    short_theta = short_medium.loc[short_medium["model"].eq("theta")]
    if set(short_dh["horizon_quarters"]) != {1, 2, 4} or set(short_theta["horizon_quarters"]) != {1, 2, 4}:
        raise ValueError("The saved short/medium validation does not contain the expected 1Q, 2Q and 4Q horizons.")
    expected_stable = int(model_decision.get("ranking", {}).get("stable_candidate_count", 0))
    audit = {
        "project_root_relative": ".",
        "input_sha256": input_hashes,
        "counts": {
            "sprint3_integrated_listings": int(route_audit["listing_rows"]),
            "sprint3_valid_route_origins": int(route_audit["valid_feature_origins"]),
            "sprint3_cbd_routes_ok": int(route_status["cbd"]["ok"]),
            "sprint3_train_routes_ok": int(route_status["train"]["ok"]),
            "sprint3_invalid_origins": int(route_status["cbd"].get("invalid_origin", 0)),
            "sprint3_statewide_amenity_records_cached_notebook_qa": cached_amenity_total,
            "q1_holdout_sa2_rows": q1_test_rows,
            "q2_distinct_history_groups_total": len(groups),
            "q2_distinct_history_groups_eligible": len(eligible),
            "q2_mapped_sa2_forecast_rows": int(len(raw_forecasts)),
            "q2_five_year_top10": int(len(top5_output)),
            "q2_three_year_top10": int(len(top3_output)),
            "q3_complete_sa2_rows": int(len(q3_all)),
            "q3_initial_median_rule_candidates": int(q3_audit["candidate_count"]),
            "q3_filtered_candidates": int(len(q3_summary)),
            "q2_q3_geography_mapping_classes": mapping_counts,
            "balanced_candidates": int(len(balanced_candidates)),
        },
        "forecast_horizons": {
            "latest_observed_period": str(period_row["latest_observed_period"]),
            "three_year_forecast_period": str(period_row["forecast_3y_period"]),
            "three_year_quarters": horizon_3y,
            "five_year_forecast_period": str(period_row["forecast_5y_period"]),
            "five_year_quarters": horizon_5y,
            "five_year_selected_model": q2_model,
            "five_year_top10_matches_saved_sprint5_scenario": True,
            "all_rankings_use_unique_history_fingerprints": True,
            "top10_overlap_3y_5y": overlap_count,
            "top10_jaccard_3y_5y": jaccard,
            "spearman_growth_rank_correlation_3y_5y": spearman,
            "stable_within_existing_path_sensitivity_count": expected_stable,
        },
        "historical_backtest": {
            "archive_start": history_start,
            "archive_end": history_end,
            "quarterly_periods": int(long_audit["source"]["quarterly_periods"]),
            "forecast_origins": origins,
            "markets_per_origin": historical_markets,
            "forecast_paths_per_model": per_model_paths,
            "models_compared": model_count,
            "forecast_paths_across_all_models": total_paths_all_models,
            "forecast_points_per_model": int(per_model_paths * 20),
            "forecast_points_across_all_models": int(per_model_paths * 20 * model_count),
            "count_note": (
                "The audit's 19,285 total is across seven models; the model-comparison table reports "
                "2,755 paths per model."
            ),
        },
        "long_history_reconciliation": {
            "sa2_rows": int(integration_audit["output_rows"]),
            "sa2_count": int(integration_audit["sa2_count"]),
            "periods": int(integration_audit["output_periods"]),
            "equal_overlap_values": int(integration_audit["overlap_equal_rent_values"]),
            "different_overlap_values": int(integration_audit["overlap_different_rent_values"]),
            "2023_max_abs_difference_aud": float(integration_audit["overlap_by_year"]["2023"]["maximum_absolute_difference_aud"]),
            "2024_2025_match_exactly": (
                int(integration_audit["overlap_by_year"]["2024"]["different_rent_values"]) == 0
                and int(integration_audit["overlap_by_year"]["2025"]["different_rent_values"]) == 0
            ),
        },
        "validation": {
            "five_year_values_equal_saved_group_audit": True,
            "three_year_values_equal_saved_group_audit": True,
            "growth_percentages_recomputed_from_saved_rents": True,
            "five_year_growth_agrees_with_saved_growth_column": True,
            "forecast_endpoints_are_12_and_20_quarters": True,
            "forecast_values_changed": False,
            "q1_model_metrics_recomputed": False,
            "q1_model_metrics_read_from_saved_output": True,
            "q3_formula_changed": False,
            "q3_screening_rules_changed": False,
            "no_new_feature_model_created": True,
            "no_model_retraining": True,
            "no_api_calls": True,
            "no_ors_calls": True,
            "no_downloads": True,
            "previous_sprint_outputs_overwritten": False,
            "balanced_candidate_interpretation_qualified": True,
        },
        "q1_answer": q1_answer,
        "q2_answer": q2_answer,
        "q3_answer": q3_answer,
        "output_files": OUTPUT_FILES,
        "figures": [
            "figures/sprint6_project_workflow.png",
            "figures/sprint6_q1_feature_evidence.png",
            "figures/sprint6_5year_top10.png",
            "figures/sprint6_3y_vs_5y_growth.png",
            "figures/sprint6_q3_affordability_liveability.png",
        ],
        "scope_alignment_note": (
            "The overall project objective is five-year rental-growth scenarios, while Sprint 6 additionally "
            "requires an explicit three-year prediction output. The notebook preserves both forecast tables "
            "and both Top-10 rankings."
        ),
    }
    (output / "sprint6_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )

    return {
        "output_dir": output,
        "figure_dir": figures,
        "figure_paths": {
            "workflow": figures / "sprint6_project_workflow.png",
            "q1": figures / "sprint6_q1_feature_evidence.png",
            "five_year_top10": figures / "sprint6_5year_top10.png",
            "horizon_comparison": figures / "sprint6_3y_vs_5y_growth.png",
            "q3": figures / "sprint6_q3_affordability_liveability.png",
        },
        "data_overview": pd.DataFrame(
            [
                ["Listings", "Domain + Kaggle Victoria snapshots", "Listing-level asking rent, attributes and coordinates; source provenance retained."],
                ["Recent rent history", "Homes Victoria SA2 panel, 2023 Q1–2025 Q3", "277 SA2s; supports current/short-medium context, not a long independent history per SA2."],
                ["Long rent history", f"Homes Victoria moving annual median, {history_start}–{history_end}", "All-properties rent target; distinct rental-history groups are the forecast unit."],
                ["Geography", "ABS ASGS 2021 SA2", "Common area key for spatial analysis; SA2 is not automatically a suburb."],
                ["Area context", "2021 Census/SEIFA and VIF projections", "Household income, IRSAD and projected population; reference data, not all model predictors."],
                ["Transport and amenities", "Saved routes, public transport, schools and statewide Vicmap", f"13,831 valid route origins and {cached_amenity_total:,} statewide amenity records from saved QA; nothing is recollected here."],
            ],
            columns=["Data layer", "Existing source", "Role and boundary"],
        ),
        "feature_catalogue": read_csv("feature_catalogue"),
        "feature_crosswalk": feature_crosswalk,
        "q1_summary": q1_summary,
        "q1_feature_importance": q1_importance_out,
        "q1_answer": q1_answer,
        "five_year_forecasts": five_year_full,
        "five_year_top10": top5_output,
        "three_year_forecasts": three_year_full,
        "three_year_top10": top3_output,
        "horizon_comparison": horizon_comparison,
        "horizon_summary": horizon_summary,
        "short_medium_comparison": short_medium,
        "five_year_backtest": long_backtest,
        "direct_growth_comparison": direct_growth,
        "direct_growth_audit": direct_growth_audit,
        "five_year_backtest_audit": long_audit,
        "history_integration_audit": integration_audit,
        "q3_candidates": q3_summary,
        "q3_top10": q3_top10,
        "q3_audit": q3_audit,
        "q3_answer": q3_answer,
        "geography_summary": pd.DataFrame(
            [{"mapping_class": key, "groups": value} for key, value in mapping_counts.items()]
        ),
        "balanced_candidate_count": int(len(balanced_candidates)),
        "final_recommendations": recommendations,
        "takeaways": takeaways,
        "audit": audit,
        "output_files": [output / filename for filename in OUTPUT_FILES],
    }

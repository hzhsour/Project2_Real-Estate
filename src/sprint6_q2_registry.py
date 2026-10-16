"""Read-only integrity checks for the frozen Sprint 6 Q2 forecast package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PACKAGE_RELATIVE_PATH = Path("data/processed/sprint6/q2_registered_research")
FORECAST_SCHEMA = (
    "candidate",
    "market_id",
    "origin_period_end",
    "target_end_date",
    "horizon_quarters",
    "origin_rent_aud",
    "predicted_growth_pct",
    "predicted_endpoint_rent_aud",
    "rank_score",
    "macro_anchor_growth_pct",
    "raw_model_residual_pct",
    "centred_model_residual_pct",
    "feature_domain_warning_count",
    "feature_domain_warning_features",
    "rank_score_rounded_10dp",
    "growth_rank",
    "top10_membership",
    "top10_candidate",
)
EXPECTED_MODELS = {
    "3y": (
        "ridge21",
        12,
        "2028-09-30",
        "origin_centred_ridge21_12q",
        "q2-stage7-20261016T153002_a38bff6679e1-ridge21-12q",
    ),
    "5y": (
        "ridge18",
        20,
        "2030-09-30",
        "origin_centred_ridge18_20q",
        "q2-stage7-20261016T153002_a38bff6679e1-ridge18-20q",
    ),
}
ORIGIN_PERIOD_END = "2025-09-30"


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"Registered Q2 package integrity check failed: {message}")


def _package_file(package_dir: Path, relative_name: str) -> Path:
    candidate = (package_dir / relative_name).resolve()
    if not candidate.is_relative_to(package_dir.resolve()):
        raise ValueError(f"Forecast package path escapes its directory: {relative_name}")
    if not candidate.is_file():
        raise FileNotFoundError(f"Registered Q2 package file is missing: {candidate}")
    return candidate


def _check_forecast_frame(frame: pd.DataFrame, spec: dict, model_key: str) -> list[str]:
    candidate, horizon, target_date, candidate_id, issue_id = EXPECTED_MODELS[model_key]
    checks: list[str] = []
    _assert(set(FORECAST_SCHEMA).issubset(frame.columns), f"{model_key} forecast schema")
    _assert(len(frame) == int(spec["row_count"]) == 145, f"{model_key} row count")
    _assert(frame["market_id"].astype(str).nunique() == 145, f"{model_key} unique markets")
    _assert(frame["candidate"].astype(str).nunique() == 1, f"{model_key} single candidate")
    _assert(frame["candidate"].astype(str).iloc[0] == candidate == spec["csv_candidate"], f"{model_key} candidate identity")
    _assert(spec["candidate_id"] == candidate_id, f"{model_key} frozen candidate id")
    _assert(spec["issue_id"] == issue_id, f"{model_key} frozen Stage 7 issue id")
    _assert(int(spec["horizon_quarters"]) == horizon, f"{model_key} frozen horizon")
    _assert(spec["target_end_date"] == target_date, f"{model_key} target date")
    _assert(frame["origin_period_end"].astype(str).eq(ORIGIN_PERIOD_END).all(), f"{model_key} origin date")
    _assert(frame["target_end_date"].astype(str).eq(target_date).all(), f"{model_key} row target dates")
    _assert(pd.to_numeric(frame["horizon_quarters"], errors="coerce").eq(horizon).all(), f"{model_key} row horizons")

    numeric = [
        "origin_rent_aud",
        "predicted_growth_pct",
        "predicted_endpoint_rent_aud",
        "rank_score",
        "macro_anchor_growth_pct",
        "raw_model_residual_pct",
        "centred_model_residual_pct",
        "feature_domain_warning_count",
        "rank_score_rounded_10dp",
        "growth_rank",
        "top10_membership",
    ]
    values = frame[numeric].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    _assert(np.isfinite(values).all(), f"{model_key} finite numeric fields")
    _assert(frame["origin_rent_aud"].gt(0).all(), f"{model_key} positive origin rents")
    _assert(frame["predicted_endpoint_rent_aud"].gt(0).all(), f"{model_key} positive endpoint rents")
    growth = pd.to_numeric(frame["predicted_growth_pct"], errors="coerce").to_numpy(dtype=float)
    endpoint_growth = 100.0 * (
        pd.to_numeric(frame["predicted_endpoint_rent_aud"], errors="coerce").to_numpy(dtype=float)
        / pd.to_numeric(frame["origin_rent_aud"], errors="coerce").to_numpy(dtype=float)
        - 1.0
    )
    _assert(float(np.max(np.abs(endpoint_growth - growth))) < 1e-9, f"{model_key} endpoint growth reconstruction")
    _assert(
        np.allclose(frame["rank_score"].to_numpy(float), growth, atol=1e-12, rtol=0.0),
        f"{model_key} rank score equals growth prediction",
    )
    _assert(
        np.allclose(
            frame["predicted_growth_pct"].to_numpy(float),
            frame["macro_anchor_growth_pct"].to_numpy(float)
            + frame["centred_model_residual_pct"].to_numpy(float),
            atol=1e-9,
            rtol=0.0,
        ),
        f"{model_key} macro plus centred residual reconstruction",
    )
    _assert(
        abs(float(frame["centred_model_residual_pct"].mean())) < 1e-9,
        f"{model_key} centred residual mean",
    )
    _assert(
        np.allclose(
            frame["rank_score_rounded_10dp"].to_numpy(float),
            np.round(frame["rank_score"].to_numpy(float), 10),
            atol=1e-12,
            rtol=0.0,
        ),
        f"{model_key} rounded rank scores",
    )
    expected_rank = pd.Series(np.round(frame["rank_score"].to_numpy(float), 10)).rank(
        method="average", ascending=False
    ).to_numpy(float)
    _assert(np.allclose(frame["growth_rank"].to_numpy(float), expected_rank), f"{model_key} growth ranks")
    membership = pd.to_numeric(frame["top10_membership"], errors="coerce").to_numpy(float)
    _assert(abs(float(membership.sum()) - 10.0) < 1e-9, f"{model_key} Top-10 fractional slots")
    _assert(
        np.array_equal(
            membership > 0,
            frame["top10_candidate"].astype(str).str.strip().str.lower().isin({"true", "1"}).to_numpy(),
        ),
        f"{model_key} Top-10 candidate flags",
    )
    _assert(int(frame["top10_candidate"].astype(str).str.strip().str.lower().isin({"true", "1"}).sum()) >= 10, f"{model_key} Top-10 candidate count")

    top10 = frame.sort_values("growth_rank", kind="stable").head(10)
    expected_top10 = spec["top10"]
    _assert(len(expected_top10) == 10, f"{model_key} provenance Top-10 length")
    _assert(
        [str(row["market_id"]) for row in expected_top10]
        == top10["market_id"].astype(str).tolist(),
        f"{model_key} provenance Top-10 markets",
    )
    _assert(
        np.allclose(
            [float(row["predicted_growth_pct"]) for row in expected_top10],
            top10["predicted_growth_pct"].to_numpy(float),
            atol=1e-9,
            rtol=0.0,
        ),
        f"{model_key} provenance Top-10 values",
    )
    checks.extend(
        [
            f"{model_key}: 145 unique native markets",
            f"{model_key}: candidate, horizon and target match frozen identity",
            f"{model_key}: endpoint rent reconstructs growth",
            f"{model_key}: macro anchor plus centred residual reconstructs prediction",
            f"{model_key}: rounded scores, ranks and fractional Top-10 are valid",
            f"{model_key}: Top-10 markets and values match provenance",
            f"{model_key}: frozen Stage 7 issue id",
        ]
    )
    return checks


def load_registered_q2_forecasts(
    project_root: str | Path,
    *,
    verify_hashes: bool = True,
) -> tuple[dict, dict[str, pd.DataFrame], list[dict[str, object]]]:
    """Load and validate frozen CSVs and provenance without modifying any file.

    Returns ``(provenance, {"3y": frame, "5y": frame}, checks)``. A changed
    digest or structural inconsistency raises instead of silently accepting data.
    """

    package_dir = (Path(project_root).resolve() / PACKAGE_RELATIVE_PATH).resolve()
    if not package_dir.is_dir():
        raise FileNotFoundError(f"Frozen Q2 registered package is missing: {package_dir}")
    provenance_path = _package_file(package_dir, "q2_research_provenance.json")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    _assert("models" in provenance and set(EXPECTED_MODELS).issubset(provenance["models"]), "provenance model entries")
    forecasts: dict[str, pd.DataFrame] = {}
    checks: list[dict[str, object]] = []
    digests: dict[str, str] = {}
    for model_key in ("3y", "5y"):
        spec = provenance["models"][model_key]
        path = _package_file(package_dir, str(spec["file"]))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        digests[model_key] = digest
        if verify_hashes:
            _assert(digest == str(spec["sha256"]).lower(), f"{model_key} frozen SHA-256")
        frame = pd.read_csv(path)
        model_checks = _check_forecast_frame(frame, spec, model_key)
        forecasts[model_key] = frame
        checks.append({"Check": f"{model_key} frozen SHA-256", "Result": digest == str(spec["sha256"]).lower()})
        checks.extend({"Check": check, "Result": True} for check in model_checks)
    three_year_markets = forecasts["3y"]["market_id"].astype(str).tolist()
    five_year_markets = forecasts["5y"]["market_id"].astype(str).tolist()
    _assert(three_year_markets == five_year_markets, "matched 145-market identity and order across horizons")
    checks.append({"Check": "3Y/5Y frozen market population and order match", "Result": True})
    return provenance, forecasts, checks

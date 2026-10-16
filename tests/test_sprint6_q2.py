"""Source-level tests for the Sprint 6 Q2 model and frozen package loader."""

from __future__ import annotations

import json
import hashlib
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.sprint5_direct_growth_model import five_year_growth_target
from src.sprint6_q2_model import (
    BEHAVIOUR_FEATURES,
    FEATURES_18,
    FEATURES_21,
    FORECAST_ORIGIN,
    MODEL_SPECS,
    VOLUME_FEATURES,
    _top_k_recovery,
    _validate_design_over_origins,
    build_market_panels,
    chronological_training_rows,
    fit_origin_centred_ridge,
    reproduce_registered_predictions_in_memory,
    tie_aware_top_k_membership,
    top_k_recovery_metrics,
)
from src.sprint6_q2_registry import load_registered_q2_forecasts


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FROZEN_HASHES = {
    "ridge21_3y_forecast.csv": "7028dfa7f8f5f85ac9d44453739b761ab665004a0b72a2a94078473109a77e78",
    "ridge18_5y_forecast.csv": "6c289262973ea542fca709f2e965adaac1d9605faffc05b625be9f7c364cdf43",
}


def _synthetic_design() -> tuple[pd.DataFrame, pd.DataFrame]:
    markets = [f"Market {i}" for i in range(6)]
    origins = [pd.Timestamp("2018-09-30"), pd.Timestamp("2019-09-30"), pd.Timestamp("2020-09-30"), FORECAST_ORIGIN]
    rows = []
    for cohort_idx, origin in enumerate(origins):
        target_end = origin + pd.offsets.QuarterEnd(20)
        for market_idx, market in enumerate(markets):
            row: dict[str, object] = {
                "market": market,
                "origin": origin,
                "target_end": target_end,
                "horizon": 20,
                "current_rent": 400.0 + market_idx * 10 + cohort_idx,
                "lease_count": 40 + market_idx,
                "actual_growth": np.nan if origin == FORECAST_ORIGIN else 8.0 + 0.7 * market_idx + 0.3 * cohort_idx,
                "complete_common": True,
            }
            for feature_idx, feature in enumerate(FEATURES_21):
                row[feature] = 0.5 * (cohort_idx + 1) * (market_idx + 2) + feature_idx / 10
            rows.append(row)
    design = pd.DataFrame(rows)
    periods = pd.date_range("2015-09-30", FORECAST_ORIGIN, freq=pd.offsets.QuarterEnd())
    rent_rows = []
    for market_idx, market in enumerate(markets):
        rent_rows.append([market, *[400 + market_idx * 13 + 2.5 * i for i in range(len(periods))]])
    rents = pd.DataFrame([row[1:] for row in rent_rows], index=markets, columns=periods, dtype=float)
    return design, rents


class Sprint6Q2ModelTests(unittest.TestCase):
    def test_final_notebook_uses_project_registry_and_retains_saved_outputs(self) -> None:
        notebook_path = PROJECT_ROOT / "notebooks/06_sprint6_final_results.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        q2_loader = "".join(notebook["cells"][7]["source"])
        q2_integration = "".join(notebook["cells"][19]["source"])
        self.assertIn("from src.sprint6_q2_registry import load_registered_q2_forecasts", q2_loader)
        self.assertIn("load_registered_q2_forecasts(ROOT)", q2_loader)
        self.assertIn("q2_registry_check_map", q2_integration)
        self.assertTrue(notebook["cells"][7].get("outputs"))
        self.assertTrue(notebook["cells"][19].get("outputs"))

    def test_frozen_candidate_feature_definitions(self) -> None:
        self.assertEqual(len(FEATURES_21), 21)
        self.assertEqual(len(FEATURES_18), 18)
        self.assertEqual(set(FEATURES_21).difference(FEATURES_18), set(BEHAVIOUR_FEATURES))
        self.assertEqual(len(VOLUME_FEATURES), 5)
        self.assertEqual(MODEL_SPECS["ridge21"]["horizon_quarters"], 12)
        self.assertEqual(MODEL_SPECS["ridge18"]["horizon_quarters"], 20)
        self.assertEqual(MODEL_SPECS["ridge21"]["alpha"], 100.0)
        self.assertEqual(MODEL_SPECS["ridge18"]["alpha"], 100.0)
        self.assertEqual(MODEL_SPECS["ridge21"]["anchor"], "project_theta_approximation")
        self.assertEqual(MODEL_SPECS["ridge18"]["anchor"], "drift")

    def test_growth_targets_are_horizon_specific_cumulative_endpoint_rates(self) -> None:
        rents = np.arange(100.0, 142.0, 2.0)
        self.assertAlmostEqual(five_year_growth_target(rents, 0, horizon=12), 24.0)
        self.assertAlmostEqual(five_year_growth_target(rents, 0, horizon=20), 40.0)
        self.assertTrue(np.isnan(five_year_growth_target(rents[:13], 1, horizon=12)))

    def test_panels_filter_summary_rows_and_reject_duplicate_market_periods(self) -> None:
        frame = pd.DataFrame(
            {
                "source_rental_market": ["A", "A", "Statewide"],
                "period_end": pd.to_datetime(["2025-06-30", "2025-09-30", "2025-09-30"]),
                "moving_annual_median_rent_aud": [400.0, 410.0, 405.0],
                "rental_count": [10, 11, 21],
                "is_summary_row": [False, False, True],
            }
        )
        rents, counts, cleaned = build_market_panels(frame, complete_only=False)
        self.assertEqual(rents.index.tolist(), ["A"])
        self.assertEqual(counts.loc["A"].tolist(), [10.0, 11.0])
        self.assertEqual(len(cleaned), 2)
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            build_market_panels(frame.drop(columns=["rental_count"]), complete_only=False)
        duplicate = pd.concat([frame.iloc[[0]], frame.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "Duplicate native-market/quarter"):
            build_market_panels(duplicate, complete_only=False)

    def test_training_is_chronological_and_labels_must_be_mature(self) -> None:
        design, _ = _synthetic_design()
        train, cutoff = chronological_training_rows(design, FORECAST_ORIGIN, 20)
        self.assertEqual(cutoff, pd.Timestamp("2025-06-30"))
        self.assertTrue((train["origin"] < FORECAST_ORIGIN).all())
        self.assertTrue((train["target_end"] <= cutoff).all())
        self.assertNotIn(pd.Timestamp("2020-09-30"), set(train["origin"]))
        self.assertEqual(train["origin"].nunique(), 2)

    def test_scaler_is_fit_only_on_training_rows_and_current_actual_is_unused(self) -> None:
        design, rents = _synthetic_design()
        fitted, audit = fit_origin_centred_ridge(design, rents, FORECAST_ORIGIN, "ridge18")
        train, _ = chronological_training_rows(design, FORECAST_ORIGIN, 20)
        features = list(MODEL_SPECS["ridge18"]["features"])
        np.testing.assert_allclose(audit["scaler_mean"], train[features].mean().to_numpy(), atol=1e-12)
        self.assertEqual(audit["training_rows_used_for_scaling"], len(train))
        self.assertEqual(audit["actual_validation_mean_used"], False)
        mutated = design.copy()
        mutated.loc[mutated["origin"].eq(FORECAST_ORIGIN), "actual_growth"] = 1e9
        mutated_fit, _ = fit_origin_centred_ridge(mutated, rents, FORECAST_ORIGIN, "ridge18")
        np.testing.assert_allclose(fitted["pred_growth"], mutated_fit["pred_growth"], atol=0.0, rtol=0.0)

    def test_tie_aware_top_k_allocates_boundary_slots_fractionally(self) -> None:
        membership = tie_aware_top_k_membership({"A": 4.0, "B": 3.0, "C": 3.0, "D": 3.0, "E": 1.0}, k=3)
        self.assertEqual(membership["A"], 1.0)
        self.assertEqual(membership["B"], 2 / 3)
        self.assertEqual(membership["C"], 2 / 3)
        self.assertEqual(membership["D"], 2 / 3)
        self.assertAlmostEqual(sum(membership.values()), 3.0)

    def test_registered_forecasts_are_hash_verified_and_structurally_valid(self) -> None:
        package = PROJECT_ROOT / "data/processed/sprint6/q2_registered_research"
        if not package.is_dir():
            self.skipTest("Frozen registered Q2 package is not present in this checkout.")
        provenance, forecasts, checks = load_registered_q2_forecasts(PROJECT_ROOT)
        self.assertEqual(len(checks), 17)
        self.assertTrue(all(check["Result"] for check in checks))
        check_names = {check["Check"] for check in checks}
        self.assertIn("3y: frozen Stage 7 issue id", check_names)
        self.assertIn("5y: frozen Stage 7 issue id", check_names)
        self.assertIn("3y: endpoint rent reconstructs growth", check_names)
        self.assertIn("5y: Top-10 markets and values match provenance", check_names)
        self.assertEqual(len(forecasts["3y"]), 145)
        self.assertEqual(len(forecasts["5y"]), 145)
        for filename, expected_hash in FROZEN_HASHES.items():
            spec = next(item for item in provenance["models"].values() if item["file"] == filename)
            self.assertEqual(spec["sha256"], expected_hash)

    def test_current_registered_predictions_reproduce_in_memory_without_writing(self) -> None:
        history = PROJECT_ROOT / "data/processed/sprint5/long_history_validation/homes_victoria_long_history_all_properties.csv"
        package = PROJECT_ROOT / "data/processed/sprint6/q2_registered_research"
        if not history.is_file() or not package.is_dir():
            self.skipTest("Authorised processed history or frozen forecast package is not present.")
        provenance, frozen, _ = load_registered_q2_forecasts(PROJECT_ROOT)
        package_files = [
            package / "q2_research_provenance.json",
            *(package / spec["file"] for spec in provenance["models"].values()),
        ]
        before_hashes = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in package_files
        }
        reproduced, audits = reproduce_registered_predictions_in_memory(PROJECT_ROOT)
        after_hashes = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in package_files
        }
        self.assertEqual(after_hashes, before_hashes)
        for model_key, candidate in (("3y", "ridge21"), ("5y", "ridge18")):
            aligned = reproduced[candidate].set_index("market_id").loc[
                frozen[model_key]["market_id"].astype(str)
            ]
            numeric_columns = [
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
            for column in numeric_columns:
                np.testing.assert_allclose(
                    aligned[column].to_numpy(float),
                    frozen[model_key][column].to_numpy(float),
                    atol=1e-9,
                    rtol=0.0,
                    err_msg=f"{candidate} differs from frozen {column}",
                )
            self.assertEqual(
                aligned["feature_domain_warning_features"].astype(str).tolist(),
                frozen[model_key]["feature_domain_warning_features"].astype(str).tolist(),
            )
            self.assertEqual(
                reproduced[candidate].sort_values("growth_rank").head(10)["market_id"].tolist(),
                frozen[model_key].sort_values("growth_rank").head(10)["market_id"].tolist(),
            )
            self.assertEqual(audits[candidate]["future_realised_targets_used"], False)


class TopKRecoveryTests(unittest.TestCase):
    @staticmethod
    def _scores(order: list[str]) -> dict[str, float]:
        return {market: float(len(order) - rank) for rank, market in enumerate(order)}

    def assertRecovery(self, actual, predicted, expected_count, expected_fraction, k=10):
        self.assertAlmostEqual(_top_k_recovery(actual, predicted, k), expected_count)
        metrics = top_k_recovery_metrics(actual, predicted, k)
        self.assertAlmostEqual(metrics["recovery_count"], expected_count)
        self.assertAlmostEqual(metrics["recovery_fraction"], expected_fraction)

    def test_identical_non_tied_top10_has_count_ten_and_fraction_one(self):
        scores = self._scores([str(i) for i in range(20)])
        self.assertRecovery(scores, scores, 10.0, 1.0)

    def test_disjoint_top10_has_zero_count_and_fraction(self):
        markets = [str(i) for i in range(20)]
        self.assertRecovery(self._scores(markets), self._scores(markets[10:] + markets[:10]), 0.0, 0.0)

    def test_three_common_markets_has_count_three_and_fraction_point_three(self):
        markets = [str(i) for i in range(30)]
        predicted_top = markets[:3] + markets[10:17]
        predicted_order = predicted_top + [market for market in markets if market not in predicted_top]
        self.assertRecovery(self._scores(markets), self._scores(predicted_order), 3.0, 0.3)

    def test_boundary_ties_have_manually_derived_expected_recovered_count(self):
        common = {f"A{i}": float(20 - i) for i in range(8)}
        actual = {**common, "B": 5.0, "C": 5.0, "D": 5.0, "E": 5.0, "F": 0.0, "G": 0.0}
        predicted = {**common, "B": 5.0, "C": 5.0, "D": 0.0, "E": 0.0, "F": 5.0, "G": 5.0}
        # Eight certain overlaps plus B and C, each with 1/2 * 1/2 membership.
        self.assertRecovery(actual, predicted, 8.0 + 2 * (0.5 * 0.5), 0.85)

    def test_population_smaller_than_k_uses_available_slots(self):
        actual = self._scores(["A", "B", "C", "D"])
        predicted = self._scores(["D", "C", "B", "A"])
        self.assertRecovery(actual, predicted, 4.0, 1.0, k=10)

    def test_historical_runner_exports_explicit_count_and_fraction_columns(self):
        markets = [str(i) for i in range(30)]
        predicted_top = markets[:3] + markets[10:17]
        predicted_order = predicted_top + [market for market in markets if market not in predicted_top]
        actual, predicted = self._scores(markets), self._scores(predicted_order)
        forecast = pd.DataFrame({
            "market": markets,
            "actual_growth": [actual[market] for market in markets],
            "pred_growth": [predicted[market] for market in markets],
        })
        rents = pd.DataFrame(1.0, index=markets, columns=[FORECAST_ORIGIN])
        with patch("src.sprint6_q2_model.fit_origin_centred_ridge", return_value=(forecast, {})):
            _, metrics = _validate_design_over_origins(pd.DataFrame(), rents, [FORECAST_ORIGIN])
        self.assertNotIn("tie_aware_top10_recovery", metrics.columns)
        np.testing.assert_allclose(metrics["tie_aware_top10_recovery_count"], 3.0)
        np.testing.assert_allclose(metrics["tie_aware_top10_recovery_fraction"], 0.3)


if __name__ == "__main__":
    unittest.main()

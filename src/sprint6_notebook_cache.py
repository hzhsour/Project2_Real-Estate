"""Read/write the small Sprint 6 presentation cache used by Notebook 06.

The cache lets a clean checkout render the final notebook without carrying the
large/restricted upstream processed datasets. Rebuilding the cache still
requires the documented local Sprint 1--5 prerequisites.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .sprint6_summary import OUTPUT_FILES


CACHE_FILES = (
    "data_overview.csv",
    "feature_crosswalk.csv",
    "short_medium_comparison.csv",
    "five_year_backtest.csv",
    "direct_growth_comparison.csv",
    "geography_summary.csv",
    "horizon_summary.csv",
    "notebook_answers.json",
)


def cache_is_complete(project_root: str | Path) -> bool:
    root = Path(project_root)
    output = root / "data" / "processed" / "sprint6"
    support = output / "presentation_cache"
    required_diagnostics = (
        "validation_12q/model_metrics_12q.csv",
        "validation_12q/growth_ranking_by_origin_12q.csv",
        "validation_12q/horizon_comparison_12q_20q.csv",
        "validation_12q/validation_12q_audit.json",
        "validation_12q/model_errors_12q.png",
        "direct_growth_leakage_audit/direct_growth_fold_maturity.csv",
        "direct_growth_leakage_audit/direct_growth_leakage_audit.json",
        "q1_cross_validation/q1_cv_summary.csv",
        "q1_cross_validation/q1_cv_audit.json",
        "q3_weight_sensitivity/q3_weight_sensitivity_summary.csv",
        "q3_weight_sensitivity/q3_weight_sensitivity_audit.json",
    )
    return all((output / filename).is_file() for filename in OUTPUT_FILES) and all(
        (support / filename).is_file() for filename in CACHE_FILES
    ) and all(
        (output / filename).is_file() for filename in required_diagnostics
    ) and all(
        (output / "figures" / filename).is_file()
        for filename in (
            "sprint6_project_workflow.png",
            "sprint6_q1_feature_evidence.png",
            "sprint6_5year_top10.png",
            "sprint6_3y_vs_5y_growth.png",
            "sprint6_q3_affordability_liveability.png",
        )
    )


def write_notebook_cache(project_root: str | Path, result: dict[str, Any]) -> Path:
    root = Path(project_root).resolve()
    support = root / "data" / "processed" / "sprint6" / "presentation_cache"
    support.mkdir(parents=True, exist_ok=True)
    for name in CACHE_FILES[:-1]:
        key = name.removesuffix(".csv")
        result[key].to_csv(support / name, index=False)
    answers = {
        "q1_answer": result["q1_answer"],
        "q2_answer": (
            "The overall project retains the saved Damped Holt 2025 Q3–2030 Q3 growth scenario. "
            "Sprint 6 also requires an explicit Damped Holt 2025 Q3–2028 Q3 three-year forecast output. "
            "Historical validation is reported by horizon and objective; the five-year scenario is not "
            "claimed to be best on every rent-level metric or a guaranteed ranking."
        ),
        "q3_answer": result["q3_answer"],
        "balanced_candidate_count": int(result["balanced_candidate_count"]),
        "horizon_summary": result["horizon_summary"].to_dict("records"),
    }
    (support / "notebook_answers.json").write_text(
        json.dumps(answers, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    return support


def load_cached_notebook_result(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    if not cache_is_complete(root):
        raise FileNotFoundError(
            "The committed Sprint 6 presentation cache is incomplete. "
            "Restore the selected data/processed/sprint6 files or rebuild them from the documented local prerequisites."
        )
    output = root / "data" / "processed" / "sprint6"
    support = output / "presentation_cache"
    figures = output / "figures"
    read = lambda name, **kwargs: pd.read_csv(output / name, low_memory=False, **kwargs)
    support_read = lambda name: pd.read_csv(support / name, low_memory=False)
    answers = json.loads((support / "notebook_answers.json").read_text(encoding="utf-8"))
    takeaways = json.loads((output / "sprint6_final_takeaways.json").read_text(encoding="utf-8"))
    audit = json.loads((output / "sprint6_audit.json").read_text(encoding="utf-8"))
    q3 = read("sprint6_q3_summary.csv", dtype={"sa2_code": "string"})
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
        "data_overview": support_read("data_overview.csv"),
        "feature_crosswalk": support_read("feature_crosswalk.csv"),
        "q1_summary": read("sprint6_q1_summary.csv"),
        "q1_feature_importance": read("sprint6_q1_feature_importance.csv"),
        "q1_answer": answers["q1_answer"],
        "q2_answer": answers["q2_answer"],
        "five_year_forecasts": read("sprint6_5year_forecasts.csv"),
        "five_year_top10": read("sprint6_5year_top10.csv"),
        "three_year_forecasts": read("sprint6_3year_forecasts.csv"),
        "three_year_top10": read("sprint6_3year_top10.csv"),
        "horizon_comparison": read("sprint6_horizon_comparison.csv"),
        "horizon_summary": pd.DataFrame(answers["horizon_summary"]),
        "short_medium_comparison": support_read("short_medium_comparison.csv"),
        "five_year_backtest": support_read("five_year_backtest.csv"),
        "direct_growth_comparison": support_read("direct_growth_comparison.csv"),
        "q3_candidates": q3,
        "q3_top10": q3.sort_values("candidate_rank").head(10),
        "q3_answer": answers["q3_answer"],
        "geography_summary": support_read("geography_summary.csv"),
        "balanced_candidate_count": int(answers["balanced_candidate_count"]),
        "final_recommendations": read("sprint6_final_recommendations.csv"),
        "takeaways": takeaways,
        "audit": audit,
        "output_files": [output / filename for filename in OUTPUT_FILES],
    }

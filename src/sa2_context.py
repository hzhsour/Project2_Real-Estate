"""Load and join Victoria SA2 population and socioeconomic context.

The project uses property listings as the observations. These sources add
area-level context to the existing listings; they do not create additional
properties.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def normalise_sa2_code(values: pd.Series) -> pd.Series:
    """Return SA2 codes as stable nine-digit strings."""

    codes = (
        values.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )
    numeric = codes.str.fullmatch(r"\d+").fillna(False)
    return codes.where(~numeric, codes.str.zfill(9))


def _normalise_column_label(value: object) -> str:
    """Collapse whitespace in imported spreadsheet column labels."""

    if isinstance(value, (int, float)) and not pd.isna(value):
        return str(int(value))
    return " ".join(str(value).strip().split())


def _require_unique(frame: pd.DataFrame, source_name: str) -> None:
    """Fail clearly when a source is not one row per SA2."""

    duplicates = frame.loc[frame["sa2_code"].duplicated(), "sa2_code"].dropna()
    if not duplicates.empty:
        examples = ", ".join(duplicates.astype(str).head(5))
        raise ValueError(f"{source_name} has duplicate SA2 codes: {examples}")


def load_vif_sa2_population(path: str | Path) -> pd.DataFrame:
    """Load VIF2023 population forecasts for Victorian SA2s."""

    raw = pd.read_excel(path, sheet_name="Total_Population", header=9)
    raw.columns = [_normalise_column_label(column) for column in raw.columns]
    required = {"SA2 code", "Region Type", "Region", "2026", "2031"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"VIF population sheet is missing columns: {sorted(missing)}")

    raw = raw.loc[
        raw["Region Type"].astype("string").str.casefold().eq("sa2")
    ].copy()
    result = raw[["SA2 code", "Region", "2026", "2031"]].rename(
        columns={
            "SA2 code": "sa2_code",
            "Region": "sa2_name_vif",
            "2026": "population_2026",
            "2031": "population_2031",
        }
    )
    result["sa2_code"] = normalise_sa2_code(result["sa2_code"])
    result["population_2026"] = pd.to_numeric(
        result["population_2026"], errors="coerce"
    )
    result["population_2031"] = pd.to_numeric(
        result["population_2031"], errors="coerce"
    )
    result["population_growth_pct"] = (
        result["population_2031"] / result["population_2026"] - 1
    ) * 100
    result = result.dropna(subset=["sa2_code"]).reset_index(drop=True)
    _require_unique(result, "VIF2023 population")
    return result.sort_values("sa2_code").reset_index(drop=True)


def load_census_sa2_income(path: str | Path) -> pd.DataFrame:
    """Load 2021 Census G02 household income for Victorian SA2s."""

    raw = pd.read_csv(path)
    required = {"SA2_CODE_2021", "Median_tot_hhd_inc_weekly"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Census G02 file is missing columns: {sorted(missing)}")

    result = raw[["SA2_CODE_2021", "Median_tot_hhd_inc_weekly"]].rename(
        columns={
            "SA2_CODE_2021": "sa2_code",
            "Median_tot_hhd_inc_weekly": "median_household_income",
        }
    )
    result["sa2_code"] = normalise_sa2_code(result["sa2_code"])
    result["median_household_income"] = pd.to_numeric(
        result["median_household_income"], errors="coerce"
    )
    result = result.dropna(subset=["sa2_code"]).reset_index(drop=True)
    _require_unique(result, "2021 Census G02")
    return result.sort_values("sa2_code").reset_index(drop=True)


def load_abs_sa2_rent_context(path: str | Path) -> pd.DataFrame:
    """Load 2021 Census SA2 rent reference and rental-stress measures.

    The source is the ABS Data by Region Family and Community table.  RENT_4
    is represented as ``census_median_rent_weekly`` and STRESS_17 as
    ``rental_stress_pct`` so these 2021 Census measures remain separate from
    the Homes Victoria historical rent series.
    """

    raw = pd.read_csv(path)
    required = {
        "sa2_code",
        "census_median_rent_weekly",
        "rental_stress_pct",
    }
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"ABS SA2 rent context file is missing columns: {sorted(missing)}")

    result = raw[
        ["sa2_code", "census_median_rent_weekly", "rental_stress_pct"]
    ].copy()
    result["sa2_code"] = normalise_sa2_code(result["sa2_code"])
    result["census_median_rent_weekly"] = pd.to_numeric(
        result["census_median_rent_weekly"], errors="coerce"
    )
    result["rental_stress_pct"] = pd.to_numeric(
        result["rental_stress_pct"], errors="coerce"
    )
    result = result.dropna(subset=["sa2_code"]).reset_index(drop=True)
    _require_unique(result, "ABS 2021 SA2 rent context")
    return result.sort_values("sa2_code").reset_index(drop=True)


def load_seifa_sa2_irsad(path: str | Path) -> pd.DataFrame:
    """Load 2021 IRSAD score and decile for Victorian SA2s."""

    raw = pd.read_excel(path, sheet_name="Table 3", header=5)
    raw.columns = [_normalise_column_label(column) for column in raw.columns]
    code_column = next(
        (
            column
            for column in raw.columns
            if "statistical area level 2" in column.casefold()
            and "9-digit code" in column.casefold()
        ),
        None,
    )
    score_column = next(
        (column for column in raw.columns if column.casefold() == "score"), None
    )
    decile_column = next(
        (column for column in raw.columns if column.casefold() == "decile"), None
    )
    if code_column is None or score_column is None or decile_column is None:
        raise ValueError("SEIFA Table 3 does not contain the expected SA2, Score and Decile columns")

    result = raw[[code_column, score_column, decile_column]].rename(
        columns={
            code_column: "sa2_code",
            score_column: "irsad_score",
            decile_column: "irsad_decile",
        }
    )
    result["sa2_code"] = normalise_sa2_code(result["sa2_code"])
    result = result.loc[result["sa2_code"].str.startswith("2", na=False)].copy()
    result["irsad_score"] = pd.to_numeric(result["irsad_score"], errors="coerce")
    result["irsad_decile"] = pd.to_numeric(result["irsad_decile"], errors="coerce")
    result = result.dropna(subset=["sa2_code"]).reset_index(drop=True)
    _require_unique(result, "SEIFA Table 3 IRSAD")
    return result.sort_values("sa2_code").reset_index(drop=True)


def build_sa2_context(project_root: str | Path) -> pd.DataFrame:
    """Build one row per Victorian SA2 from the external context sources."""

    project_root = Path(project_root)
    vif = load_vif_sa2_population(
        project_root
        / "data/external/vif2023/VIF2023_SA2_Pop_Hhold_Dwelling_Projections_to_2036_Release_2.xlsx"
    )
    census = load_census_sa2_income(
        project_root
        / "data/external/abs_census/sa2/2021Census_G02_VIC_SA2.csv"
    )
    census_rent = load_abs_sa2_rent_context(
        project_root
        / "data/external/abs_data_by_region/family_community_sa2_2021.csv"
    )
    seifa = load_seifa_sa2_irsad(
        project_root
        / "data/external/seifa/Statistical Area Level 2, Indexes, SEIFA 2021.xlsx"
    )

    context = vif.merge(census, on="sa2_code", how="left", validate="one_to_one")
    context = context.merge(
        census_rent, on="sa2_code", how="left", validate="one_to_one"
    )
    context = context.merge(seifa, on="sa2_code", how="left", validate="one_to_one")
    context["census_income_available"] = context["median_household_income"].notna()
    context["census_rent_available"] = context["census_median_rent_weekly"].notna()
    context["rental_stress_available"] = context["rental_stress_pct"].notna()
    context["irsad_available"] = context["irsad_score"].notna()
    return context.sort_values("sa2_code").reset_index(drop=True)


def enrich_listings_with_sa2_context(
    listings: pd.DataFrame, context: pd.DataFrame
) -> pd.DataFrame:
    """Attach one SA2 context row to each listing already assigned to an SA2."""

    if "sa2_code" not in listings.columns:
        raise ValueError("Listings must contain sa2_code before context enrichment")
    if "sa2_code" not in context.columns:
        raise ValueError("SA2 context must contain sa2_code")

    result = listings.copy()
    result["sa2_code"] = normalise_sa2_code(result["sa2_code"])
    context = context.copy()
    context["sa2_code"] = normalise_sa2_code(context["sa2_code"])
    _require_unique(context, "SA2 context")
    return result.merge(context, on="sa2_code", how="left", validate="many_to_one")


def build_sa2_listing_summary(enriched_listings: pd.DataFrame) -> pd.DataFrame:
    """Create an SA2-level summary for choropleths and rankings."""

    required = {"sa2_code", "sa2_name", "listing_id", "weekly_rent_aud"}
    missing = required.difference(enriched_listings.columns)
    if missing:
        raise ValueError(f"Enriched listings are missing columns: {sorted(missing)}")

    source = enriched_listings.copy()
    source["sa2_code"] = normalise_sa2_code(source["sa2_code"])
    source["weekly_rent_aud"] = pd.to_numeric(
        source["weekly_rent_aud"], errors="coerce"
    )
    summary = (
        source.dropna(subset=["sa2_code"])
        .groupby(["sa2_code", "sa2_name"], as_index=False)
        .agg(
            listing_count=("listing_id", "nunique"),
            rent_observation_count=("weekly_rent_aud", "count"),
            median_weekly_rent_aud=("weekly_rent_aud", "median"),
            mean_weekly_rent_aud=("weekly_rent_aud", "mean"),
        )
    )
    context_columns = [
        "sa2_code",
        "population_2026",
        "population_2031",
        "population_growth_pct",
        "median_household_income",
        "census_median_rent_weekly",
        "rental_stress_pct",
        "irsad_score",
        "irsad_decile",
        "census_income_available",
        "census_rent_available",
        "rental_stress_available",
        "irsad_available",
    ]
    context_columns = [
        column for column in context_columns if column in source.columns
    ]
    context = source[context_columns].drop_duplicates("sa2_code")
    summary = summary.merge(context, on="sa2_code", how="left", validate="one_to_one")
    return summary.sort_values("sa2_code").reset_index(drop=True)

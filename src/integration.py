"""Load the Victoria rental data layers used by Project 2."""

from pathlib import Path

import pandas as pd

from .homes_victoria import load_moving_annual_rents, load_quarterly_benchmarks
from .kaggle_data import load_kaggle_rental_market
from .sa2 import add_sa2_to_listings, load_abs_sa2_boundaries


def load_victoria_data_layers(project_root: str | Path) -> dict[str, pd.DataFrame]:
    """Load listing-level, historical, and affordability data together.

    The tables intentionally remain separate because they have different
    observational units: listings, suburb-quarter medians, and annual
    affordability snapshots. They can be joined later using suburb/type/time
    keys without pretending they are the same kind of observation.
    """

    project_root = Path(project_root)
    listings = load_kaggle_rental_market(
        project_root
        / "data/external/australian-rental-market-data-2026"
        / "australian_rental_market_2026.csv",
        state="VIC",
    )
    history = load_moving_annual_rents(
        project_root
        / "data/external/homes_victoria"
        / "moving_annual_rent_suburb_sep_2025.xlsx",
        start_period="2024-01-01",
        end_period="2025-09-30",
    )
    affordability = pd.read_csv(
        project_root
        / "data/external/anglicare_victoria"
        / "anglicare_ras_victoria_2022_2026_summary.csv"
    )
    benchmarks = load_quarterly_benchmarks(
        project_root
        / "data/external/homes_victoria/quarterly_tables_sep_2025.xlsx"
    )
    sa2_boundaries = load_abs_sa2_boundaries(
        project_root
        / "data/external/abs_asgs/SA2_2021_AUST_GDA2020/SA2_2021_AUST_GDA2020.shp"
    )
    listings = add_sa2_to_listings(listings, sa2_boundaries)

    return {
        "kaggle_listings_vic_2026": listings,
        "homes_victoria_suburb_history": history,
        "anglicare_affordability": affordability,
        "homes_victoria_quarterly_benchmarks": benchmarks,
        "abs_sa2_boundaries": sa2_boundaries,
    }


def save_victoria_data_layers(
    project_root: str | Path,
    layers: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Path]:
    """Save reproducible processed copies of the three data layers."""

    project_root = Path(project_root)
    layers = layers or load_victoria_data_layers(project_root)
    output_dir = project_root / "data/processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "kaggle_listings_vic_2026": output_dir / "kaggle_listings_vic_2026.csv",
        "homes_victoria_suburb_history": output_dir / "homes_victoria_rents_2024_2025.csv",
        "anglicare_affordability": output_dir / "anglicare_ras_victoria_2022_2026_summary.csv",
    }
    for name, path in paths.items():
        layers[name].to_csv(path, index=False)
    return paths

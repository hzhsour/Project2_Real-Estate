"""Load ABS SA2 boundaries and assign listing points to SA2 areas."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd


SA2_COLUMNS = {
    "SA2_CODE21": "sa2_code",
    "SA2_NAME21": "sa2_name",
    "SA3_CODE21": "sa3_code",
    "SA3_NAME21": "sa3_name",
    "SA4_CODE21": "sa4_code",
    "SA4_NAME21": "sa4_name",
    "GCC_CODE21": "gcc_code",
    "GCC_NAME21": "gcc_name",
    "STE_CODE21": "state_code",
    "STE_NAME21": "state_name",
}


def load_abs_sa2_boundaries(
    boundary_path: str | Path,
    *,
    state_code: str = "2",
) -> gpd.GeoDataFrame:
    """Return Victorian ABS ASGS 2021 SA2 polygons in WGS84 coordinates.

    The ABS shapefile is supplied in GDA2020 (EPSG:7844). Only the Victorian
    polygons are retained, and source field names are mapped to the project's
    lower-case canonical names.
    """

    boundary_path = Path(boundary_path)
    if boundary_path.is_dir():
        candidates = sorted(boundary_path.glob("*.shp"))
        if len(candidates) != 1:
            raise ValueError(
                f"Expected one SA2 shapefile in {boundary_path}, found {len(candidates)}"
            )
        boundary_path = candidates[0]

    boundaries = gpd.read_file(boundary_path)
    required = set(SA2_COLUMNS) | {"geometry"}
    missing = required.difference(boundaries.columns)
    if missing:
        raise ValueError(f"ABS SA2 file is missing columns: {sorted(missing)}")

    boundaries = boundaries.loc[
        boundaries["STE_CODE21"].astype("string").eq(str(state_code))
    ].copy()
    boundaries = boundaries.rename(columns=SA2_COLUMNS)
    boundaries = boundaries[list(SA2_COLUMNS.values()) + ["geometry"]]

    if boundaries.crs is None:
        boundaries = boundaries.set_crs("EPSG:7844")
    boundaries = boundaries.to_crs("EPSG:4326")
    invalid = ~boundaries.geometry.is_valid
    if invalid.any() and hasattr(boundaries.geometry, "make_valid"):
        boundaries.loc[invalid, "geometry"] = boundaries.loc[invalid].geometry.make_valid()

    return boundaries.reset_index(drop=True)


def add_sa2_to_listings(
    listings: pd.DataFrame,
    boundaries: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Add SA2/SA3/SA4 fields to listings using a point-in-polygon join.

    The row count and order of ``listings`` are preserved. Listings with
    invalid coordinates remain in the output with missing geography fields.
    """

    required = {"latitude", "longitude"}
    missing = required.difference(listings.columns)
    if missing:
        raise ValueError(f"Missing listing coordinate columns: {sorted(missing)}")
    boundary_columns = {"sa2_code", "sa2_name", "geometry"}
    missing_boundaries = boundary_columns.difference(boundaries.columns)
    if missing_boundaries:
        raise ValueError(
            f"Missing normalised SA2 boundary columns: {sorted(missing_boundaries)}"
        )

    base = listings.copy().reset_index(drop=True)
    base["latitude"] = pd.to_numeric(base["latitude"], errors="coerce")
    base["longitude"] = pd.to_numeric(base["longitude"], errors="coerce")
    base["_listing_row_id"] = range(len(base))

    valid = base.dropna(subset=["latitude", "longitude"]).copy()
    if not valid.empty:
        points = gpd.GeoDataFrame(
            valid[["_listing_row_id", "latitude", "longitude"]],
            geometry=gpd.points_from_xy(valid["longitude"], valid["latitude"]),
            crs="EPSG:4326",
        )
        join_columns = [
            column
            for column in boundaries.columns
            if column in set(SA2_COLUMNS.values()) | {"geometry"}
        ]
        joined = gpd.sjoin(
            points,
            boundaries[join_columns],
            how="left",
            predicate="within",
        )
        joined = joined.sort_values("_listing_row_id").drop_duplicates(
            "_listing_row_id"
        )
        joined = joined.set_index("_listing_row_id")
        for column in join_columns:
            if column == "geometry":
                continue
            base[column] = joined[column].reindex(base["_listing_row_id"]).to_numpy()

    return base.drop(columns="_listing_row_id")

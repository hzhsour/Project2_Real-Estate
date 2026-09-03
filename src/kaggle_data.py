"""Adapter for the public Australian Rental Market Data 2026 dataset."""

from pathlib import Path

import pandas as pd


KAGGLE_DATASET_URL = (
    "https://www.kaggle.com/datasets/kanchana1990/"
    "australian-rental-market-data-2026"
)

REQUIRED_COLUMNS = {
    "title",
    "price_display",
    "propertyType",
    "latitude",
    "longitude",
    "postcode",
    "state",
    "street_address",
    "suburb",
    "bathrooms",
    "bedrooms",
    "parking_spaces",
}


def load_kaggle_rental_market(
    csv_path: str | Path,
    *,
    state: str = "VIC",
) -> pd.DataFrame:
    """Load Kaggle rows into the project's canonical listing schema.

    The dataset is a static listing snapshot. It is suitable for Sprint 1
    mapping and baseline rental modelling, but it is not a time series for
    forecasting by itself.
    """

    raw = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS.difference(raw.columns)
    if missing:
        raise ValueError(f"Kaggle file is missing required columns: {sorted(missing)}")

    target_state = state.upper()
    state_values = raw["state"].astype("string").str.upper()
    selected = raw.loc[state_values.eq(target_state)].copy()

    selected["state"] = state_values.loc[selected.index]
    selected["weekly_rent_aud"] = pd.to_numeric(
        selected["price_display"], errors="coerce"
    )
    for column in ["bedrooms", "bathrooms", "parking_spaces", "latitude", "longitude"]:
        selected[column] = pd.to_numeric(selected[column], errors="coerce")

    selected["listing_id"] = [f"kaggle_2026_{index}" for index in selected.index]
    selected["source"] = "Kaggle Australian Rental Market Data 2026"
    selected["source_page_url"] = KAGGLE_DATASET_URL
    selected["listing_url"] = pd.NA
    selected["collected_at_utc"] = pd.NaT

    listings = selected.rename(
        columns={
            "street_address": "address",
            "propertyType": "property_type",
        }
    )[
        [
            "listing_id",
            "source",
            "source_page_url",
            "listing_url",
            "collected_at_utc",
            "title",
            "address",
            "suburb",
            "state",
            "postcode",
            "property_type",
            "weekly_rent_aud",
            "bedrooms",
            "bathrooms",
            "parking_spaces",
            "latitude",
            "longitude",
        ]
    ].reset_index(drop=True)

    return listings

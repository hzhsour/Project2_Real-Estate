"""Sprint 1 visualisation helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def create_property_map(
    listings: pd.DataFrame, output_path: str | Path, *, zoom_start: int = 9
) -> object:
    """Create and save an interactive Folium map for geocoded listings."""

    import folium

    required = {"latitude", "longitude"}
    missing = required.difference(listings.columns)
    if missing:
        raise ValueError(f"Missing map columns: {sorted(missing)}")

    points = listings.copy()
    points["latitude"] = pd.to_numeric(points["latitude"], errors="coerce")
    points["longitude"] = pd.to_numeric(points["longitude"], errors="coerce")
    points = points.dropna(subset=["latitude", "longitude"])
    if points.empty:
        raise ValueError("No valid latitude/longitude pairs available for mapping")

    centre = [float(points["latitude"].mean()), float(points["longitude"].mean())]
    fmap = folium.Map(location=centre, zoom_start=zoom_start, control_scale=True)
    for _, row in points.iterrows():
        rent = row.get("weekly_rent_aud")
        rent_text = f"${rent:,.0f}/week" if pd.notna(rent) else "rent unavailable"
        title = row.get("title") or row.get("address") or row.get("listing_id")
        popup = f"{title}<br>{rent_text}"
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=5,
            color="#1f77b4",
            fill=True,
            fill_opacity=0.75,
            popup=popup,
        ).add_to(fmap)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(output_path)
    return fmap


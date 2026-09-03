"""Interactive Sprint 1 visualisations for the Victoria rental data."""

from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd


def _safe_text(value: object, default: str = "Not available") -> str:
    """Return HTML-safe text for a scalar dataframe value."""

    if value is None or pd.isna(value):
        return default
    text = str(value).strip()
    return escape(text) if text else default


def _format_currency(value: object, suffix: str = "/week") -> str:
    """Format a numeric AUD value without exposing NaN in the popup."""

    if value is None or pd.isna(value):
        return "Not available"
    try:
        return f"${float(value):,.0f}{suffix}"
    except (TypeError, ValueError):
        return "Not available"


def _normalise_key(value: object) -> str | None:
    """Create a stable, case-insensitive join key for suburb names."""

    if value is None or pd.isna(value):
        return None
    text = str(value).strip().casefold()
    return text or None


def _first_available(row: pd.Series, *columns: str) -> object:
    """Return the first non-empty scalar value from a listing row."""

    for column in columns:
        value = row.get(column)
        if value is not None and not pd.isna(value) and str(value).strip():
            return value
    return None


def _history_lookup(history: pd.DataFrame | None) -> dict[str, dict[str, object]]:
    """Summarise Homes Victoria history for safe suburb-level map lookups.

    Homes Victoria contains more than one row for a small number of suburb and
    period combinations. Aggregating to one row per suburb-period prevents a
    many-to-many join from duplicating listing markers.
    """

    if history is None or history.empty:
        return {}

    required = {
        "suburb",
        "property_type",
        "period_end",
        "median_weekly_rent_aud",
        "new_lease_count",
    }
    if not required.issubset(history.columns):
        return {}

    source = history.copy()
    all_properties = source.loc[
        source["property_type"].astype("string").str.casefold().eq("all properties")
    ].copy()
    if all_properties.empty:
        return {}

    all_properties["period_end"] = pd.to_datetime(
        all_properties["period_end"], errors="coerce"
    )
    all_properties["suburb_key"] = all_properties["suburb"].map(_normalise_key)
    all_properties["median_weekly_rent_aud"] = pd.to_numeric(
        all_properties["median_weekly_rent_aud"], errors="coerce"
    )
    all_properties["new_lease_count"] = pd.to_numeric(
        all_properties["new_lease_count"], errors="coerce"
    )
    all_properties = all_properties.dropna(subset=["suburb_key", "period_end"])
    if all_properties.empty:
        return {}

    grouped = (
        all_properties.groupby(["suburb_key", "period_end"], as_index=False)
        .agg(
            median_weekly_rent_aud=("median_weekly_rent_aud", "median"),
            new_lease_count=("new_lease_count", "sum"),
        )
        .sort_values(["suburb_key", "period_end"])
    )
    first = grouped.groupby("suburb_key", as_index=True).first()
    latest = grouped.groupby("suburb_key", as_index=True).last()
    first_periods = grouped.groupby("suburb_key")["period_end"].min()
    latest_periods = grouped.groupby("suburb_key")["period_end"].max()

    lookup: dict[str, dict[str, object]] = {}
    for suburb_key in latest.index:
        latest_row = latest.loc[suburb_key]
        first_row = first.loc[suburb_key]
        latest_rent = latest_row["median_weekly_rent_aud"]
        first_rent = first_row["median_weekly_rent_aud"]
        growth = pd.NA
        if pd.notna(first_rent) and float(first_rent) != 0 and pd.notna(latest_rent):
            growth = (float(latest_rent) / float(first_rent) - 1) * 100

        lookup[str(suburb_key)] = {
            "latest_period": latest_periods.loc[suburb_key],
            "latest_rent": latest_rent,
            "latest_new_leases": latest_row["new_lease_count"],
            "first_period": first_periods.loc[suburb_key],
            "growth_pct": growth,
        }
    return lookup


def _affordability_note(affordability: pd.DataFrame | None) -> str:
    """Create a compact latest-year affordability note for the map panel."""

    if affordability is None or affordability.empty:
        return "No affordability snapshot loaded"
    required = {
        "year",
        "snapshot_period",
        "median_weekly_rent_aud",
        "minimum_wage_affordable_pct",
        "income_support_affordable_pct",
    }
    if not required.issubset(affordability.columns):
        return "Affordability snapshot unavailable"

    source = affordability.copy()
    source["year"] = pd.to_numeric(source["year"], errors="coerce")
    source = source.dropna(subset=["year"]).sort_values("year")
    if source.empty:
        return "Affordability snapshot unavailable"
    row = source.iloc[-1]
    period = _safe_text(row["snapshot_period"])
    median = _format_currency(row["median_weekly_rent_aud"])
    minimum_wage = (
        f"{float(row['minimum_wage_affordable_pct']):.1f}%"
        if pd.notna(row["minimum_wage_affordable_pct"])
        else "not available"
    )
    income_support = (
        f"{float(row['income_support_affordable_pct']):.2f}%"
        if pd.notna(row["income_support_affordable_pct"])
        else "not available"
    )
    return (
        f"{period}: median {median}; affordable on minimum wage {minimum_wage}; "
        f"income support {income_support}"
    )


def _rent_band(value: object) -> tuple[str, str]:
    """Return a human-readable rent band and its map colour."""

    if value is None or pd.isna(value):
        return "Rent unavailable", "#7f7f7f"
    value = float(value)
    if value < 500:
        return "Under $500", "#2ca02c"
    if value < 700:
        return "$500–699", "#1f77b4"
    if value < 900:
        return "$700–899", "#ff7f0e"
    return "$900+", "#d62728"


def create_property_map(
    listings: pd.DataFrame, output_path: str | Path, *, zoom_start: int = 9
) -> object:
    """Create and save the original simple Folium map for compatibility."""

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
        rent_text = _format_currency(rent)
        title = _first_available(row, "title", "address", "listing_id")
        popup = f"{_safe_text(title)}<br>{rent_text}"
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


def create_enriched_property_map(
    listings: pd.DataFrame,
    output_path: str | Path,
    *,
    history: pd.DataFrame | None = None,
    affordability: pd.DataFrame | None = None,
    zoom_start: int = 9,
) -> object:
    """Create an interactive, clustered Victoria property-location map.

    The markers remain listing-level observations from the Kaggle snapshot.
    Homes Victoria history is only used as suburb-level context in popups, and
    the Anglicare table is shown as a latest-year context note. This keeps the
    three datasets separate instead of presenting different observational
    units as if they were one listing table.
    """

    import folium
    from folium.plugins import FeatureGroupSubGroup, Fullscreen, MarkerCluster, MiniMap

    required = {"latitude", "longitude"}
    missing = required.difference(listings.columns)
    if missing:
        raise ValueError(f"Missing map columns: {sorted(missing)}")

    points = listings.copy()
    points["latitude"] = pd.to_numeric(points["latitude"], errors="coerce")
    points["longitude"] = pd.to_numeric(points["longitude"], errors="coerce")
    points = points.dropna(subset=["latitude", "longitude"]).copy()
    if points.empty:
        raise ValueError("No valid latitude/longitude pairs available for mapping")

    history_by_suburb = _history_lookup(history)
    centre = [float(points["latitude"].mean()), float(points["longitude"].mean())]
    fmap = folium.Map(
        location=centre,
        zoom_start=zoom_start,
        control_scale=True,
        tiles="OpenStreetMap",
    )

    parent = MarkerCluster(
        name="All listings (clustered)",
        overlay=True,
        control=True,
        options={"disableClusteringAtZoom": 13},
    )
    parent.add_to(fmap)

    property_series = points.get(
        "property_type", pd.Series("Unknown", index=points.index)
    ).fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
    points["_property_type_key"] = property_series
    type_groups: dict[str, object] = {}
    for property_type in sorted(points["_property_type_key"].unique()):
        type_groups[property_type] = FeatureGroupSubGroup(
            parent,
            name=f"Property type: {property_type}",
            overlay=True,
            control=True,
            show=True,
        )
        type_groups[property_type].add_to(fmap)

    for _, row in points.iterrows():
        rent = row.get("weekly_rent_aud")
        rent_band, colour = _rent_band(rent)
        property_type = _safe_text(row.get("property_type"), "Unknown")
        title = _first_available(row, "title", "address", "listing_id")
        suburb = _safe_text(row.get("suburb"))
        suburb_key = _normalise_key(row.get("suburb"))
        history_row = history_by_suburb.get(suburb_key or "")

        history_html = "<b>Homes Victoria history:</b> no suburb match"
        if history_row is not None:
            latest_period = pd.Timestamp(history_row["latest_period"]).strftime("%b %Y")
            first_period = pd.Timestamp(history_row["first_period"]).strftime("%b %Y")
            history_html = (
                "<b>Homes Victoria suburb context:</b> "
                f"{_format_currency(history_row['latest_rent'])} median in {latest_period}"
            )
            if pd.notna(history_row["growth_pct"]):
                history_html += (
                    f"; {float(history_row['growth_pct']):+.1f}% since {first_period}"
                )

        popup_html = f"""
        <div style="min-width: 275px; line-height: 1.45;">
          <h4 style="margin: 0 0 6px 0;">{_safe_text(title)}</h4>
          <b>Asking rent:</b> {_format_currency(rent)}<br>
          <b>Rent band:</b> {rent_band}<br>
          <b>Property type:</b> {property_type}<br>
          <b>Bedrooms / bathrooms / parking:</b>
          {_safe_text(row.get('bedrooms'))} /
          {_safe_text(row.get('bathrooms'))} /
          {_safe_text(row.get('parking_spaces'))}<br>
          <b>Suburb:</b> {suburb}<br>
          <b>Postcode:</b> {_safe_text(row.get('postcode'))}<br>
          <hr style="margin: 7px 0;">
          {history_html}<br>
          <small>Listing coordinates and asking rent: Kaggle Victoria snapshot.<br>
          Historical context: Homes Victoria suburb rental report.</small>
        </div>
        """
        marker = folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=5,
            color=colour,
            fill=True,
            fill_color=colour,
            fill_opacity=0.78,
            weight=1,
            popup=folium.Popup(popup_html, max_width=380),
            tooltip=f"{suburb} · {_format_currency(rent)}",
        )
        marker.add_to(type_groups[row["_property_type_key"]])

    legend_html = """
    <div style="position: fixed; z-index: 9999; bottom: 28px; left: 12px;
                background: white; border: 2px solid #777; border-radius: 5px;
                padding: 9px 11px; font-size: 13px; line-height: 1.55;
                box-shadow: 0 1px 4px rgba(0,0,0,.25);">
      <b>Asking rent (AUD/week)</b><br>
      <span style="color:#2ca02c;">●</span> Under $500<br>
      <span style="color:#1f77b4;">●</span> $500–699<br>
      <span style="color:#ff7f0e;">●</span> $700–899<br>
      <span style="color:#d62728;">●</span> $900+<br>
      <span style="color:#7f7f7f;">●</span> Unavailable
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))

    info_html = f"""
    <div style="position: fixed; z-index: 9998; top: 12px; right: 52px;
                max-width: 365px; background: rgba(255,255,255,.95);
                border: 1px solid #999; border-radius: 5px; padding: 9px 12px;
                font-size: 12px; line-height: 1.45; box-shadow: 0 1px 4px rgba(0,0,0,.2);">
      <b>Victoria rental listings</b><br>
      {len(points):,} geocoded listing points · click a point for details.<br>
      Use the layer control to filter by property type.<br>
      <b>Affordability context:</b> {_affordability_note(affordability)}
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(info_html))
    Fullscreen(position="topleft").add_to(fmap)
    MiniMap(toggle_display=True).add_to(fmap)
    folium.LayerControl(collapsed=False, position="topright").add_to(fmap)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(output_path)
    return fmap

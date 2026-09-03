"""Responsible, source-agnostic helpers for rental-listing collection.

The parser supports common Schema.org JSON-LD listing records and a small
HTML-card fallback. Live collection is opt-in and checks robots.txt before a
request. It must only be used with an authorised source or course-provided
export/API.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import pandas as pd


OUTPUT_COLUMNS = [
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


def _walk_dicts(value: Any) -> Iterable[Mapping[str, Any]]:
    """Yield dictionaries nested inside JSON-LD lists and graph objects."""

    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _walk_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_dicts(child)


def _first_value(record: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = record.get(key)
        if value not in (None, "", [], {}):
            return value
    return None


def _text(value: Any) -> str | None:
    if value in (None, ""):
        return None
    if isinstance(value, Mapping):
        value = value.get("name") or value.get("value") or value.get("text")
    if value in (None, ""):
        return None
    return str(value).strip() or None


def parse_weekly_rent(value: Any) -> float | None:
    """Parse a weekly rent such as ``$620 per week`` or ``620 pw``."""

    if value in (None, ""):
        return None
    if isinstance(value, (int, float)) and not pd.isna(value):
        return float(value)
    text = str(value)
    match = re.search(r"\$\s*([0-9][0-9,]*(?:\.\d+)?)", text)
    if not match:
        match = re.search(
            r"(?<!\w)([0-9][0-9,]*(?:\.\d+)?)\s*(?:per\s*week|pw|p/w|weekly)\b",
            text,
            flags=re.IGNORECASE,
        )
    if not match:
        return None
    return float(match.group(1).replace(",", ""))


def parse_integer(value: Any) -> int | None:
    if value in (None, ""):
        return None
    match = re.search(r"\d+", str(value))
    return int(match.group()) if match else None


def _normalise_property_type(record: Mapping[str, Any]) -> str | None:
    value = _first_value(record, "property_type", "additionalType", "@type")
    if isinstance(value, list):
        value = value[0] if value else None
    value = _text(value)
    if value and value.startswith("http"):
        value = value.rsplit("/", 1)[-1]
    return value


def _address_parts(address: Any) -> tuple[str | None, str | None, str | None, str | None]:
    if isinstance(address, Mapping):
        street = _text(address.get("streetAddress"))
        suburb = _text(address.get("addressLocality"))
        state = _text(address.get("addressRegion"))
        postcode = _text(address.get("postalCode"))
        full = ", ".join(part for part in (street, suburb, state, postcode) if part)
        return full or None, suburb, state, postcode
    return _text(address), None, None, None


def _coordinates(record: Mapping[str, Any]) -> tuple[float | None, float | None]:
    geo = record.get("geo") or record.get("location", {}).get("geo") if isinstance(record.get("location"), Mapping) else record.get("geo")
    if not isinstance(geo, Mapping):
        return None, None
    try:
        latitude = float(geo.get("latitude")) if geo.get("latitude") is not None else None
        longitude = float(geo.get("longitude")) if geo.get("longitude") is not None else None
    except (TypeError, ValueError):
        return None, None
    return latitude, longitude


def _offer_price(record: Mapping[str, Any]) -> Any:
    offers = record.get("offers")
    if isinstance(offers, list):
        offers = offers[0] if offers else None
    if isinstance(offers, Mapping):
        return _first_value(offers, "price", "lowPrice", "highPrice")
    return None


def _stable_listing_id(listing_url: str | None, address: str | None, title: str | None) -> str:
    key = "|".join(part or "" for part in (listing_url, address, title))
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def _record_from_jsonld(
    item: Mapping[str, Any], source_page_url: str, collected_at_utc: str
) -> dict[str, Any] | None:
    address, suburb, state, postcode = _address_parts(item.get("address"))
    title = _text(_first_value(item, "name", "headline", "title"))
    listing_url = _text(_first_value(item, "url", "mainEntityOfPage"))
    if isinstance(listing_url, str):
        listing_url = urljoin(source_page_url, listing_url)
    rent = parse_weekly_rent(_offer_price(item))
    if rent is None and not address and not title:
        return None
    latitude, longitude = _coordinates(item)
    return {
        "listing_id": _stable_listing_id(listing_url, address, title),
        "source": urlparse(source_page_url).netloc or "local_fixture",
        "source_page_url": source_page_url,
        "listing_url": listing_url,
        "collected_at_utc": collected_at_utc,
        "title": title,
        "address": address or title,
        "suburb": suburb,
        "state": state,
        "postcode": postcode,
        "property_type": _normalise_property_type(item),
        "weekly_rent_aud": rent,
        "bedrooms": parse_integer(_first_value(item, "numberOfBedrooms", "bedrooms")),
        "bathrooms": parse_integer(
            _first_value(item, "numberOfBathroomsTotal", "numberOfBathrooms", "bathrooms")
        ),
        "parking_spaces": parse_integer(
            _first_value(item, "numberOfParkingSpaces", "parkingSpaces", "parking")
        ),
        "latitude": latitude,
        "longitude": longitude,
    }


def _record_from_html_card(card: Any, source_page_url: str, collected_at_utc: str) -> dict[str, Any] | None:
    text = " ".join(card.stripped_strings)
    rent = parse_weekly_rent(text)
    if rent is None:
        return None
    heading = card.find(["h1", "h2", "h3", "h4"])
    title = _text(heading.get_text(" ", strip=True) if heading else None)
    link = card.find("a", href=True)
    listing_url = urljoin(source_page_url, link["href"]) if link else None
    suburb_match = re.search(r"\b([A-Z][A-Za-z -]+)\s+(?:VIC|Victoria)\b", title or text)
    suburb = suburb_match.group(1).strip() if suburb_match else None
    return {
        "listing_id": _stable_listing_id(listing_url, title, text),
        "source": urlparse(source_page_url).netloc or "local_fixture",
        "source_page_url": source_page_url,
        "listing_url": listing_url,
        "collected_at_utc": collected_at_utc,
        "title": title,
        "address": title,
        "suburb": suburb,
        "state": "VIC" if re.search(r"\bVIC\b|\bVictoria\b", text, flags=re.I) else None,
        "postcode": None,
        "property_type": None,
        "weekly_rent_aud": rent,
        "bedrooms": parse_integer(re.search(r"(\d+)\s*(?:beds?|bedrooms?)", text, flags=re.I).group(1) if re.search(r"(\d+)\s*(?:beds?|bedrooms?)", text, flags=re.I) else None),
        "bathrooms": parse_integer(re.search(r"(\d+)\s*(?:baths?|bathrooms?)", text, flags=re.I).group(1) if re.search(r"(\d+)\s*(?:baths?|bathrooms?)", text, flags=re.I) else None),
        "parking_spaces": parse_integer(re.search(r"(\d+)\s*(?:parking|car spaces?)", text, flags=re.I).group(1) if re.search(r"(\d+)\s*(?:parking|car spaces?)", text, flags=re.I) else None),
        "latitude": None,
        "longitude": None,
    }


def extract_listing_records(
    html: str, source_page_url: str = "file://local-fixture", collected_at_utc: str | None = None
) -> list[dict[str, Any]]:
    """Extract normalised listing records from one HTML page."""

    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    collected_at_utc = collected_at_utc or datetime.now(timezone.utc).isoformat()
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = script.string or script.get_text()
        try:
            payload = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            continue
        for item in _walk_dicts(payload):
            record = _record_from_jsonld(item, source_page_url, collected_at_utc)
            if record and record["listing_id"] not in seen_ids and record["weekly_rent_aud"] is not None:
                records.append(record)
                seen_ids.add(record["listing_id"])

    for card in soup.select("article, [data-testid*='listing'], [class*='listing-card']"):
        record = _record_from_html_card(card, source_page_url, collected_at_utc)
        if record and record["listing_id"] not in seen_ids:
            records.append(record)
            seen_ids.add(record["listing_id"])

    return records


def records_to_frame(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Return a stable-column DataFrame for downstream analysis."""

    frame = pd.DataFrame.from_records(records)
    for column in OUTPUT_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame.loc[:, OUTPUT_COLUMNS]


def robots_allows(url: str, user_agent: str = "MAST30034-StudentProject/1.0") -> bool:
    """Fail closed when robots.txt cannot be read or disallows the URL."""

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return True
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    parser = RobotFileParser(robots_url)
    try:
        parser.read()
    except OSError as exc:
        raise PermissionError(f"Could not verify robots.txt for {parsed.netloc}; refusing to crawl") from exc
    return parser.can_fetch(user_agent, url)


def fetch_html(
    url: str,
    *,
    session: requests.Session | None = None,
    timeout_seconds: int = 30,
    user_agent: str = "MAST30034-StudentProject/1.0 (+replace-with-group-contact)",
) -> str:
    """Fetch one page only after a robots.txt permission check."""

    import requests

    if not robots_allows(url, user_agent=user_agent):
        raise PermissionError(f"robots.txt does not allow this automated request: {url}")
    client = session or requests.Session()
    response = client.get(url, headers={"User-Agent": user_agent}, timeout=timeout_seconds)
    response.raise_for_status()
    return response.text


def crawl_pages(
    urls: Iterable[str], *, min_delay_seconds: float = 2.0, user_agent: str = "MAST30034-StudentProject/1.0 (+replace-with-group-contact)"
) -> list[dict[str, Any]]:
    """Crawl a small set of authorised URLs politely and parse each page."""

    import requests

    urls = list(urls)
    records: list[dict[str, Any]] = []
    with requests.Session() as session:
        for index, url in enumerate(urls):
            html = fetch_html(url, session=session, user_agent=user_agent)
            records.extend(extract_listing_records(html, source_page_url=url))
            if index < len(urls) - 1:
                time.sleep(min_delay_seconds)
    return records


def save_records(records: Sequence[Mapping[str, Any]], output_path: str | Path) -> pd.DataFrame:
    """Save normalised records as a local CSV and return the DataFrame."""

    frame = records_to_frame(records)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame

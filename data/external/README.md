# External data notes

Keep a short provenance record for every external dataset used in the project. Record:

- provider and dataset name;
- source URL or API endpoint;
- access/download date;
- licence or usage conditions;
- fields used and any transformations; and
- whether the file is reproducible from a public download/API.

Large external files are excluded from Git by the repository `.gitignore`. Commit small documentation and source notes instead.

## Sprint 1 dataset

We use the public [Australian Rental Market Data 2026 dataset on Kaggle](https://www.kaggle.com/datasets/kanchana1990/australian-rental-market-data-2026)
as a temporary external dataset while direct access to the property portals is
blocked. The dataset page reports 6,767 Australian rental listings, including
Victoria records, with rent, property features, suburbs, postcodes and
coordinates.

- Provider/author: Kanchana1990, Kaggle.
- Downloaded: 2026-09-03.
- Local file: `data/external/australian-rental-market-data-2026/australian_rental_market_2026.csv`.
- Claimed licence: CC0 / public domain, according to the Kaggle data card.
- Fields used: `price_display`, `propertyType`, `suburb`, `state`, `postcode`,
  `street_address`, `bedrooms`, `bathrooms`, `parking_spaces`, `latitude`,
  and `longitude`.
- Transformation: `src/kaggle_data.py` filters `state == VIC` and maps the
  Kaggle names into the project's canonical listing schema.

This is a static 2026 snapshot, not a historical time series. It supports the
Sprint 1 map and a current-rent baseline, but the three-year forecast will
require additional historical or repeated-snapshot rental data and external
forecast variables. The group should confirm with the tutor that this dataset
can supplement the required listing-source collection.

## Historical and affordability layers

### Homes Victoria Rental Report

The [Homes Victoria Rental Report time series](https://www.dffh.vic.gov.au/publications/rental-report)
is the historical rental layer. The September 2025 workbook contains moving
annual median rents and new-lease counts by Victorian suburb/town, property
type, and quarter from 2000 to September 2025. The project keeps the 2024 Q1
to 2025 Q3 slice in `data/processed/homes_victoria_rents_2024_2025.csv`.

- Raw file: `data/external/homes_victoria/moving_annual_rent_suburb_sep_2025.xlsx`.
- Downloaded: 2026-09-03.
- Licence: Creative Commons Attribution 4.0 International.
- Parser: `src/homes_victoria.py`.
- Main fields: `state`, `region`, `suburb`, `property_type`, `period_end`,
  `new_lease_count`, and `median_weekly_rent_aud`.

### Homes Victoria Quarterly Data Tables

The [Rental Report quarterly data tables](https://discover.data.vic.gov.au/dataset/rental-report-quarterly-data-tables)
are the official aggregate companion to the suburb time series. The project
uses the September 2025 workbook only for Table 1 overall metropolitan/
regional benchmarks and Table 3 property-type benchmarks. These values are
shown as reference context for a listing when its ABS geography and property
type have an exact supported match; they are not appended as additional
listings. Table 2 is not used because its statistical regions are not SA2s.

- Raw file: `data/external/homes_victoria/quarterly_tables_sep_2025.xlsx`.
- Downloaded: 2026-09-03.
- Provider: Homes Victoria; the DataVic record identifies quarterly updates
  and a Creative Commons Attribution 4.0 International licence.
- Parser: `src/homes_victoria.py` (`load_quarterly_benchmarks`).
- Processed output: `homes_victoria_quarterly_benchmarks` in the integration
  loader.

### ABS ASGS 2021 SA2 Digital Boundaries

The [ABS ASGS Edition 3 digital boundary files](https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs/edition-3-july-2021-june-2026/access-and-downloads/digital-boundary-files)
provide the official 2021 SA2 polygons in GDA2020. The project filters the
shapefile to Victoria (`STE_CODE21 == 2`), converts it to WGS84 for Folium,
and assigns each of the existing 1,118 listing coordinates to an SA2 using a
point-in-polygon join.

- Raw archive: `data/external/abs_asgs/SA2_2021_AUST_SHP_GDA2020.zip`.
- Extracted shapefile: `data/external/abs_asgs/SA2_2021_AUST_GDA2020/`.
- Downloaded: 2026-09-03.
- Main fields used: `SA2_CODE21`, `SA2_NAME21`, `SA3_NAME21`, `SA4_NAME21`,
  `GCC_NAME21`, and `STE_CODE21`.
- Parser/assignment: `src/sa2.py`.

### Evaluated but excluded: Victoria State GEO & Postcodes

The [Kaggle Victoria State GEO & Postcodes dataset](https://www.kaggle.com/datasets/aasirwaseer/victoria-state-geo-and-postcodes)
contains postcode/locality lookup fields and representative coordinates. It
does not add listing records, listing rents, or official SA2 polygons, and the
current 1,118 listings already have coordinates. It is therefore not merged
into the map, avoiding duplicated or invented property points.

### Anglicare Victoria Rental Affordability Snapshot

The [2026 Anglicare Victoria Rental Affordability Snapshot](https://www.anglicarevic.org.au/research/victorian-rental-affordability-snapshot-2026/)
provides an annual Victoria-wide affordability and supply comparison. Its 2026
report compares March snapshots from 2022 to 2026 and reports listing counts,
median weekly rents, and affordability by household type. The extracted tables
are `anglicare_ras_victoria_2022_2026_summary.csv` and
`anglicare_ras_2026_household_affordability.csv`.

- Raw file: `data/external/anglicare_victoria/rental_affordability_snapshot_2026.pdf`.
- Downloaded: 2026-09-03.
- Coverage: Victoria only; the 2026 snapshot covers 32 metropolitan and 48
  regional LGAs.
- This is a supplementary affordability layer, not individual listing data.

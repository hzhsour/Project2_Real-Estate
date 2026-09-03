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

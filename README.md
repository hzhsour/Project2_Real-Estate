# MAST30034 Project 2 - Victoria Rental Prices

This repository is for the MAST30034 real-estate industry project. The project studies rental prices for residential properties in Victoria and is designed to answer three business questions:

1. Which internal and external features are most useful for predicting rental prices?
2. Which suburbs have the highest predicted rental-price growth?
3. Which suburbs are the most liveable and affordable under the group's chosen metrics?

## Current project decision

The initial modelling horizon is **three years**, following the group's current interpretation of Sprint 6. The assignment overview also mentions five-year growth, so the final horizon should be confirmed with the tutor. The horizon will be kept as a configuration value rather than hard-coded throughout the analysis.

Sprint 1 focuses on:

- starting the property-listing collection plan;
- defining a reproducible property schema;
- loading an authorised external listing export while the direct portal source
  is being confirmed; and
- producing a property-location map for Victoria.

The local HTML fixture is synthetic and is included only to test the parser. It is not analysis data.

## Repository structure

- `notebooks/01_sprint1_web_scraping_and_mapping.ipynb`: Sprint 1 walkthrough and parser/map smoke test.
- `src/scraper.py`: source-agnostic listing parser, robots check, and crawl helpers.
- `src/kaggle_data.py`: adapter for the Kaggle rental snapshot used in Sprint 1.
- `src/visualisation.py`: interactive property-location map.
- `data/raw/`: local raw pages or downloaded files; large raw data are ignored by Git.
- `data/external/`: external datasets and provenance notes.
- `data/processed/`: generated tables and maps; ignored by Git.

## Setup

```bash
python -m venv project2_Mast30034
```

Activate the environment and install the requirements:

```bash
pip install -r requirements.txt
```

Run the notebook from the repository root:

```bash
jupyter notebook
```

## Sprint 1 workflow

1. Confirm an authorised property source with the tutor or use the supplied course skeleton/API/export.
2. Record the source URL, access date, licence, and attribution in `data/external/README.md`.
3. Use `src/kaggle_data.py` to load the Kaggle snapshot and filter Victoria.
4. Check rent, bedrooms, bathrooms, parking, suburb, and coordinates.
5. Save raw inputs locally and keep generated outputs out of GitHub.
6. Produce the location map and show it at the weekly checkpoint.

The live crawler is deliberately opt-in. It checks `robots.txt` and raises an error when automated access is not allowed; it does not bypass bot controls, rate limits, login walls, or other access restrictions. Do not run it against a source unless your group has permission to do so.

## Planned three-year modelling path

The Sprint 1 listing scrape is a current asking-rent snapshot. By itself, it cannot support a defensible three-year time forecast. Later sprints therefore need a time dimension, such as repeated snapshots or an authorised historical rental dataset, plus suburb-level population/income forecasts. The planned sequence is:

- Sprint 1: property scrape design, parser validation, and location map.
- Sprint 2: complete property and external data collection and save the data.
- Sprint 3: add route distance and nearby amenity features.
- Sprint 4: build a baseline and interpretable model; define the three-year target.
- Sprint 5: compare predictions with business-oriented liveability and affordability metrics.
- Sprint 6: report three-year predictions and answer the three business questions.

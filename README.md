# Victoria Rental Market Analysis

## Project Overview

This repository documents the University of Melbourne MAST30034 Applied Data Science industry project on Victorian residential rental markets. It combines saved rental-listing snapshots with Homes Victoria rent histories, SA2 geography, socioeconomic context, transport accessibility and local amenities to examine current asking rents, modelled future rental growth, and relative affordability and liveability.

The three questions use different outcomes and geographic units. Q1 analyses current asking-rent differences using SA2 summaries; Q2 models cumulative growth in Homes Victoria rental-history markets; Q3 screens SA2s using project-selected affordability and liveability measures. Results are decision-support evidence, not causal effects, investment advice or guaranteed forecasts.

## Research Questions

| Question | Project wording | Implemented interpretation |
| --- | --- | --- |
| Q1 | What are the most important internal and external features in predicting rental prices? | Predict current SA2-level median asking rent from property and neighbourhood features. This is predictive association, not a causal estimate. |
| Q2 | What are the top 10 suburbs with the highest predicted growth rate? | Compare three- and five-year cumulative growth in 145 native Homes Victoria rental-history markets. These reporting groups are not necessarily suburbs or SA2s, so the output is not a direct forecast for every suburb. |
| Q3 | What are the most liveable and affordable suburbs according to your chosen metrics? | Screen SA2 areas using a rent-to-income affordability burden and a provisional equal-weight local-access/liveability score. SA2s should not automatically be called suburbs. |

## Data Sources

| Source | Use and coverage | Provenance and constraints |
| --- | --- | --- |
| Course-provided Domain Victoria rental snapshot | Property asking rents, characteristics and locations; September 2025 snapshot, 12,717 cleaned listing rows. | Supplied by teaching staff for MAST30034. Use only as authorised for the course; do not redistribute the raw files or listing-level maps. |
| Australian Rental Market Data 2026 (Kaggle) | Additional Victoria listing snapshot with rent, property attributes and location fields. | Separately obtained public dataset, filtered to Victoria and retained as a distinct source. It is a static snapshot, not a historical rent series. The combined Domain and Kaggle integration contains 13,835 records with source labels. |
| Homes Victoria / DFFH | Moving-annual median weekly rents for Q2, 2000 Q1–2025 Q3; recent quarter-specific and mapped SA2 rent context, 2023 Q1–2025 Q3. | Official historical aggregates. The moving-annual target overlaps between adjacent quarters and the archive is revised; historical values do not provide complete origin-time data vintages. |
| ABS, Census 2021, SEIFA and VIF2023 | 2021 SA2 boundaries, household income, IRSAD scores and population projections used as geographic or socioeconomic context. | Area-level measures, not additional rental listings. VIF2023 population values are projections, not observed counts. |
| Transport Victoria and Victorian open data | Public transport stops/lines and statewide Vicmap amenity information; schools, parks, shopping and entertainment measures. | Used for proximity/accessibility and local-amenity features. Saved OpenRouteService driving-route results are distinct from straight-line proximity measures. |
| Anglicare Victoria Rental Affordability Snapshot | Supplementary Victoria-wide affordability and rental-supply context. | A supporting benchmark, not an individual-property listing source or the Q2 forecasting target. |

The final project did not obtain its rental listings through live web scraping. Scraping was initially considered, then cancelled once the teaching staff supplied the required rental dataset. The implemented analysis uses that course-provided Domain snapshot together with the separately obtained Kaggle snapshot and other previously obtained public datasets. The project does not document an upstream collection method for the teacher-provided Domain files.

Further source paths, download notes, transformations and usage conditions are recorded in [the local data provenance notes](data/external/README.md). The statewide amenity package and accessibility definitions are described in [the Vicmap source notes](raw_data/shopping/README_WEEK8_VICTORIA_POI.md) and [the accessibility workflow](docs/SPRINT3_ACCESSIBILITY.md).

## Project Workflow

**Listing snapshots and initial inspection → geographic, historical-rent and socioeconomic integration → transport and amenity enrichment → current-rent feature analysis and initial forecasts → rolling and long-history forecast validation → final Q1–Q3 synthesis and registered Q2 research forecasts.**

## Sprint Progress

### Sprint 1 — Data acquisition and initial exploration

**Objective**

Inspect the available rental records, define a consistent property schema and establish an initial geographic view.

**Implementation**

The team used the teaching-provided Domain snapshot and a separately obtained Kaggle snapshot. Work included checking rents, property characteristics, suburb/postcode and coordinates, preserving source labels, and preparing an initial listing-location map against official SA2 geography. A local synthetic HTML fixture was used for a parser smoke test; it was not rental analysis data. Live scraping was cancelled when the required teaching dataset became available. The notebook's original filename retains an early web-scraping label, but that is not the final data-collection method.

**Key outputs**

An inspected, source-labelled listing base and an initial Victoria property-location map, with the input observations kept distinct by source.

**Notebook**

[Sprint 1 notebook](notebooks/01_sprint1_web_scraping_and_mapping.ipynb)

### Sprint 2 — Data integration and geographic context

**Objective**

Combine listing records with compatible historical-rent, SA2 and socioeconomic context while retaining each dataset's observational unit.

**Implementation**

The integration cleans and stacks the Domain and Kaggle snapshots, assigns listing coordinates to ABS ASGS 2021 SA2 polygons, and joins area context such as household income, IRSAD and VIF2023 population projections. Homes Victoria historical rents and quarterly benchmarks remain separate aggregate layers. A documented suburb-group-to-SA2 crosswalk supports the recent historical map; ambiguous groups are left unmapped rather than assigned to a guessed area.

**Key outputs**

Integrated listing and SA2 context tables, Homes Victoria rent-history layers, source/coverage checks and an interactive SA2 rent-and-context explorer.

**Notebook**

[Sprint 2 notebook](notebooks/02_data_integration_and_history.ipynb)

### Sprint 3 — Transport and amenities

**Objective**

Add transport accessibility and local amenity features relevant to rental-price interpretation and the Q3 screen.

**Implementation**

Listing coordinates were related to public-transport and facility reference points. Saved driving routes to Melbourne CBD and to the straight-line-nearest selected train station were built with OpenRouteService. Statewide Vicmap features support park, shopping and entertainment proximity/count measures; school and transport measures remain separately labelled. The early model comparison is exploratory and predates the complete statewide amenity integration.

**Key outputs**

Accessibility and amenity feature tables, route-coverage QA and the Victoria Rental Property Explorer. Four listings lacked valid routing origins; reported route coverage uses the 13,831 valid-coordinate listings.

**Notebook**

[Sprint 3 notebook](notebooks/03_sprint3_accessibility.ipynb)

### Sprint 4 — Feature analysis and modelling

**Objective**

Explore rental-price associations, compare feature groups and establish initial forecasting baselines.

**Implementation**

Current asking rent was analysed at SA2 level using descriptive comparisons, Ridge Regression and Random Forest. Nested feature sets assessed the incremental information in socioeconomic, transport and amenity context. Rent-history forecasts also began with transparent baselines and chronological validation. The initial Naive baseline performed well on a short validation horizon but, by construction, assigned zero growth and could not rank future high-growth markets.

**Key outputs**

Exploratory feature comparisons and importance results, saved forecast baselines and explicit leakage/geography cautions. Correlations and model importance are not causal explanations.

**Notebook**

[Sprint 4 notebook](notebooks/04_sprint4_forecasting_and_feature_analysis.ipynb)

### Sprint 5 — Forecasting and validation

**Objective**

Develop and compare rent-history forecasting approaches, test short/medium horizons and evaluate longer-horizon rent levels and growth rankings.

**Implementation**

Expanding-window 1-, 2- and 4-quarter comparisons selected Damped Holt as the original short/medium rent-forecast scenario, although it did not win every horizon. Separate long-history experiments used Homes Victoria moving-annual rents and distinct rental-history markets; Drift performed best on the reported 20-quarter rent-level RMSE. These are different targets and validation designs. The later Ridge research was not part of the Sprint 5 implementation.

**Key outputs**

Rolling-origin comparisons, model and history-group audits, long-history backtests, preliminary Q3 affordability/liveability measures and an interim project summary. The original Damped Holt outputs were retained for the following final-results integration.

**Notebook**

[Sprint 5 summary notebook](notebooks/05_sprint5_summary.ipynb)

### Sprint 6 — Final results and research integration

**Objective**

Bring the saved Q1–Q3 analyses together and integrate the completed Q2 model-selection and forecast-registration research.

**Implementation**

The final notebook reports Q1 and Q3 results, preserves the original Damped Holt scenarios, and adds the timestamped Ridge21 three-year and Ridge18 five-year research forecasts. The supporting Q2 research Stages 1–10 are methodological work within Sprint 6, not ten additional course sprints. Forecasts are displayed at native Homes Victoria market level; geographic mapping diagnostics do not turn them into independent SA2 forecasts.

**Key outputs**

The final results notebook, Q1/Q3 summaries and the project-local frozen Q2 package: [Ridge21 3Y forecast](data/processed/sprint6/q2_registered_research/ridge21_3y_forecast.csv), [Ridge18 5Y forecast](data/processed/sprint6/q2_registered_research/ridge18_5y_forecast.csv) and [Q2 provenance](data/processed/sprint6/q2_registered_research/q2_research_provenance.json).

**Notebook**

[Sprint 6 final-results notebook](notebooks/06_sprint6_final_results.ipynb)

## Final Results

### Q1 — Current asking rent

On the saved 98-SA2 holdout, the property-only Random Forest had an RMSE of AUD 131.82 per week; the full feature set reduced it to AUD 114.64 per week, approximately 13% lower. IRSAD was important in the saved permutation-importance analysis. These are area-level predictive associations on the available split, not individual-property causal effects or proof of transfer to new geographies.

### Q2 — Future rental growth

The Q2 target is cumulative percentage change in moving-annual median weekly rent, not annualised inflation or investment return. The registered forecasts use a 2025 Q3 information reference and were issued on 16 October 2026; they were not predictions issued in September 2025.

| Research candidate | Horizon and target | Markets | Matched historical growth RMSE | Mean tie-aware Top-10 hits |
| --- | --- | ---: | ---: | ---: |
| Origin-centred Ridge21, 21 features | 3 years, 12 quarters; 2028 Q3 | 145 | 7.54 percentage points | 2.82 / 10 |
| Origin-centred Ridge18, 18 features | 5 years, 20 quarters; 2030 Q3 | 145 | 12.72 percentage points | 2.86 / 10 |

The historical comparison covers 145 markets and 11 annual Q3 origins from 2010 Q3 to 2020 Q3. The overlapping targets use revised data, so these are descriptive backtest results, not independent prospective trials or percentages of correct predictions. The simple lower-starting-rent ranking baseline had higher mean Top-10 recovery in those primary averages (3.40/10 at 3Y and 3.82/10 at 5Y). In two additional 2021–2022 three-year origins, Ridge21 was better than Ridge18 on RMSE (25.31 vs 25.45 pp) and mean Top-10 hits (5.5 vs 3.5), but both underpredicted common growth by about 24.06 pp; approximately 90.4% of Ridge21's squared error was attributed to the common-growth component.

Damped Holt remains the original saved production scenario and historical baseline; it was selected in the earlier short/medium comparison and its outputs remain intact. In the matched cumulative-growth comparison, its RMSE was 9.80 pp at 3Y and 19.11 pp at 5Y. Ridge21 and Ridge18 are later, horizon-specific research candidates, not deployment-approved or proven reliable winners. No registered target has a mature outcome, and the model ranking is period-sensitive.

### Q3 — Affordability and liveability

The saved median-based screen first identified 124 SA2 candidates and retained 105 after the existing listing-count and special-purpose-area filters. Affordability burden compares current median asking rent with 2021 Census median household weekly income; liveability combines percentile-scaled train, school, parks, shopping and entertainment measures with provisional equal weights. South Melbourne, Collingwood and Prahran–Windsor are examples from the saved results, not universally optimal suburb recommendations. Q2 rental-history groups and Q3 SA2s are not interchangeable; the project does not claim one fully reconciled Q2/Q3 shortlist.

## Repository Structure

```text
notebooks/                         Sprint 1–6 analysis notebooks and summary
src/                               data integration, feature and model code
data/external/README.md            source provenance, access and usage notes
data/processed/sprint6/            saved final summaries and research outputs
  q2_registered_research/          frozen Ridge21/Ridge18 forecasts and provenance
raw_data/shopping/                 statewide amenity package and source notes
docs/                              accessibility and project-support documentation
requirements.txt                   recorded Python dependencies
```

Useful entry points are [the final notebook](notebooks/06_sprint6_final_results.ipynb), [the accessibility method and QA notes](docs/SPRINT3_ACCESSIBILITY.md), and [the statewide amenity source README](raw_data/shopping/README_WEEK8_VICTORIA_POI.md).

## Reproducibility and Data Availability

Install the dependencies listed in `requirements.txt` in the project environment, then open the notebooks from the repository root:

```powershell
python -m venv .venv
\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
jupyter notebook
```

The Sprint 6 notebook contains saved results and reads the frozen Q2 forecast tables and provenance from `data/processed/sprint6/q2_registered_research/`; it does not refit Ridge21 or Ridge18. The Q2 forecast files each contain 145 market rows. To reproduce earlier stages or regenerate all analyses, the team needs the corresponding authorised raw and processed inputs, including the restricted course Domain snapshot, ABS and government source files, local transport/amenity snapshots, and saved route/cache outputs. A clean clone should not be assumed to reproduce every result from Git alone.

The Sprint 6 Q2 implementation is now available as reusable source in [`src/sprint6_q2_model.py`](src/sprint6_q2_model.py) and [`src/sprint6_q2_registry.py`](src/sprint6_q2_registry.py); its research-to-project provenance is documented in [`docs/SPRINT6_Q2_CODE_PROVENANCE.md`](docs/SPRINT6_Q2_CODE_PROVENANCE.md). Use the workflow appropriate to the task:

1. **View frozen results:** open the final notebook; it reads the registered 3Y Ridge21 and 5Y Ridge18 files and validates their hashes, schemas, identities and rankings. This does not fit a model.
2. **Run source-level checks:** `python -m unittest discover -s tests -p "test_sprint6_q2.py"`. Where the local processed Homes Victoria history is available, the tests also refit the two current candidates in memory and compare them with the frozen predictions; they do not write or register forecasts.
3. **Re-run retrospective validation:** call `run_matched_historical_validation(project_root)` from `src.sprint6_q2_model`. It returns predictions and per-origin metrics in memory for the pre-specified 2010–2020 Q3 origins; these are retrospective results on revised history, not an independent prospective test.
4. **Recompute from authorised processed data:** use `load_homes_victoria_history`, `build_market_panels`, `build_growth_design` and `fit_origin_centred_ridge` from the same module. This requires the saved Homes Victoria long-history input at `data/processed/sprint5/long_history_validation/`; a clean clone may not contain it. The source has no forecast-package or ledger writer, and the raw or restricted source data must be obtained through the authorised project workflow.

Generated files under `data/processed/` are subject to repository ignore rules. The frozen Q2 package, final Sprint 6 notebook and reusable Q2 source modules were included in commit `bc71eff`; other local notebooks, documentation and generated outputs are not all committed. Do not publish the restricted Domain records. Reuse saved route results for ordinary notebook review; new route requests are a separate operation and are not needed to inspect the final results.

## Limitations and Responsible Use

- The Domain snapshot is restricted to authorised course use. Keep its raw files and listing-level maps in an approved private/local location.
- Listing, Census, population, transport and amenity sources have different dates and coverage. Homes Victoria moving-annual observations overlap, the archive is revised, and complete historical publication vintages are unavailable.
- Neither Ridge forecast has reached its 2028 Q3 or 2030 Q3 target. Historical validation is modest and era-sensitive; no calibrated prediction intervals are available.
- Homes Victoria rental-history groups do not automatically match SA2s or suburbs. Fixed-weight mapped proxies are not independently fitted forecasts and do not validate future accuracy.
- Q1 holdout/CV results do not establish causal effects or reliable transfer to new geographies. Q3 uses a project-defined screen and omits household preferences and other affordability dimensions.
- The data do not directly measure vacancy, enquiries, applications or days on market. New-lease counts and listing-based features should not be described as direct rental demand.
- Maps and forecasts are offline research prototypes, not a live rental-market product, property valuation or investment recommendation.

## Acknowledgements

Official source links and local attribution/usage details are provided in [the data provenance notes](data/external/README.md). Key upstream references include the [Homes Victoria Rental Report](https://www.dffh.vic.gov.au/publications/rental-report), [ABS ASGS Edition 3 boundaries](https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs/edition-3-july-2021-june-2026/access-and-downloads/digital-boundary-files), [Transport Victoria public transport data](https://opendata.transport.vic.gov.au/dataset/public-transport-lines-and-stops), [Victorian Vicmap Features of Interest](https://discover.data.vic.gov.au/dataset/vicmap-features-features-of-interest-index-foi-centroid), and the [Kaggle Australian Rental Market Data 2026 snapshot](https://www.kaggle.com/datasets/kanchana1990/australian-rental-market-data-2026).

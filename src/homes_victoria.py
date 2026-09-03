"""Load and reshape Homes Victoria Rental Report workbooks."""

from pathlib import Path

import pandas as pd


def _period_end(value: object) -> pd.Timestamp:
    """Convert workbook labels such as ``Mar 2025`` to quarter-end dates."""

    return pd.to_datetime(str(value).strip(), format="%b %Y") + pd.offsets.MonthEnd(0)


def load_moving_annual_rents(
    workbook_path: str | Path,
    *,
    start_period: str | None = None,
    end_period: str | None = None,
) -> pd.DataFrame:
    """Return Homes Victoria suburb rents in tidy long format.

    The workbook has one sheet per property type. Each sheet stores quarterly
    count/median pairs in a wide layout. The source covers Victoria only.
    """

    workbook_path = Path(workbook_path)
    excel = pd.ExcelFile(workbook_path)
    records: list[dict[str, object]] = []

    for sheet_name in excel.sheet_names:
        raw = pd.read_excel(workbook_path, sheet_name=sheet_name, header=None)
        periods = {
            column: _period_end(raw.iat[1, column])
            for column in range(2, raw.shape[1], 2)
            if pd.notna(raw.iat[1, column])
        }

        for row in range(3, raw.shape[0]):
            region = raw.iat[row, 0]
            suburb = raw.iat[row, 1]
            if pd.isna(suburb):
                continue

            for count_column, period in periods.items():
                count = raw.iat[row, count_column]
                median = raw.iat[row, count_column + 1]
                records.append(
                    {
                        "state": "VIC",
                        "region": str(region).strip() if pd.notna(region) else pd.NA,
                        "suburb": str(suburb).strip(),
                        "property_type": sheet_name,
                        "period_end": period,
                        "new_lease_count": pd.to_numeric(count, errors="coerce"),
                        "median_weekly_rent_aud": pd.to_numeric(
                            median, errors="coerce"
                        ),
                    }
                )

    rents = pd.DataFrame.from_records(records)
    rents["period_end"] = pd.to_datetime(rents["period_end"])

    if start_period is not None:
        rents = rents.loc[rents["period_end"] >= pd.Timestamp(start_period)]
    if end_period is not None:
        rents = rents.loc[rents["period_end"] <= pd.Timestamp(end_period)]

    return rents.reset_index(drop=True)


def load_quarterly_benchmarks(
    workbook_path: str | Path,
    *,
    snapshot_period: str = "Sep 2025",
) -> pd.DataFrame:
    """Extract useful regional and property-type benchmarks from Table 1/3.

    These are aggregate Homes Victoria reference values, not extra listing
    records. Table 2 is deliberately not parsed because its statistical
    regions are not the same geography as the project's SA2 polygons.
    """

    workbook_path = Path(workbook_path)

    def _number(value: object) -> object:
        parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        return parsed if pd.notna(parsed) else pd.NA

    records: list[dict[str, object]] = []

    table1 = pd.read_excel(workbook_path, sheet_name="Table 1", header=None)
    for row in range(3, table1.shape[0]):
        label = table1.iat[row, 0]
        median = _number(table1.iat[row, 1])
        if pd.isna(label) or pd.isna(median):
            continue
        label = str(label).strip()
        if label not in {"Melbourne", "Regional Victoria"}:
            continue
        region_group = (
            "Metropolitan Melbourne" if label == "Melbourne" else "Regional Victoria"
        )
        records.append(
            {
                "snapshot_period": snapshot_period,
                "benchmark_group": "geography",
                "region_group": region_group,
                "benchmark_label": region_group,
                "median_weekly_rent_aud": median,
                "quarterly_change": _number(table1.iat[row, 3]),
                "annual_change": _number(table1.iat[row, 4]),
            }
        )

    table3 = pd.read_excel(workbook_path, sheet_name="Table 3", header=None)
    current_group: str | None = None
    for row in range(3, table3.shape[0]):
        label = table3.iat[row, 0]
        if pd.isna(label):
            continue
        label = str(label).strip()
        median = _number(table3.iat[row, 1])
        if label in {"Metropolitan Melbourne", "Regional Victoria"}:
            current_group = label
            continue
        if current_group is None or pd.isna(median):
            continue
        records.append(
            {
                "snapshot_period": snapshot_period,
                "benchmark_group": "property_type",
                "region_group": current_group,
                "benchmark_label": label,
                "median_weekly_rent_aud": median,
                "quarterly_change": _number(table3.iat[row, 2]),
                "annual_change": _number(table3.iat[row, 3]),
            }
        )

    return pd.DataFrame.from_records(records)

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

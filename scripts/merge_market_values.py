from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

APPEARANCES_PATH = DATA_DIR / "appearances.csv"
VALUATIONS_PATH = DATA_DIR / "player_valuations.csv"
OUTPUT_PATH = DATA_DIR / "appearances_with_market_values.csv"


def main() -> None:
    appearances = pd.read_csv(APPEARANCES_PATH, parse_dates=["date"])
    valuations = pd.read_csv(
        VALUATIONS_PATH,
        usecols=["player_id", "date", "market_value_in_eur"],
        parse_dates=["date"],
    )

    appearances = appearances.sort_values(["date", "player_id"])
    valuations = valuations.sort_values(["date", "player_id"])

    merged = pd.merge_asof(
        appearances,
        valuations.rename(columns={"date": "market_value_date"}),
        left_on="date",
        right_on="market_value_date",
        by="player_id",
        direction="backward",
    )

    merged.to_csv(OUTPUT_PATH, index=False)

    matched = merged["market_value_in_eur"].notna().sum()
    total = len(merged)
    print(f"Wrote {OUTPUT_PATH}")
    print(f"Rows: {total:,}")
    print(f"Rows with market value: {matched:,} ({matched / total:.1%})")
    print(f"Rows without prior market value: {total - matched:,}")


if __name__ == "__main__":
    main()

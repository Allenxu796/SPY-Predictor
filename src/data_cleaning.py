import argparse
import os
import sys
import pandas as pd


REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_csv(path)

    if "Date" in df.columns:
        date_col = "Date"
    else:
        date_col = df.columns[0]

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.set_index(date_col)

    df.index.name = "Date"
    return df


def validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_index()

    df = df[REQUIRED_COLUMNS].copy()

    for c in REQUIRED_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[~df.index.isna()]

    df = df.dropna()

    df = df[~df.index.duplicated(keep="last")]

    return df


def summarize(df_before: pd.DataFrame, df_after: pd.DataFrame) -> None:
    print("=== Cleaning Summary ===")
    print(f"Rows before: {len(df_before):,}")
    print(f"Rows after : {len(df_after):,}")
    print(f"Date range : {df_after.index.min().date()} -> {df_after.index.max().date()}")
    print("\nMissing values after cleaning (should be all 0):")
    print(df_after.isna().sum())
    print("\nHead:")
    print(df_after.head(3))
    print("\nTail:")
    print(df_after.tail(3))


def main():
    parser = argparse.ArgumentParser(description="Clean SPY/S&P500 daily OHLCV data.")
    parser.add_argument("--input", default="data/sp500_data.csv", help="Input CSV path")
    parser.add_argument("--output", default="data/sp500_data_clean.csv", help="Output CSV path")
    args = parser.parse_args()

    df_raw = load_csv(args.input)
    validate_columns(df_raw)

    df_clean = clean_data(df_raw)

    df_clean.reset_index().to_csv(args.output, index=False)

    summarize(df_raw, df_clean)
    print(f"\nSaved cleaned data to: {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

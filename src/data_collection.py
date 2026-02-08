import os
import pandas as pd
import yfinance as yf


def main():
    ticker = "SPY"
    start_date = "2000-01-01"

    print("Downloading SPY data...")
    df = yf.download(ticker, start=start_date, progress=False)

    if df.empty:
        raise RuntimeError("No data downloaded")

    df = df.reset_index()
    df = df.sort_values("Date")

    os.makedirs("data", exist_ok=True)
    output_path = "data/raw_spy.csv"
    df.to_csv(output_path, index=False)

    print(f"Saved to {output_path}")
    print(df.head())
    print(f"Rows: {len(df)}, From {df['Date'].min()} to {df['Date'].max()}")


if __name__ == "__main__":
    main()

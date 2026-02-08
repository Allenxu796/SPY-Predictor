from pathlib import Path
import pandas as pd


def make_target_close_to_close(df: pd.DataFrame) -> pd.DataFrame:

    cols = df.columns.tolist()

    if "Date" not in df.columns:
        first_col = cols[0]
        df = df.rename(columns={first_col: "Date"})

    if "Close" not in df.columns:
        raise ValueError(f"Cannot find Close column：{cols}")

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    next_close = df["Close"].shift(-1)

    df["Target"] = (next_close > df["Close"]).astype(int)
    df["Return_1d"] = (next_close / df["Close"]) - 1

    df = df.iloc[:-1].reset_index(drop=True)

    df = df.dropna().reset_index(drop=True)

    return df


def main():
    project_root = Path(__file__).resolve().parents[1]
    input_path = project_root / "data" / "sp500_data_clean.csv"
    output_path = project_root / "data" / "sp500_data_with_target.csv"

    if not input_path.exists():
        raise FileNotFoundError(f"Cannot find input file：{input_path}")

    df = pd.read_csv(input_path)
    df_out = make_target_close_to_close(df)

    df_out.to_csv(output_path, index=False)

    print("Step 3 Done.")
    print("Saved to:", output_path)
    print("Preview:")
    print(df_out[["Date", "Close", "Target", "Return_1d"]].head(10))


if __name__ == "__main__":
    main()

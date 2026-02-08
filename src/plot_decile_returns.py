from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def find_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Cannot find any of {candidates}. Existing: {df.columns.tolist()}")


def load_price_table(project_root: Path) -> pd.DataFrame:
    """
    Load a price table that contains Date and Close (or Adj Close).
    We try several files you likely have in /data.
    """
    candidates = [
        project_root / "data" / "sp500_data_with_target.csv",
        project_root / "data" / "sp500_data_clean.csv",
        project_root / "data" / "sp500_data.csv",
    ]

    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            date_col = find_col(df, ["Date", "date", "Datetime", "timestamp"])
            close_col = None
            for c in ["Close", "close", "Adj Close", "Adj_Close", "adj_close"]:
                if c in df.columns:
                    close_col = c
                    break
            if close_col is None:
                continue

            df = df[[date_col, close_col]].copy()
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(date_col).reset_index(drop=True)
            df = df.rename(columns={date_col: "Date", close_col: "Close"})
            return df

    raise FileNotFoundError(
        "Cannot find a usable price file with Date and Close. "
        "Tried: sp500_data_with_target.csv, sp500_data_clean.csv, sp500_data.csv"
    )


def main(num_bins: int = 10) -> None:
    project_root = Path(__file__).resolve().parents[1]

    pred_path = project_root / "results" / "walk_forward_predictions.csv"
    out_table_path = project_root / "results" / "decile_return_table.csv"

    if not pred_path.exists():
        raise FileNotFoundError(f"Missing file: {pred_path}")

    # 1) Load predictions (has Date, p_up, etc.)
    pred = pd.read_csv(pred_path)
    pred_date_col = find_col(pred, ["Date", "date", "Datetime", "timestamp"])
    p_col = find_col(pred, ["p_up", "P_up", "Prob_Up", "prob_up", "proba_1", "pred_proba", "pred_prob"])

    pred = pred.copy()
    pred[pred_date_col] = pd.to_datetime(pred[pred_date_col])
    pred = pred.sort_values(pred_date_col).reset_index(drop=True)
    pred = pred.rename(columns={pred_date_col: "Date"})

    # 2) Load prices (Date + Close)
    price = load_price_table(project_root)

    # 3) Merge predictions with prices on Date
    df = pred.merge(price, on="Date", how="left")
    missing_close = df["Close"].isna().mean()
    if missing_close > 0:
        print(f"Warning: {missing_close:.2%} of rows missing Close after merge. Dropping those rows.")
        df = df.dropna(subset=["Close"]).reset_index(drop=True)

    # 4) Realized next-day return
    df["ret_1d"] = df["Close"].shift(-1) / df["Close"] - 1.0
    df = df.iloc[:-1].copy()

    # 5) Create bins by p_up
    df["bin"] = pd.qcut(df[p_col].astype(float), q=num_bins, labels=False, duplicates="drop")

    # 6) Aggregate realized returns by bin
    table = (
        df.groupby("bin")
        .agg(
            count=("ret_1d", "size"),
            avg_ret=("ret_1d", "mean"),
            med_ret=("ret_1d", "median"),
            win_rate=("ret_1d", lambda x: (x > 0).mean()),
            avg_p=(p_col, "mean"),
        )
        .reset_index()
        .sort_values("bin")
    )

    table.to_csv(out_table_path, index=False)
    print(f"Saved decile table to: {out_table_path}")
    print(table)

    # 7) Plot
    plt.figure()
    plt.bar(table["bin"].astype(int), table["avg_ret"])
    plt.title("Decile Returns: Avg Next-day Return by Predicted p_up Bin")
    plt.xlabel("Bin (0=lowest p_up, higher=more bullish)")
    plt.ylabel("Average next-day return")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main(num_bins=10)

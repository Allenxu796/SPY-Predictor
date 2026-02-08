from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError("Missing columns. Need one of: {}. Existing: {}".format(candidates, df.columns.tolist()))


def load_price_table(project_root):
    candidates = [
        project_root / "data" / "sp500_data_with_target.csv",
        project_root / "data" / "sp500_data_clean.csv",
        project_root / "data" / "sp500_data.csv",
    ]

    for path in candidates:
        if not path.exists():
            continue

        df = pd.read_csv(path)
        date_col = find_col(df, ["Date", "date", "Datetime", "timestamp"])

        close_col = None
        for c in ["Close", "close", "Adj Close", "Adj_Close", "adj_close"]:
            if c in df.columns:
                close_col = c
                break
        if close_col is None:
            continue

        out = df[[date_col, close_col]].copy()
        out[date_col] = pd.to_datetime(out[date_col])
        out = out.sort_values(date_col).reset_index(drop=True)
        out = out.rename(columns={date_col: "Date", close_col: "Close"})
        return out

    raise FileNotFoundError("No usable price file found under data/ (need Date + Close).")


def spearman_ic(x, y):
    """Spearman rank correlation (IC)."""
    x = pd.Series(x).astype(float)
    y = pd.Series(y).astype(float)
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(df) < 30:
        return np.nan
    return float(df["x"].rank().corr(df["y"].rank()))


def max_drawdown(equity):
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def main(num_bins=10, top_k=1, bottom_k=1, fee_bps=0.0, rolling_window=252):
    """
    num_bins: number of quantile bins, default 10.
    top_k: how many top bins to long (e.g. 1 -> top decile, 2 -> top 2 deciles)
    bottom_k: how many bottom bins to short (e.g. 1 -> bottom decile)
    fee_bps: transaction cost in bps for daily long-short portfolio turnover (simple)
    rolling_window: rolling IC window, default ~1 trading year.
    """
    project_root = Path(__file__).resolve().parents[1]
    pred_path = project_root / "results" / "walk_forward_predictions.csv"
    daily_out = project_root / "results" / "decile_long_short_daily.csv"
    ic_year_out = project_root / "reports" / "ic_by_year.csv"
    port_year_out = project_root / "reports" / "decile_portfolio_by_year.csv"

    pred = pd.read_csv(pred_path)
    date_col = find_col(pred, ["Date", "date", "Datetime", "timestamp"])
    p_col = find_col(pred, ["p_up", "P_up", "Prob_Up", "prob_up", "proba_1", "pred_proba", "pred_prob"])

    pred = pred.copy()
    pred[date_col] = pd.to_datetime(pred[date_col])
    pred = pred.sort_values(date_col).reset_index(drop=True)
    pred = pred.rename(columns={date_col: "Date"})

    price = load_price_table(project_root)

    df = pred.merge(price, on="Date", how="left").dropna(subset=["Close"]).reset_index(drop=True)

    # realized next-day return
    df["ret_1d"] = df["Close"].shift(-1) / df["Close"] - 1.0
    df = df.iloc[:-1].copy()

    # -----------------------
    # Part A: IC (Spearman)
    # -----------------------
    df["Year"] = df["Date"].dt.year
    ic_all = spearman_ic(df[p_col], df["ret_1d"])

    ic_by_year = (
        df.groupby("Year")
        .apply(lambda g: spearman_ic(g[p_col], g["ret_1d"]))
        .reset_index()
        .rename(columns={0: "IC_spearman"})
    )

    ic_by_year.to_csv(ic_year_out, index=False)

    # Rolling IC
    rolling_ic = []
    p_vals = df[p_col].astype(float).values
    r_vals = df["ret_1d"].astype(float).values
    for i in range(len(df)):
        start = max(0, i - rolling_window + 1)
        ic = spearman_ic(p_vals[start:i+1], r_vals[start:i+1])
        rolling_ic.append(ic)
    df["IC_rolling"] = rolling_ic

    print("\n=== IC Summary ===")
    print("IC (all samples, Spearman):", round(ic_all, 4))
    print("Saved IC-by-year to:", ic_year_out)

    # Plot rolling IC
    plt.figure()
    plt.plot(df["Date"], df["IC_rolling"])
    plt.title("Rolling Spearman IC (p_up vs next-day return)")
    plt.xlabel("Date")
    plt.ylabel("IC")
    plt.tight_layout()
    plt.show()

    # ---------------------------------------
    # Part B: Decile portfolio (Long-Short)
    # ---------------------------------------
    # Bin by p_up each day across the whole sample
    df["bin"] = pd.qcut(df[p_col].astype(float), q=num_bins, labels=False, duplicates="drop")

    # Define top and bottom bins
    max_bin = int(df["bin"].max())
    top_bins = list(range(max_bin - top_k + 1, max_bin + 1))
    bottom_bins = list(range(0, bottom_k))

    df["long_w"] = df["bin"].isin(top_bins).astype(float)
    df["short_w"] = df["bin"].isin(bottom_bins).astype(float)

    # Normalize to 1 on each side when active
    long_sum = df["long_w"].sum()
    short_sum = df["short_w"].sum()
    if long_sum == 0 or short_sum == 0:
        raise ValueError("No rows in top or bottom bins. Try smaller num_bins or different top_k/bottom_k.")

    # Here we do a simple daily long-short with equal weight on signal days:
    # long_ret = ret_1d if in top bin else 0
    # short_ret = -ret_1d if in bottom bin else 0
    df["ls_gross_ret"] = df["long_w"] * df["ret_1d"] - df["short_w"] * df["ret_1d"]

    # Simple turnover cost approximation: position switches in/out of bins
    df["ls_position"] = df["long_w"] - df["short_w"]  # +1, 0, -1 conceptually
    df["turnover"] = df["ls_position"].diff().abs().fillna(0.0)
    fee = fee_bps / 10000.0
    df["ls_ret"] = df["ls_gross_ret"] - fee * df["turnover"]

    df["ls_equity"] = (1.0 + df["ls_ret"]).cumprod()

    df[["Date", "Year", p_col, "ret_1d", "bin", "ls_ret", "ls_equity"]].to_csv(daily_out, index=False)
    print("Saved daily long-short to:", daily_out)

    # Yearly portfolio stats
    port_year = (
        df.groupby("Year")
        .agg(
            days=("ls_ret", "size"),
            avg_daily=("ls_ret", "mean"),
            vol_daily=("ls_ret", "std"),
            total_return=("ls_equity", lambda x: float(x.iloc[-1] / x.iloc[0] - 1.0) if len(x) > 1 else np.nan),
        )
        .reset_index()
    )
    port_year.to_csv(port_year_out, index=False)
    print("Saved portfolio-by-year to:", port_year_out)

    # Plot long-short equity
    plt.figure()
    plt.plot(df["Date"], df["ls_equity"])
    plt.title("Top-Bin minus Bottom-Bin Long-Short Equity")
    plt.xlabel("Date")
    plt.ylabel("Equity (start=1.0)")
    plt.tight_layout()
    plt.show()

    # Print quick summary
    total_ls = df["ls_equity"].iloc[-1] - 1.0
    mdd_ls = max_drawdown(df["ls_equity"])
    print("\n=== Long-Short Summary ===")
    print("top_bins:", top_bins, "bottom_bins:", bottom_bins)
    print("fee_bps:", fee_bps)
    print("total_return:", round(float(total_ls), 4))
    print("max_drawdown:", round(float(mdd_ls), 4))


if __name__ == "__main__":
    main(
        num_bins=10,
        top_k=1,
        bottom_k=1,
        fee_bps=0.0,
        rolling_window=252
    )

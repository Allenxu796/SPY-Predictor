from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def find_col(df, candidates):
    """Return the first matching column name from candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError("Cannot find any of columns {}. Existing: {}".format(candidates, df.columns.tolist()))


def load_price_table(project_root):
    """
    Load a price table with Date and Close (or Adj Close) from /data.
    """
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

    raise FileNotFoundError("Cannot find any price file with Date and Close/Adj Close under data/.")


def max_drawdown(equity):
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def main(threshold_long=0.55, threshold_short=None, fee_bps=2.0, save_fig=False):
    # 1) Paths
    project_root = Path(__file__).resolve().parents[1]
    pred_path = project_root / "results" / "walk_forward_predictions.csv"
    out_csv = project_root / "results" / "backtest_results.csv"
    out_fig = project_root / "reports" / "fig_equity_curve.png"

    if not pred_path.exists():
        raise FileNotFoundError("Missing file: {}".format(pred_path))

    # 2) Load predictions
    pred = pd.read_csv(pred_path)
    date_col = find_col(pred, ["Date", "date", "Datetime", "timestamp"])
    p_col = find_col(pred, ["p_up", "P_up", "Prob_Up", "prob_up", "proba_1", "pred_proba", "pred_prob"])

    pred = pred.copy()
    pred[date_col] = pd.to_datetime(pred[date_col])
    pred = pred.sort_values(date_col).reset_index(drop=True)
    pred = pred.rename(columns={date_col: "Date"})

    # 3) Load prices
    price = load_price_table(project_root)

    # 4) Merge
    df = pred.merge(price, on="Date", how="left")
    if df["Close"].isna().any():
        missing = df["Close"].isna().mean()
        print("Warning: {:.2%} missing Close after merge. Dropping missing rows.".format(missing))
        df = df.dropna(subset=["Close"]).reset_index(drop=True)

    # 5) Realized next-day return
    df["ret_1d"] = df["Close"].shift(-1) / df["Close"] - 1.0
    df = df.iloc[:-1].copy()

    # 6) Trading rule -> position
    p = df[p_col].astype(float)
    if threshold_short is None:
        df["position"] = (p > threshold_long).astype(int)  # 1 or 0
    else:
        df["position"] = np.where(p > threshold_long, 1, np.where(p < threshold_short, -1, 0))

    # 7) Transaction cost
    df["turnover"] = df["position"].diff().abs().fillna(0.0)
    fee = fee_bps / 10000.0
    df["strategy_ret"] = df["position"] * df["ret_1d"] - fee * df["turnover"]

    # 8) Benchmark
    df["bh_ret"] = df["ret_1d"]

    # 9) Equity curves
    df["equity"] = (1.0 + df["strategy_ret"]).cumprod()
    df["bh_equity"] = (1.0 + df["bh_ret"]).cumprod()

    # 10) Save table
    df.to_csv(out_csv, index=False)
    print("Saved backtest table to: {}".format(out_csv))

    # 11) Metrics
    total_ret = df["equity"].iloc[-1] - 1.0
    bh_total_ret = df["bh_equity"].iloc[-1] - 1.0
    mdd = max_drawdown(df["equity"])
    bh_mdd = max_drawdown(df["bh_equity"])

    print("\n=== Strategy Summary ===")
    print("threshold_long: {}".format(threshold_long))
    print("fee_bps:        {}".format(fee_bps))
    print("total_return:   {:.4f}".format(total_ret))
    print("max_drawdown:   {:.4f}".format(mdd))
    print("trade_days:     {} / {}".format(int((df["position"] != 0).sum()), len(df)))
    print("exposure:       {:.2%}".format((df["position"] != 0).mean()))

    print("\n=== Buy & Hold Summary ===")
    print("total_return:   {:.4f}".format(bh_total_ret))
    print("max_drawdown:   {:.4f}".format(bh_mdd))

    # 12) Plot equity curves
    plt.figure()
    plt.plot(df["Date"], df["equity"], label="Strategy")
    plt.plot(df["Date"], df["bh_equity"], label="Buy & Hold")
    plt.title("Equity Curve (Strategy vs Buy & Hold)")
    plt.xlabel("Date")
    plt.ylabel("Equity (start = 1.0)")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main(threshold_long=0.55, threshold_short=None, fee_bps=2.0, save_fig=False)

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


def max_drawdown(equity):
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def backtest_threshold(df, p_col, thr, fee_bps):
    # position: 1 if p_up > thr else 0
    pos = (df[p_col].astype(float) > thr).astype(int)

    turnover = pos.diff().abs().fillna(0.0)
    fee = fee_bps / 10000.0

    strat_ret = pos * df["ret_1d"] - fee * turnover
    equity = (1.0 + strat_ret).cumprod()

    total_ret = float(equity.iloc[-1] - 1.0)
    mdd = max_drawdown(equity)
    exposure = float((pos != 0).mean())
    trade_days = int((pos != 0).sum())

    # annualized sharpe (simple)
    mu = float(strat_ret.mean())
    sd = float(strat_ret.std())
    sharpe = np.nan
    if sd > 0:
        sharpe = (mu / sd) * np.sqrt(252.0)

    return {
        "threshold": thr,
        "total_return": total_ret,
        "max_drawdown": mdd,
        "exposure": exposure,
        "trade_days": trade_days,
        "sharpe": float(sharpe) if sharpe == sharpe else np.nan,  # keep nan
    }


def main(fee_bps=2.0):
    project_root = Path(__file__).resolve().parents[1]
    pred_path = project_root / "results" / "walk_forward_predictions.csv"
    out_csv = project_root / "reports" / "threshold_sweep.csv"

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

    # thresholds to sweep
    thresholds = np.arange(0.50, 0.701, 0.01)  # 0.50 ... 0.70

    rows = []
    for thr in thresholds:
        rows.append(backtest_threshold(df, p_col, float(thr), fee_bps))

    res = pd.DataFrame(rows)
    res.to_csv(out_csv, index=False)
    print("Saved sweep table to:", out_csv)

    # print best thresholds by different criteria
    best_ret = res.sort_values("total_return", ascending=False).head(5)
    best_mdd = res.sort_values("max_drawdown", ascending=False).head(5)  # less negative is better
    best_sharpe = res.sort_values("sharpe", ascending=False).head(5)

    print("\nTop 5 by total_return:")
    print(best_ret[["threshold", "total_return", "max_drawdown", "exposure", "sharpe"]])

    print("\nTop 5 by max_drawdown (less negative is better):")
    print(best_mdd[["threshold", "total_return", "max_drawdown", "exposure", "sharpe"]])

    print("\nTop 5 by sharpe:")
    print(best_sharpe[["threshold", "total_return", "max_drawdown", "exposure", "sharpe"]])

    # plot: threshold vs metrics
    plt.figure()
    plt.plot(res["threshold"], res["total_return"], label="total_return")
    plt.plot(res["threshold"], res["max_drawdown"], label="max_drawdown")
    plt.plot(res["threshold"], res["exposure"], label="exposure")
    plt.title("Threshold Sweep: Performance vs Trading Threshold")
    plt.xlabel("threshold (p_up > threshold => long)")
    plt.ylabel("metric value")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main(fee_bps=2.0)

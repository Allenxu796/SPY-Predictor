from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BaselineFeatureConfig:
    ma_slope_lookback: int = 5
    obv_change_window: int = 10


BASELINE_FEATURES_15 = [
    # Momentum (3)
    "Ret_1d",
    "Ret_5d",
    "Ret_20d",
    # Trend (3)
    "Close_to_MA_5",
    "Close_to_MA_20",
    "MA_Slope_20",
    # Volatility (3)
    "Vol_5",
    "Vol_20",
    "ATR_14",
    # Volume/Flow (3)
    "Volume_to_VMA_20",
    "Volume_z20",
    "OBV_Change_10",
    # Candle/Microstructure (3)
    "Candle_Body",
    "HL_Range",
    "Gap",
]


def _ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols = df.columns.tolist()

    # Make sure we have a Date column
    if "Date" not in df.columns:
        df = df.rename(columns={cols[0]: "Date"})

    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}. Found: {cols}")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def add_baseline_features_15(
    df: pd.DataFrame,
    cfg: BaselineFeatureConfig = BaselineFeatureConfig(),
) -> pd.DataFrame:
    """
    Create 15 baseline features grouped by:
    Momentum (3), Trend (3), Volatility (3), Volume/Flow (3), Candle (3).

    All features are computed using information available at time t or earlier.
    """
    df = _ensure_ohlcv(df)

    # ---------- Momentum (3) ----------
    df["Ret_1d"] = df["Close"].pct_change(1)
    df["Ret_5d"] = df["Close"].pct_change(5)
    df["Ret_20d"] = df["Close"].pct_change(20)

    # ---------- Trend (3) ----------
    ma5 = df["Close"].rolling(5).mean()
    ma20 = df["Close"].rolling(20).mean()
    df["Close_to_MA_5"] = (df["Close"] / ma5) - 1.0
    df["Close_to_MA_20"] = (df["Close"] / ma20) - 1.0

    # MA slope proxy: change of MA20 over a lookback window
    df["MA_Slope_20"] = ma20.pct_change(cfg.ma_slope_lookback)

    # ---------- Volatility (3) ----------
    df["Vol_5"] = df["Ret_1d"].rolling(5).std()
    df["Vol_20"] = df["Ret_1d"].rolling(20).std()

    # True Range and ATR(14)
    prev_close = df["Close"].shift(1)
    tr = pd.concat(
        [
            (df["High"] - df["Low"]).abs(),
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["ATR_14"] = tr.rolling(14).mean()

    # ---------- Volume / Flow (3) ----------
    vma20 = df["Volume"].rolling(20).mean()
    df["Volume_to_VMA_20"] = (df["Volume"] / vma20) - 1.0

    v_mean = df["Volume"].rolling(20).mean()
    v_std = df["Volume"].rolling(20).std()
    df["Volume_z20"] = (df["Volume"] - v_mean) / (v_std + 1e-12)

    # OBV and OBV change over 10 days
    direction = np.sign(df["Close"].diff()).fillna(0.0)
    obv = (direction * df["Volume"]).cumsum()
    df["OBV_Change_10"] = obv.diff(cfg.obv_change_window)

    # ---------- Candle / Microstructure (3) ----------
    df["Candle_Body"] = (df["Close"] - df["Open"]) / df["Open"]
    df["HL_Range"] = (df["High"] / df["Low"]) - 1.0
    df["Gap"] = (df["Open"] / prev_close) - 1.0

    # Clean up infinities
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def get_feature_columns_15() -> list[str]:
    return list(BASELINE_FEATURES_15)


def drop_rows_with_feature_nans(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows with any NaN in the 15 features.
    This also implicitly removes early rows where rolling windows are not available.
    """
    feat_cols = get_feature_columns_15()
    return df.dropna(subset=feat_cols).reset_index(drop=True)

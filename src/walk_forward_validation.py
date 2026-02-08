from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
)


@dataclass
class FoldMetrics:
    test_year: int
    n_test: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: float
    logloss: float
    baseline_acc: float
    market_return: float


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if "Date" not in df.columns:
        raise ValueError("Dataset must contain a 'Date' column.")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    if "Target" not in df.columns:
        raise ValueError("Dataset must contain a 'Target' column.")

    return df


def detect_feature_columns(df: pd.DataFrame) -> List[str]:
    excluded = {"Date", "Target", "Return_1d"}
    feature_cols = [c for c in df.columns if c not in excluded]

    if len(feature_cols) == 0:
        raise ValueError("No feature columns found. Run feature engineering first.")

    return feature_cols


def build_model(random_state: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=600,
        max_depth=None,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=random_state,
    )


def walk_forward_yearly(
    df: pd.DataFrame,
    feature_cols: List[str],
    initial_train_years: int = 7,
    min_test_samples: int = 50,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Expanding-window walk-forward validation by calendar year.

    For each test year Y:
      - Train set: all rows with Year < Y
      - Test set : all rows with Year == Y

    initial_train_years controls the first test year:
      first_test_year = min_year + initial_train_years
    """
    df = df.copy()
    df["Year"] = df["Date"].dt.year

    years = sorted(df["Year"].unique())
    if len(years) <= initial_train_years + 1:
        raise ValueError("Not enough years in the dataset for walk-forward validation.")

    min_year = years[0]
    first_test_year = min_year + initial_train_years

    fold_rows: List[FoldMetrics] = []
    pred_frames: List[pd.DataFrame] = []

    for year in years:
        if year < first_test_year:
            continue

        train_df = df[df["Year"] < year]
        test_df = df[df["Year"] == year]

        if len(test_df) < min_test_samples or len(train_df) < 200:
            continue

        X_train = train_df[feature_cols].values
        y_train = train_df["Target"].astype(int).values

        X_test = test_df[feature_cols].values
        y_test = test_df["Target"].astype(int).values

        model = build_model()
        model.fit(X_train, y_train)

        p_up = model.predict_proba(X_test)[:, 1]
        y_pred = (p_up >= 0.5).astype(int)

        majority_class = int((y_train.mean() >= 0.5))
        baseline_pred = np.full_like(y_test, majority_class)
        baseline_acc = accuracy_score(y_test, baseline_pred)

        if "Close" in test_df.columns and len(test_df) >= 2:
            market_return = (test_df["Close"].iloc[-1] / test_df["Close"].iloc[0]) - 1.0
        else:
            market_return = np.nan

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        try:
            auc = roc_auc_score(y_test, p_up)
        except ValueError:
            auc = np.nan

        ll = log_loss(y_test, np.clip(p_up, 1e-6, 1 - 1e-6))

        fold_rows.append(
            FoldMetrics(
                test_year=int(year),
                n_test=int(len(test_df)),
                accuracy=float(acc),
                precision=float(prec),
                recall=float(rec),
                f1=float(f1),
                auc=float(auc),
                logloss=float(ll),
                baseline_acc=float(baseline_acc),
                market_return=float(market_return) if market_return == market_return else np.nan,
            )
        )

        fold_pred = test_df[["Date"]].copy()
        fold_pred["Year"] = int(year)
        fold_pred["y_true"] = y_test
        fold_pred["p_up"] = p_up
        fold_pred["y_pred"] = y_pred
        pred_frames.append(fold_pred)

    metrics_df = pd.DataFrame([m.__dict__ for m in fold_rows]).sort_values("test_year")
    preds_df = pd.concat(pred_frames, axis=0).reset_index(drop=True) if pred_frames else pd.DataFrame()

    return metrics_df, preds_df


def add_market_regime(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple yearly regime label based on market_return:
      > +10% => Bull
      < -10% => Bear
      else  => Sideways
    """
    out = metrics_df.copy()

    def label(ret: float) -> str:
        if pd.isna(ret):
            return "Unknown"
        if ret > 0.10:
            return "Bull"
        if ret < -0.10:
            return "Bear"
        return "Sideways"

    out["Regime"] = out["market_return"].apply(label)
    return out


def summarize_overall(metrics_df: pd.DataFrame) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    cols = ["accuracy", "auc", "logloss", "precision", "recall", "f1", "baseline_acc"]

    for c in cols:
        if c in metrics_df.columns and len(metrics_df) > 0:
            summary[f"{c}_mean"] = float(np.nanmean(metrics_df[c].values))
            summary[f"{c}_std"] = float(np.nanstd(metrics_df[c].values))

    return summary


def main() -> None:
    root = get_project_root()

    dataset_path = root / "data" / "processed" / "model_dataset.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Cannot find dataset at: {dataset_path}. "
            "Expected file: data/processed/model_dataset.csv. "
            "Update dataset_path if your file name/path is different."
        )


    df = load_dataset(dataset_path)
    feature_cols = detect_feature_columns(df)

    metrics_df, preds_df = walk_forward_yearly(
        df=df,
        feature_cols=feature_cols,
        initial_train_years=7,
        min_test_samples=50,
    )
    metrics_df = add_market_regime(metrics_df)

    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics_out = results_dir / "walk_forward_year_metrics.csv"
    preds_out = results_dir / "walk_forward_predictions.csv"

    metrics_df.to_csv(metrics_out, index=False)
    preds_df.to_csv(preds_out, index=False)

    print("=== Walk-forward validation completed ===")
    print(f"Saved yearly metrics: {metrics_out}")
    print(f"Saved daily predictions: {preds_out}")

    if len(metrics_df) == 0:
        print("No folds were produced. Check your data coverage and parameters.")
        return

    summary = summarize_overall(metrics_df)
    print("\n=== Overall summary (mean ± std across years) ===")
    for k, v in summary.items():
        print(f"{k}: {v:.4f}")

    print("\n=== Worst 5 years by accuracy ===")
    print(
        metrics_df.sort_values("accuracy")
        .head(5)[["test_year", "Regime", "accuracy", "auc", "logloss", "baseline_acc", "market_return"]]
    )

    print("\n=== Best 5 years by accuracy ===")
    print(
        metrics_df.sort_values("accuracy", ascending=False)
        .head(5)[["test_year", "Regime", "accuracy", "auc", "logloss", "baseline_acc", "market_return"]]
    )

    print("\n=== Average metrics by regime ===")
    print(metrics_df.groupby("Regime")[["accuracy", "auc", "logloss"]].mean(numeric_only=True))


if __name__ == "__main__":
    main()

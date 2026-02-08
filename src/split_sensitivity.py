from __future__ import annotations

import argparse
from pathlib import Path
from dataclasses import dataclass
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


@dataclass
class SplitResult:
    split_name: str
    test_years: int
    test_start: str
    test_end: str
    n_train: int
    n_test: int
    accuracy: float
    f1: float
    auc: float


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "Date" not in df.columns:
        raise ValueError("Missing required column: Date")
    if "Target" not in df.columns:
        raise ValueError("Missing required column: Target")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def infer_feature_cols(df: pd.DataFrame) -> list[str]:
    # Exclude label and obvious non-features
    exclude = {"Date", "Target", "Return_1d"}
    cols = [c for c in df.columns if c not in exclude]

    # Keep only numeric columns
    num_cols = []
    for c in cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            num_cols.append(c)

    if len(num_cols) == 0:
        raise ValueError("No numeric feature columns found after exclusions.")
    return num_cols


def time_split_last_n_years(df: pd.DataFrame, n_years: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    last_date = df["Date"].iloc[-1]
    test_start = last_date - pd.DateOffset(years=n_years)

    train_df = df[df["Date"] < test_start].copy()
    test_df = df[df["Date"] >= test_start].copy()

    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError(f"Split produced empty train or test. n_years={n_years}, test_start={test_start}")

    return train_df, test_df, test_start


def build_model() -> Pipeline:
    # Logistic regression as a clean baseline (probability output for AUC)
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            solver="lbfgs",
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        ))
    ])


def eval_one_split(df: pd.DataFrame, feature_cols: list[str], n_years: int, split_name: str) -> SplitResult:
    train_df, test_df, test_start = time_split_last_n_years(df, n_years=n_years)

    X_train = train_df[feature_cols]
    y_train = train_df["Target"].astype(int)

    X_test = test_df[feature_cols]
    y_test = test_df["Target"].astype(int)

    model = build_model()
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    acc = float(accuracy_score(y_test, pred))
    f1 = float(f1_score(y_test, pred))
    # AUC requires both classes present in y_test
    try:
        auc = float(roc_auc_score(y_test, proba))
    except ValueError:
        auc = float("nan")

    return SplitResult(
        split_name=split_name,
        test_years=n_years,
        test_start=str(pd.to_datetime(test_start).date()),
        test_end=str(pd.to_datetime(df["Date"].iloc[-1]).date()),
        n_train=int(len(train_df)),
        n_test=int(len(test_df)),
        accuracy=acc,
        f1=f1,
        auc=auc
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the feature-engineered CSV (must include Date and Target)."
    )
    parser.add_argument(
        "--out",
        type=str,
        default="reports/split_sensitivity_results.csv",
        help="Where to save the results CSV."
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_data(input_path)
    feature_cols = infer_feature_cols(df)

    splits = [
        ("A_main_last_5y", 5),
        ("B_last_3y", 3),
        ("C_last_7y", 7),
    ]

    results: list[SplitResult] = []
    for name, yrs in splits:
        r = eval_one_split(df, feature_cols, n_years=yrs, split_name=name)
        results.append(r)

    res_df = pd.DataFrame([r.__dict__ for r in results])
    res_df = res_df.sort_values("test_years").reset_index(drop=True)

    print("\n=== Split Sensitivity Results ===")
    print(res_df.to_string(index=False))

    res_df.to_csv(out_path, index=False)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()

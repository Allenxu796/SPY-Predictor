from __future__ import annotations

from pathlib import Path
import pandas as pd

from feature_engineering import (
    add_baseline_features_15,
    drop_rows_with_feature_nans,
    get_feature_columns_15,
)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    # Adjust this path to your Step 3 output file
    # Example: data/processed/with_target.csv
    input_path = project_root / "data" / "sp500_data_with_target.csv"
    if not input_path.exists():
        raise FileNotFoundError(
            f"Cannot find input file: {input_path}. "
            f"Please export your Step 3 output to this path."
        )

    df = pd.read_csv(input_path)

    df = add_baseline_features_15(df)
    df = drop_rows_with_feature_nans(df)

    # Keep columns needed for training and regime diagnostics
    # Include Close so walk-forward can compute yearly market_return for Regime labels.
    keep_cols = ["Date", "Target", "Close"] + get_feature_columns_15()
    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns after feature engineering: {missing}")

    out = df[keep_cols].copy()

    out_dir = project_root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    output_path = out_dir / "features.csv"
    out.to_csv(output_path, index=False)

    print("Saved:", output_path)
    print("Rows, Cols:", out.shape)
    print("Feature count:", len(get_feature_columns_15()))
    print(out.tail(3).T)


if __name__ == "__main__":
    main()

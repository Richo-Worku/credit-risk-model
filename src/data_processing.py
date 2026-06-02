# src/data_processing.py

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.preprocessing import StandardScaler


# =========================
# 1. LOAD DATA
# =========================
def load_data(filepath):
    """
    Load raw transaction data.
    """
    return pd.read_csv(filepath)


# =========================
# 2. TIME FEATURES
# =========================
def extract_time_features(df):
    """
    Extract time-based features from TransactionStartTime.
    """

    df = df.copy()

    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])

    df["TransactionHour"] = df["TransactionStartTime"].dt.hour
    df["TransactionDay"] = df["TransactionStartTime"].dt.day
    df["TransactionMonth"] = df["TransactionStartTime"].dt.month
    df["TransactionYear"] = df["TransactionStartTime"].dt.year

    return df


# =========================
# 3. AGGREGATE FEATURES
# =========================
def create_aggregate_features(df):
    """
    Create customer-level aggregate features.
    """

    customer_features = (
        df.groupby("CustomerId")
          .agg(
              TotalTransactionAmount=("Amount", "sum"),
              AverageTransactionAmount=("Amount", "mean"),
              TransactionCount=("TransactionId", "count"),
              StdTransactionAmount=("Amount", lambda x: x.std(ddof=0))
          )
          .reset_index()
    )

    # -------------------------
    # HANDLE MISSING VALUES
    # -------------------------
    customer_features["StdTransactionAmount"] = customer_features["StdTransactionAmount"].fillna(0)

    return customer_features


# =========================
# 4. MAIN PIPELINE
# =========================
def main():

    # Path handling
    base_path = Path(__file__).resolve().parent.parent
    data_path = base_path / "data" / "raw" / "data.csv"
    output_path = base_path / "data" / "processed" / "processed_data.csv"

    # Load data
    df = load_data(data_path)

    print("Raw data shape:", df.shape)

    # Feature engineering
    df = extract_time_features(df)

    # Aggregate to customer level
    customer_features = create_aggregate_features(df)

    print("Customer-level data shape:", customer_features.shape)

    # =========================
    # SCALING (STANDARDIZATION)
    # =========================
    numeric_features = [
        "TotalTransactionAmount",
        "AverageTransactionAmount",
        "TransactionCount",
        "StdTransactionAmount"
    ]

    scaler = StandardScaler()

    customer_features[numeric_features] = scaler.fit_transform(
        customer_features[numeric_features]
    )

    print("\nScaled features preview:")
    print(customer_features.head())

    # Save processed dataset
    output_path.parent.mkdir(parents=True, exist_ok=True)
    customer_features.to_csv(output_path, index=False)

    print(f"\nProcessed data saved to: {output_path}")
    print("Pipeline executed successfully ✔")


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    main()
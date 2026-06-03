import pandas as pd
import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.data_processing import (
    extract_time_features,
    create_aggregate_features
)


def test_extract_time_features_creates_expected_columns():

    df = pd.DataFrame({
        "TransactionStartTime": ["2025-01-15 10:30:00"]
    })

    result = extract_time_features(df)

    expected_columns = [
        "TransactionHour",
        "TransactionDay",
        "TransactionMonth",
        "TransactionYear"
    ]

    for col in expected_columns:
        assert col in result.columns


def test_create_aggregate_features_returns_expected_columns():

    df = pd.DataFrame({
        "CustomerId": ["C1", "C1", "C2"],
        "TransactionId": ["T1", "T2", "T3"],
        "Amount": [100, 200, 300]
    })

    result = create_aggregate_features(df)

    expected_columns = [
        "CustomerId",
        "TotalTransactionAmount",
        "AverageTransactionAmount",
        "TransactionCount",
        "StdTransactionAmount"
    ]

    for col in expected_columns:
        assert col in result.columns
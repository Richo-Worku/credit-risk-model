import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# =========================
# 1. LOAD DATA
# =========================
def load_data(filepath):
    return pd.read_csv(filepath)


# =========================
# 2. TIME FEATURES
# =========================
def extract_time_features(df):

    df = df.copy()
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])

    df["TransactionHour"] = df["TransactionStartTime"].dt.hour
    df["TransactionDay"] = df["TransactionStartTime"].dt.day
    df["TransactionMonth"] = df["TransactionStartTime"].dt.month
    df["TransactionYear"] = df["TransactionStartTime"].dt.year

    return df


# =========================
# 3. TASK 3: AGGREGATE FEATURES
# =========================
def create_aggregate_features(df):

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

    customer_features["StdTransactionAmount"] = (
        customer_features["StdTransactionAmount"].fillna(0)
    )

    return customer_features


# =========================
# 4. TASK 4: RFM FEATURES
# =========================
def create_rfm_features(df):

    df = df.copy()
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])

    snapshot_date = df["TransactionStartTime"].max()

    rfm = (
        df.groupby("CustomerId")
        .agg(
            Recency=("TransactionStartTime", lambda x: (snapshot_date - x.max()).days),
            Frequency=("TransactionId", "count"),
            Monetary=("Amount", "sum")
        )
        .reset_index()
    )

    return rfm


# =========================
# 5. KMEANS CLUSTERING
# =========================
def create_rfm_clusters(rfm_df):

    rfm = rfm_df.copy()

    features = ["Recency", "Frequency", "Monetary"]

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[features])

    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

    return rfm


# =========================
# 6. PROXY TARGET LABEL
# =========================
def assign_risk_label(rfm_clustered):

    df = rfm_clustered.copy()

    # Based on cluster analysis (least engaged customers)
    high_risk_cluster = 0

    df["is_high_risk"] = (
        df["Cluster"] == high_risk_cluster
    ).astype(int)

    return df


# =========================
# 7. MAIN PIPELINE
# =========================
def main():

    # Paths
    base_path = Path(__file__).resolve().parent.parent

    data_path = base_path / "data" / "raw" / "data.csv"

    task3_output = base_path / "data" / "processed" / "customer_features.csv"
    task4_output = base_path / "data" / "processed" / "rfm_clustered.csv"
    final_output = base_path / "data" / "processed" / "training_data.csv"

    # Load data
    df = load_data(data_path)

    print("Raw data shape:", df.shape)

    # Time features
    df = extract_time_features(df)

    # =========================
    # TASK 3
    # =========================
    customer_features = create_aggregate_features(df)

    print("\nTask 3 Features Preview:")
    print(customer_features.head())

    task3_output.parent.mkdir(parents=True, exist_ok=True)

    customer_features.to_csv(task3_output, index=False)
    print(f"\nSaved Task 3 → {task3_output}")

    # =========================
    # TASK 4
    # =========================
    rfm_features = create_rfm_features(df)

    print("\nRFM Features Preview:")
    print(rfm_features.head())

    rfm_clustered = create_rfm_clusters(rfm_features)

    print("\nCluster Distribution:")
    print(rfm_clustered["Cluster"].value_counts())

    print("\nCluster Profiles:")
    print(
        rfm_clustered.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()
    )

    # Assign risk label
    rfm_clustered = assign_risk_label(rfm_clustered)

    print("\nRisk Label Distribution:")
    print(rfm_clustered["is_high_risk"].value_counts())

    # =========================
    # FINAL MERGE (MODEL DATASET)
    # =========================
    final_dataset = customer_features.merge(
        rfm_clustered,
        on="CustomerId",
        how="inner"
    )

    print("\nFinal Dataset Shape:", final_dataset.shape)
    print("\nFinal Dataset Preview:")
    print(final_dataset.head())

    # Save outputs
    rfm_clustered.to_csv(task4_output, index=False)
    final_dataset.to_csv(final_output, index=False)

    print(f"\nSaved Task 4 → {task4_output}")
    print(f"Saved FINAL Training Data → {final_output}")

    print("\nPipeline executed successfully ✔")


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    main()
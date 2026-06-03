import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report, roc_auc_score


# =========================
# LOAD DATA
# =========================
def load_data(filepath):
    return pd.read_csv(filepath)


# =========================
# MODEL EVALUATION FUNCTION
# =========================
def evaluate_model(model, X_train, X_test, y_train, y_test, name):

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Some models may not always support predict_proba safely
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    else:
        auc = None

    print("\n==============================")
    print(f"MODEL: {name}")
    print("==============================")

    print(classification_report(y_test, y_pred))

    if auc is not None:
        print("ROC-AUC:", auc)


# =========================
# MAIN
# =========================
def main():

    base_path = Path(__file__).resolve().parent.parent
    data_path = base_path / "data" / "processed" / "training_data.csv"

    df = load_data(data_path)

    print("Dataset shape:", df.shape)

    # =========================
    # FEATURES & TARGET
    # =========================
    X = df.drop(columns=["is_high_risk", "CustomerId", "Cluster"])
    y = df["is_high_risk"]

    # =========================
    # TRAIN / TEST SPLIT
    # =========================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # =========================
    # SCALING (for Logistic Regression)
    # =========================
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # =========================
    # MODELS
    # =========================
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42)
    }

    # =========================
    # TRAIN & EVALUATE
    # =========================
    for name, model in models.items():
        evaluate_model(model, X_train, X_test, y_train, y_test, name)


if __name__ == "__main__":
    main()
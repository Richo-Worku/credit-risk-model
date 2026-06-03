import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# =========================
# LOAD DATA
# =========================
def load_data(filepath):
    return pd.read_csv(filepath)


# =========================
# MAIN
# =========================
def main():

    base_path = Path(__file__).resolve().parent.parent
    data_path = base_path / "data" / "processed" / "training_data.csv"

    df = load_data(data_path)

    # FEATURES / TARGET
    X = df.drop(columns=["is_high_risk", "CustomerId", "Cluster"])
    y = df["is_high_risk"]

    # TRAIN TEST SPLIT
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # SCALING
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # =========================
    # MODEL
    # =========================
    model = LogisticRegression(max_iter=1000)

    # =========================
    # 1. GRID SEARCH
    # =========================
    grid_params = {
        "C": [0.01, 0.1, 1, 10],
        "penalty": ["l2"],
        "solver": ["lbfgs"]
    }

    grid_search = GridSearchCV(
        model,
        grid_params,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    print("\n================ GRID SEARCH ================")
    print("Best Params:", grid_search.best_params_)
    print("Best CV Score:", grid_search.best_score_)

    best_grid_model = grid_search.best_estimator_

    y_pred_prob = best_grid_model.predict_proba(X_test)[:, 1]
    print("Test ROC-AUC:", roc_auc_score(y_test, y_pred_prob))


    # =========================
    # 2. RANDOM SEARCH
    # =========================
    from scipy.stats import uniform

    random_params = {
        "C": uniform(0.01, 10),
        "penalty": ["l2"],
        "solver": ["lbfgs"]
    }

    random_search = RandomizedSearchCV(
        model,
        param_distributions=random_params,
        n_iter=10,
        scoring="roc_auc",
        cv=5,
        random_state=42,
        n_jobs=-1
    )

    random_search.fit(X_train, y_train)

    print("\n================ RANDOM SEARCH ================")
    print("Best Params:", random_search.best_params_)
    print("Best CV Score:", random_search.best_score_)

    best_random_model = random_search.best_estimator_

    y_pred_prob = best_random_model.predict_proba(X_test)[:, 1]
    print("Test ROC-AUC:", roc_auc_score(y_test, y_pred_prob))


if __name__ == "__main__":
    main()
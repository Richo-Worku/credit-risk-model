import pandas as pd
from pathlib import Path

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, roc_auc_score


# Load data
base_path = Path(__file__).resolve().parent.parent
data_path = base_path / "data" / "processed" / "training_data.csv"

df = pd.read_csv(data_path)

X = df.drop(columns=["is_high_risk", "CustomerId", "Cluster"])
y = df["is_high_risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = {
    "LogisticRegression": LogisticRegression(
        max_iter=1000,
        C=0.5908361216819946
    ),
    "DecisionTree": DecisionTreeClassifier(
        random_state=42
    )
}

mlflow.set_experiment("Credit Risk Modeling")

for model_name, model in models.items():

    with mlflow.start_run(run_name=model_name):

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)

        # Parameters
        mlflow.log_param("model_type", model_name)

        # Metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("roc_auc", roc_auc)

        # Model artifact
        mlflow.sklearn.log_model(
            model,
            artifact_path="model"
        )

        print(f"{model_name}")
        print("Accuracy:", accuracy)
        print("ROC-AUC:", roc_auc)
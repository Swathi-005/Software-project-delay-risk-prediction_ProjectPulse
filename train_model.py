import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# -----------------------------
# LOAD DATA
# -----------------------------

data = pd.read_csv("data/project_data.csv")


FEATURES = [
    "project_type",
    "team_size",
    "experience_level",
    "duration_months",
    "planned_tasks",
    "completed_tasks",
    "pending_tasks",
    "requirement_changes",
    "critical_bugs",
    "minor_bugs",
    "budget_used_percent",
    "schedule_progress_percent",
    "developer_workload",
    "client_change_requests",
    "testing_completion_percent",
    "previous_delays"
]


TARGET = "delay_risk"

X = data[FEATURES]
y = data[TARGET]


categorical_features = [
    "project_type",
    "experience_level"
]


numeric_features = [
    column
    for column in FEATURES
    if column not in categorical_features
]


# -----------------------------
# PREPROCESSING
# -----------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        ),
        (
            "numeric",
            StandardScaler(),
            numeric_features
        )
    ]
)


# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# -----------------------------
# MODELS
# -----------------------------

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=1500,
            random_state=42
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            max_depth=10,
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=250,
            max_depth=15,
            random_state=42
        ),

    "Gradient Boosting":
        GradientBoostingClassifier(
            n_estimators=150,
            random_state=42
        )
}


results = []

best_model = None
best_model_name = None
best_score = -1


# -----------------------------
# TRAIN MODELS
# -----------------------------

for name, algorithm in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", algorithm)
        ]
    )

    pipeline.fit(
        X_train,
        y_train
    )

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1 Score": round(f1, 4)
    })

    print(
        f"{name}: "
        f"Accuracy={accuracy:.4f}, "
        f"F1={f1:.4f}"
    )

    # Select using F1 rather than accuracy alone
    if f1 > best_score:

        best_score = f1
        best_model = pipeline
        best_model_name = name


# -----------------------------
# SAVE RESULTS
# -----------------------------

os.makedirs(
    "models",
    exist_ok=True
)

results_df = pd.DataFrame(results)

results_df.to_csv(
    "models/model_results.csv",
    index=False
)

joblib.dump(
    best_model,
    "models/projectpulse_model.pkl"
)


with open(
    "models/best_model.txt",
    "w"
) as file:

    file.write(best_model_name)


print("\n-----------------------------")
print("BEST MODEL")
print("-----------------------------")

print(best_model_name)

print(
    f"Weighted F1 Score: "
    f"{best_score:.4f}"
)

print(
    "\nModel saved to "
    "models/projectpulse_model.pkl"
)
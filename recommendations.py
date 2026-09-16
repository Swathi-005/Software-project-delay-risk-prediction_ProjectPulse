import joblib
import pandas as pd


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


def load_model():

    return joblib.load(
        "models/projectpulse_model.pkl"
    )


def predict_project(model, project):

    input_df = pd.DataFrame(
        [project],
        columns=FEATURES
    )

    prediction = model.predict(
        input_df
    )[0]

    probabilities = model.predict_proba(
        input_df
    )[0]

    classes = model.classes_

    probability_dictionary = dict(
        zip(classes, probabilities)
    )

    confidence = (
        probability_dictionary[prediction]
        * 100
    )

    return (
        prediction,
        confidence,
        probability_dictionary
    )
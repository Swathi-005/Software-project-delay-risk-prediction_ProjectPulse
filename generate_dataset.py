import os
import numpy as np
import pandas as pd

np.random.seed(42)

NUMBER_OF_PROJECTS = 1500

project_types = [
    "Web Application",
    "Mobile Application",
    "Data Analytics",
    "Machine Learning",
    "Enterprise Software"
]

experience_levels = ["Junior", "Mixed", "Senior"]

records = []

for i in range(NUMBER_OF_PROJECTS):

    project_type = np.random.choice(project_types)

    team_size = np.random.randint(3, 21)

    experience_level = np.random.choice(
        experience_levels,
        p=[0.25, 0.50, 0.25]
    )

    duration_months = np.random.randint(2, 25)

    planned_tasks = np.random.randint(40, 301)

    completion_ratio = np.random.uniform(0.20, 1.0)

    completed_tasks = int(planned_tasks * completion_ratio)

    pending_tasks = planned_tasks - completed_tasks

    requirement_changes = np.random.randint(0, 16)

    critical_bugs = np.random.randint(0, 13)

    minor_bugs = np.random.randint(0, 31)

    budget_used_percent = np.random.randint(35, 121)

    schedule_progress_percent = np.random.randint(20, 111)

    developer_workload = np.random.randint(40, 121)

    client_change_requests = np.random.randint(0, 13)

    testing_completion_percent = np.random.randint(10, 101)

    previous_delays = np.random.randint(0, 6)

    # -----------------------------
    # Synthetic risk generation
    # -----------------------------

    risk_score = 0

    pending_percent = (pending_tasks / planned_tasks) * 100

    if pending_percent > 55:
        risk_score += 3
    elif pending_percent > 35:
        risk_score += 2
    elif pending_percent > 20:
        risk_score += 1

    if requirement_changes >= 10:
        risk_score += 3
    elif requirement_changes >= 6:
        risk_score += 2
    elif requirement_changes >= 3:
        risk_score += 1

    if critical_bugs >= 8:
        risk_score += 3
    elif critical_bugs >= 4:
        risk_score += 2
    elif critical_bugs >= 2:
        risk_score += 1

    if budget_used_percent > 105:
        risk_score += 3
    elif budget_used_percent > 90:
        risk_score += 2
    elif budget_used_percent > 75:
        risk_score += 1

    if developer_workload > 105:
        risk_score += 3
    elif developer_workload > 85:
        risk_score += 2
    elif developer_workload > 70:
        risk_score += 1

    if client_change_requests >= 8:
        risk_score += 2
    elif client_change_requests >= 4:
        risk_score += 1

    if testing_completion_percent < 35:
        risk_score += 3
    elif testing_completion_percent < 60:
        risk_score += 2
    elif testing_completion_percent < 80:
        risk_score += 1

    if previous_delays >= 4:
        risk_score += 2
    elif previous_delays >= 2:
        risk_score += 1

    if experience_level == "Junior":
        risk_score += 1
    elif experience_level == "Senior":
        risk_score -= 1

    # Add some randomness so model does not simply
    # reproduce perfectly deterministic rules.
    risk_score += np.random.choice(
        [-2, -1, 0, 0, 0, 1, 2]
    )

    if risk_score <= 7:
        delay_risk = "Low"
    elif risk_score <= 14:
        delay_risk = "Medium"
    else:
        delay_risk = "High"

    records.append([
        f"Project-{i + 1}",
        project_type,
        team_size,
        experience_level,
        duration_months,
        planned_tasks,
        completed_tasks,
        pending_tasks,
        requirement_changes,
        critical_bugs,
        minor_bugs,
        budget_used_percent,
        schedule_progress_percent,
        developer_workload,
        client_change_requests,
        testing_completion_percent,
        previous_delays,
        delay_risk
    ])


columns = [
    "project_name",
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
    "previous_delays",
    "delay_risk"
]

df = pd.DataFrame(records, columns=columns)

os.makedirs("data", exist_ok=True)

df.to_csv(
    "data/project_data.csv",
    index=False
)

print("Dataset generated successfully!")
print(f"Total projects: {len(df)}")

print("\nRisk distribution:")
print(df["delay_risk"].value_counts())
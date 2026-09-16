import streamlit as st
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="ProjectPulse",
    page_icon="📊",
    layout="wide"
)


# ---------------- LOAD DATA ----------------

@st.cache_data
def load_data():
    return pd.read_csv("project_data.csv")


data = load_data()

FEATURES = [
    "team_size",
    "duration_months",
    "completed_tasks",
    "pending_tasks",
    "requirement_changes",
    "critical_bugs",
    "budget_used_percent"
]

X = data[FEATURES]
y = data["delay_risk"]


# ---------------- TRAIN MODEL ----------------

@st.cache_resource
def train_model(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=150,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    return model, accuracy


model, accuracy = train_model(X, y)


# ---------------- HEADER ----------------

st.title("📊 ProjectPulse")

st.subheader("ML-Based Software Project Delay Risk Predictor")

st.write(
    """
    ProjectPulse analyzes software project indicators and predicts
    whether a project has a **Low, Medium, or High risk of delay**.
    """
)

st.divider()


# ---------------- SIDEBAR ----------------

st.sidebar.header("Project Details")

team_size = st.sidebar.slider(
    "Team Size",
    1, 20, 5
)

duration = st.sidebar.slider(
    "Project Duration (Months)",
    1, 24, 6
)

completed = st.sidebar.slider(
    "Completed Tasks (%)",
    0, 100, 70
)

pending = st.sidebar.slider(
    "Pending Tasks (%)",
    0, 100, 30
)

requirement_changes = st.sidebar.slider(
    "Requirement Changes",
    0, 15, 3
)

critical_bugs = st.sidebar.slider(
    "Critical Bugs",
    0, 15, 2
)

budget_used = st.sidebar.slider(
    "Budget Used (%)",
    0, 120, 75
)


# ---------------- DASHBOARD METRICS ----------------

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Team Size",
    team_size
)

col2.metric(
    "Completed",
    f"{completed}%"
)

col3.metric(
    "Pending",
    f"{pending}%"
)

col4.metric(
    "Budget Used",
    f"{budget_used}%"
)


st.divider()


# ---------------- PREDICTION ----------------

if st.button(
    "🔍 Analyze Project Risk",
    type="primary",
    use_container_width=True
):

    input_data = pd.DataFrame(
        [[
            team_size,
            duration,
            completed,
            pending,
            requirement_changes,
            critical_bugs,
            budget_used
        ]],
        columns=FEATURES
    )

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]

    probability_dict = dict(
        zip(model.classes_, probabilities)
    )

    confidence = probability_dict[prediction] * 100


    st.subheader("Risk Analysis Result")


    # ---------------- RESULT ----------------

    if prediction == "Low":

        st.success(
            f"🟢 LOW DELAY RISK — {confidence:.1f}% confidence"
        )

    elif prediction == "Medium":

        st.warning(
            f"🟠 MEDIUM DELAY RISK — {confidence:.1f}% confidence"
        )

    else:

        st.error(
            f"🔴 HIGH DELAY RISK — {confidence:.1f}% confidence"
        )


    # ---------------- PROBABILITIES ----------------

    st.subheader("Prediction Probability")

    probability_df = pd.DataFrame({
        "Risk Level": model.classes_,
        "Probability": probabilities
    })

    st.bar_chart(
        probability_df.set_index("Risk Level")
    )


    # ---------------- RISK FACTORS ----------------

    st.subheader("📌 Project Observations")

    observations = []

    if pending >= 50:
        observations.append(
            "⚠️ Large percentage of tasks are still pending."
        )

    if requirement_changes >= 6:
        observations.append(
            "⚠️ Frequent requirement changes may affect the schedule."
        )

    if critical_bugs >= 5:
        observations.append(
            "⚠️ High number of critical bugs requires attention."
        )

    if budget_used >= 90:
        observations.append(
            "⚠️ Project has consumed most of its allocated budget."
        )

    if completed >= 80:
        observations.append(
            "✅ A large portion of planned tasks has been completed."
        )

    if critical_bugs <= 1:
        observations.append(
            "✅ Critical bug count is currently low."
        )

    if len(observations) == 0:
        observations.append(
            "Project indicators are currently within moderate ranges."
        )

    for observation in observations:
        st.write(observation)


    # ---------------- RECOMMENDATIONS ----------------

    st.subheader("💡 Recommendations")

    if prediction == "High":

        st.write(
            """
            - Prioritize critical bugs.
            - Reduce unnecessary requirement changes.
            - Review pending tasks with the development team.
            - Reallocate resources to high-priority modules.
            - Review project budget and schedule immediately.
            """
        )

    elif prediction == "Medium":

        st.write(
            """
            - Monitor pending tasks closely.
            - Review requirement changes before approval.
            - Resolve critical bugs early.
            - Conduct regular project progress reviews.
            """
        )

    else:

        st.write(
            """
            - Continue monitoring project milestones.
            - Maintain the current development pace.
            - Prevent unnecessary scope changes.
            - Continue regular testing and bug tracking.
            """
        )


# ---------------- MODEL INFORMATION ----------------

st.divider()

with st.expander("🤖 Machine Learning Model Information"):

    st.write("**Algorithm:** Random Forest Classifier")

    st.write(
        f"**Demo test accuracy:** {accuracy * 100:.1f}%"
    )

    st.write(
        "**Prediction classes:** Low, Medium, High"
    )

    st.write(
        """
        The model analyzes software project indicators such as
        task completion, pending work, requirement changes,
        critical bugs and budget utilization.
        """
    )

    st.caption(
        "This prototype uses synthetic demonstration data. "
        "Accuracy on this small dataset should not be interpreted "
        "as real-world performance."
    )


# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "ProjectPulse • ML-Based Software Project Risk Analytics"
)
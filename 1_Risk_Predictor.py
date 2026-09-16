import os
import streamlit as st
import pandas as pd
import joblib


st.set_page_config(
    page_title="Risk Predictor",
    page_icon="🔮",
    layout="wide"
)

st.title("🔮 Software Project Risk Predictor")

st.write(
    "Enter the current project information to predict "
    "whether the project has Low, Medium, or High delay risk."
)

st.divider()


# ==========================================
# LOAD MODEL
# ==========================================

MODEL_PATH = "models/projectpulse_model.pkl"


if not os.path.exists(MODEL_PATH):

    st.error(
        "❌ Trained model not found."
    )

    st.write(
        "Run the following command first:"
    )

    st.code("python train_model.py")

    st.stop()


@st.cache_resource
def load_trained_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_trained_model()

except Exception as e:

    st.error("Unable to load the trained model.")

    st.exception(e)

    st.stop()


# ==========================================
# PROJECT DETAILS
# ==========================================

st.subheader("📋 Project Information")

project_name = st.text_input(
    "Project Name",
    value="Project Alpha"
)


col1, col2, col3 = st.columns(3)


# ==========================================
# COLUMN 1
# ==========================================

with col1:

    st.markdown("### Basic Details")

    project_type = st.selectbox(
        "Project Type",
        [
            "Web Application",
            "Mobile Application",
            "Data Analytics",
            "Machine Learning",
            "Enterprise Software"
        ]
    )

    team_size = st.slider(
        "Team Size",
        1,
        30,
        6
    )

    experience_level = st.selectbox(
        "Team Experience",
        [
            "Junior",
            "Mixed",
            "Senior"
        ]
    )

    duration_months = st.slider(
        "Project Duration (Months)",
        1,
        36,
        8
    )


# ==========================================
# COLUMN 2
# ==========================================

with col2:

    st.markdown("### Development Progress")

    planned_tasks = st.number_input(
        "Planned Tasks",
        min_value=1,
        max_value=1000,
        value=100
    )

    completed_tasks = st.number_input(
        "Completed Tasks",
        min_value=0,
        max_value=1000,
        value=65
    )

    requirement_changes = st.slider(
        "Requirement Changes",
        0,
        20,
        3
    )

    critical_bugs = st.slider(
        "Critical Bugs",
        0,
        20,
        2
    )

    minor_bugs = st.slider(
        "Minor Bugs",
        0,
        50,
        5
    )


# ==========================================
# COLUMN 3
# ==========================================

with col3:

    st.markdown("### Project Health")

    budget_used = st.slider(
        "Budget Used (%)",
        0,
        120,
        70
    )

    schedule_progress = st.slider(
        "Schedule Progress (%)",
        0,
        110,
        65
    )

    developer_workload = st.slider(
        "Developer Workload (%)",
        0,
        120,
        75
    )

    client_changes = st.slider(
        "Client Change Requests",
        0,
        20,
        2
    )

    testing_completion = st.slider(
        "Testing Completion (%)",
        0,
        100,
        65
    )

    previous_delays = st.slider(
        "Previous Milestone Delays",
        0,
        10,
        1
    )


# ==========================================
# VALIDATION
# ==========================================

if completed_tasks > planned_tasks:

    st.warning(
        "⚠️ Completed tasks cannot be greater than planned tasks."
    )


pending_tasks = max(
    planned_tasks - completed_tasks,
    0
)


completion_percentage = (
    completed_tasks / planned_tasks
) * 100


st.divider()


# ==========================================
# CURRENT PROJECT STATUS
# ==========================================

st.subheader("📊 Current Project Status")


m1, m2, m3, m4 = st.columns(4)


m1.metric(
    "Completed Tasks",
    completed_tasks
)

m2.metric(
    "Pending Tasks",
    pending_tasks
)

m3.metric(
    "Completion",
    f"{completion_percentage:.1f}%"
)

m4.metric(
    "Budget Used",
    f"{budget_used}%"
)


st.divider()


# ==========================================
# PREDICTION
# ==========================================

if st.button(
    "🔍 Analyze Project Risk",
    type="primary",
    use_container_width=True
):

    if completed_tasks > planned_tasks:

        st.error(
            "Please correct the task values before prediction."
        )

        st.stop()


    input_data = pd.DataFrame(
        [{
            "project_type": project_type,

            "team_size": team_size,

            "experience_level":
                experience_level,

            "duration_months":
                duration_months,

            "planned_tasks":
                planned_tasks,

            "completed_tasks":
                completed_tasks,

            "pending_tasks":
                pending_tasks,

            "requirement_changes":
                requirement_changes,

            "critical_bugs":
                critical_bugs,

            "minor_bugs":
                minor_bugs,

            "budget_used_percent":
                budget_used,

            "schedule_progress_percent":
                schedule_progress,

            "developer_workload":
                developer_workload,

            "client_change_requests":
                client_changes,

            "testing_completion_percent":
                testing_completion,

            "previous_delays":
                previous_delays
        }]
    )


    try:

        prediction = model.predict(
            input_data
        )[0]

        probabilities = model.predict_proba(
            input_data
        )[0]

        classes = model.classes_

    except Exception as e:

        st.error(
            "❌ Error while making prediction."
        )

        st.exception(e)

        st.stop()


    probability_dictionary = dict(
        zip(classes, probabilities)
    )


    confidence = (
        probability_dictionary[prediction]
        * 100
    )


    st.subheader("🎯 Prediction Result")


    if prediction == "High":

        st.error(
            "🔴 HIGH DELAY RISK"
        )

    elif prediction == "Medium":

        st.warning(
            "🟠 MEDIUM DELAY RISK"
        )

    else:

        st.success(
            "🟢 LOW DELAY RISK"
        )


    r1, r2 = st.columns(2)


    with r1:

        st.metric(
            "Predicted Risk",
            prediction
        )


    with r2:

        st.metric(
            "Model Confidence",
            f"{confidence:.1f}%"
        )


    # ======================================
    # PROBABILITY CHART
    # ======================================

    st.subheader("📊 Risk Probability")


    probability_data = pd.DataFrame({

        "Risk": classes,

        "Probability": [
            value * 100
            for value in probabilities
        ]
    })


    st.bar_chart(
        probability_data.set_index(
            "Risk"
        )
    )


    # ======================================
    # RISK FACTORS
    # ======================================

    st.subheader("⚠️ Risk Indicators")


    risk_factors = []


    pending_percentage = (
        pending_tasks / planned_tasks
    ) * 100


    if pending_percentage > 40:

        risk_factors.append(
            "Large percentage of project tasks are still pending."
        )


    if requirement_changes >= 6:

        risk_factors.append(
            "Frequent requirement changes may affect the schedule."
        )


    if critical_bugs >= 4:

        risk_factors.append(
            "The project currently has several critical bugs."
        )


    if budget_used >= 90:

        risk_factors.append(
            "Budget utilization is approaching or exceeding the planned limit."
        )


    if developer_workload >= 90:

        risk_factors.append(
            "Developer workload is high."
        )


    if client_changes >= 5:

        risk_factors.append(
            "Multiple client change requests may cause scope instability."
        )


    if testing_completion < 60:

        risk_factors.append(
            "Testing completion is relatively low."
        )


    if previous_delays >= 2:

        risk_factors.append(
            "Previous milestone delays indicate schedule instability."
        )


    if risk_factors:

        for factor in risk_factors:

            st.warning(
                "⚠️ " + factor
            )

    else:

        st.success(
            "No major rule-based warning indicators were detected."
        )


    # ======================================
    # RECOMMENDATIONS
    # ======================================

    st.subheader("💡 Recommended Actions")


    recommendations = []


    if pending_percentage > 40:

        recommendations.append(
            "Prioritize pending high-impact tasks."
        )


    if requirement_changes >= 6:

        recommendations.append(
            "Introduce stricter requirement change control."
        )


    if critical_bugs >= 4:

        recommendations.append(
            "Resolve critical bugs before developing lower-priority features."
        )


    if budget_used >= 90:

        recommendations.append(
            "Review project expenditure and remaining budget."
        )


    if developer_workload >= 90:

        recommendations.append(
            "Redistribute tasks among team members."
        )


    if testing_completion < 60:

        recommendations.append(
            "Increase testing activities before the next milestone."
        )


    if previous_delays >= 2:

        recommendations.append(
            "Review previous delay causes and revise milestone planning."
        )


    if not recommendations:

        recommendations.append(
            "Continue monitoring project milestones, quality and resources."
        )


    for recommendation in recommendations:

        st.info(
            "💡 " + recommendation
        )


    # ======================================
    # SAVE PREDICTION
    # ======================================

    history_file = (
        "data/prediction_history.csv"
    )


    history_record = {
        "project_name":
            project_name,

        "project_type":
            project_type,

        "team_size":
            team_size,

        "completed_tasks":
            completed_tasks,

        "pending_tasks":
            pending_tasks,

        "budget_used_percent":
            budget_used,

        "critical_bugs":
            critical_bugs,

        "requirement_changes":
            requirement_changes,

        "predicted_risk":
            prediction,

        "confidence":
            round(confidence, 2),

        "prediction_time":
            pd.Timestamp.now()
    }


    history_df = pd.DataFrame(
        [history_record]
    )


    os.makedirs(
        "data",
        exist_ok=True
    )


    if os.path.exists(history_file):

        history_df.to_csv(
            history_file,
            mode="a",
            header=False,
            index=False
        )

    else:

        history_df.to_csv(
            history_file,
            index=False
        )


    st.success(
        "✅ Prediction saved successfully!"
    )
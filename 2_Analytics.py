import streamlit as st
import pandas as pd
import plotly.express as px


st.title("📈 Project Analytics")


@st.cache_data
def load_data():

    return pd.read_csv(
        "data/project_data.csv"
    )


data = load_data()


# -----------------------------
# MODEL PERFORMANCE
# -----------------------------

st.subheader(
    "🤖 Model Comparison"
)


try:

    results = pd.read_csv(
        "models/model_results.csv"
    )

    display_results = (
        results.copy()
    )

    for column in [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]:

        display_results[column] = (
            display_results[column] * 100
        ).round(2)


    st.dataframe(
        display_results,
        use_container_width=True,
        hide_index=True
    )


    fig = px.bar(
        display_results,
        x="Model",
        y="F1 Score",
        title="Model F1 Score Comparison"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


except FileNotFoundError:

    st.warning(
        "Run train_model.py first."
    )


st.divider()


# -----------------------------
# RISK DISTRIBUTION
# -----------------------------

st.subheader(
    "Delay Risk Distribution"
)

risk_counts = (
    data["delay_risk"]
    .value_counts()
    .reset_index()
)

risk_counts.columns = [
    "Risk",
    "Projects"
]

fig = px.pie(
    risk_counts,
    names="Risk",
    values="Projects",
    hole=0.4
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# -----------------------------
# BUDGET ANALYSIS
# -----------------------------

st.subheader(
    "Budget Utilization vs Delay Risk"
)

fig = px.box(
    data,
    x="delay_risk",
    y="budget_used_percent",
    labels={
        "delay_risk":
            "Delay Risk",

        "budget_used_percent":
            "Budget Used (%)"
    }
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# -----------------------------
# BUG ANALYSIS
# -----------------------------

st.subheader(
    "Critical Bugs vs Risk"
)

bug_analysis = (
    data.groupby(
        "delay_risk"
    )["critical_bugs"]
    .mean()
    .reset_index()
)

fig = px.bar(
    bug_analysis,
    x="delay_risk",
    y="critical_bugs",
    labels={
        "delay_risk":
            "Risk Level",

        "critical_bugs":
            "Average Critical Bugs"
    }
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# -----------------------------
# TESTING
# -----------------------------

st.subheader(
    "Testing Completion vs Risk"
)

testing = (
    data.groupby(
        "delay_risk"
    )["testing_completion_percent"]
    .mean()
    .reset_index()
)

fig = px.bar(
    testing,
    x="delay_risk",
    y="testing_completion_percent",
    labels={
        "delay_risk":
            "Risk Level",

        "testing_completion_percent":
            "Average Testing Completion (%)"
    }
)

st.plotly_chart(
    fig,
    use_container_width=True
)
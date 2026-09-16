import streamlit as st
import pandas as pd
import plotly.express as px


st.set_page_config(
    page_title="ProjectPulse",
    page_icon="📊",
    layout="wide"
)


# -----------------------------
# LOAD DATA
# -----------------------------

@st.cache_data
def load_data():

    return pd.read_csv(
        "data/project_data.csv"
    )


data = load_data()


# -----------------------------
# HEADER
# -----------------------------

st.title("📊 ProjectPulse")

st.subheader(
    "Intelligent Software Project Risk "
    "& Delay Prediction System"
)

st.write(
    """
    ProjectPulse uses machine learning and project
    analytics to identify software projects that may
    be at risk of schedule delays.
    """
)

st.divider()


# -----------------------------
# METRICS
# -----------------------------

total_projects = len(data)

high_risk = (
    data["delay_risk"] == "High"
).sum()

medium_risk = (
    data["delay_risk"] == "Medium"
).sum()

average_budget = (
    data["budget_used_percent"].mean()
)


col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Projects",
    f"{total_projects:,}"
)

col2.metric(
    "High Risk",
    f"{high_risk:,}"
)

col3.metric(
    "Medium Risk",
    f"{medium_risk:,}"
)

col4.metric(
    "Average Budget Used",
    f"{average_budget:.1f}%"
)


st.divider()


# -----------------------------
# CHARTS
# -----------------------------

left, right = st.columns(2)


with left:

    st.subheader(
        "Risk Distribution"
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
        hole=0.45
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


with right:

    st.subheader(
        "Average Budget by Risk"
    )

    budget_data = (
        data.groupby(
            "delay_risk"
        )["budget_used_percent"]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        budget_data,
        x="delay_risk",
        y="budget_used_percent",
        labels={
            "delay_risk": "Risk Level",
            "budget_used_percent":
                "Average Budget Used (%)"
        }
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


st.divider()


# -----------------------------
# PROJECT TYPE ANALYTICS
# -----------------------------

st.subheader(
    "Project Type Distribution"
)

type_counts = (
    data["project_type"]
    .value_counts()
    .reset_index()
)

type_counts.columns = [
    "Project Type",
    "Projects"
]

fig = px.bar(
    type_counts,
    x="Project Type",
    y="Projects"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# -----------------------------
# ABOUT
# -----------------------------

st.divider()

st.info(
    """
    👈 Use the sidebar navigation to open the
    **Risk Predictor**, **Analytics**, and
    **Project History** pages.
    """
)

st.caption(
    "ProjectPulse • Machine Learning "
    "Project Risk Analytics"
)
import os
import streamlit as st
import pandas as pd


st.title("📁 Project Prediction History")


history_file = (
    "data/prediction_history.csv"
)


if not os.path.exists(history_file):

    st.info(
        """
        No prediction history is available yet.

        Analyze a project using the
        **Risk Predictor** page first.
        """
    )

    st.stop()


data = pd.read_csv(
    history_file
)


# -----------------------------
# METRICS
# -----------------------------

total = len(data)

high = (
    data["predicted_risk"] == "High"
).sum()

medium = (
    data["predicted_risk"] == "Medium"
).sum()

low = (
    data["predicted_risk"] == "Low"
).sum()


col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Predictions",
    total
)

col2.metric(
    "High Risk",
    high
)

col3.metric(
    "Medium Risk",
    medium
)

col4.metric(
    "Low Risk",
    low
)


st.divider()


# -----------------------------
# FILTER
# -----------------------------

risk_filter = st.selectbox(
    "Filter by Risk",
    [
        "All",
        "Low",
        "Medium",
        "High"
    ]
)


filtered_data = data.copy()


if risk_filter != "All":

    filtered_data = (
        filtered_data[
            filtered_data[
                "predicted_risk"
            ] == risk_filter
        ]
    )


st.dataframe(
    filtered_data,
    use_container_width=True,
    hide_index=True
)


# -----------------------------
# DOWNLOAD
# -----------------------------

csv = filtered_data.to_csv(
    index=False
).encode("utf-8")


st.download_button(
    label="⬇️ Download Prediction History",
    data=csv,
    file_name="projectpulse_predictions.csv",
    mime="text/csv"
)


# -----------------------------
# CLEAR HISTORY
# -----------------------------

st.divider()


if st.button(
    "Clear Prediction History"
):

    os.remove(history_file)

    st.success(
        "Prediction history cleared."
    )

    st.rerun()
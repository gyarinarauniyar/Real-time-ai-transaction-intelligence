"""
Chart-building functions for the Streamlit dashboard.
"""

import plotly.express as px


def risk_distribution_chart(df):
    """Create a transaction count chart by risk level."""

    fig = px.bar(
        df,
        x="final_risk_level",
        y="transaction_count",
        text="transaction_count",
        title="Transaction Risk Distribution",
        labels={
            "final_risk_level": "Risk Level",
            "transaction_count": "Transactions",
        },
    )

    fig.update_traces(textposition="outside")

    fig.update_layout(
        xaxis_title="Risk Level",
        yaxis_title="Number of Transactions",
        showlegend=False,
    )

    return fig


def risk_value_chart(df):
    """Create a transaction value chart by risk level."""

    fig = px.bar(
        df,
        x="final_risk_level",
        y="transaction_value",
        text_auto=".2s",
        title="Transaction Value by Risk Level",
        labels={
            "final_risk_level": "Risk Level",
            "transaction_value": "Transaction Value",
        },
    )

    fig.update_layout(
        xaxis_title="Risk Level",
        yaxis_title="Transaction Value",
        showlegend=False,
    )

    return fig


def daily_risk_chart(df):
    """Create a daily average-risk trend."""

    fig = px.line(
        df,
        x="transaction_date",
        y="average_risk_score",
        markers=True,
        title="Average Risk Score Over Time",
        labels={
            "transaction_date": "Date",
            "average_risk_score": "Average Risk Score",
        },
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Average Risk Score",
    )

    return fig


def daily_transaction_chart(df):
    """Create daily transaction volume chart."""

    fig = px.bar(
        df,
        x="transaction_date",
        y="transaction_count",
        title="Daily Transaction Volume",
        labels={
            "transaction_date": "Date",
            "transaction_count": "Transactions",
        },
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Transactions",
        showlegend=False,
    )

    return fig
"""
Real-Time AI Transaction Intelligence Dashboard.
"""

import streamlit as st
import pandas as pd

from dashboard.queries import (
    get_overall_kpis,
    get_risk_distribution,
    get_daily_risk,
    get_top_customers,
    get_top_merchants,
    get_transaction_details,
    get_filtered_transactions,
    get_transaction_date_range,
    get_live_risk_feed,
)

from dashboard.charts import (
    risk_distribution_chart,
    risk_value_chart,
    daily_risk_chart,
    daily_transaction_chart,
)


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="AI Transaction Intelligence",
    page_icon="💳",
    layout="wide",
)

# ---------------------------------------------------------
# Dashboard filters
# ---------------------------------------------------------

st.sidebar.header("🔎 Dashboard Filters")

selected_risk_levels = st.sidebar.multiselect(
    "Risk Level",
    options=["high", "medium", "low"],
    default=["high", "medium", "low"],
)

selected_payment_methods = st.sidebar.multiselect(
    "Payment Method",
    options=[
        "credit_card",
        "debit_card",
        "digital_wallet",
        "upi",
    ],
    default=[
        "credit_card",
        "debit_card",
        "digital_wallet",
        "upi",
    ],
)

selected_categories = st.sidebar.multiselect(
    "Merchant Category",
    options=[
        "electronics",
        "entertainment",
        "fashion",
        "fuel",
        "grocery",
        "healthcare",
        "online_services",
        "restaurant",
        "travel",
        "utilities",
    ],
    default=[
        "electronics",
        "entertainment",
        "fashion",
        "fuel",
        "grocery",
        "healthcare",
        "online_services",
        "restaurant",
        "travel",
        "utilities",
    ],
)

selected_locations = st.sidebar.multiselect(
    "Location",
    options=[
        "Ahmedabad",
        "Bangalore",
        "Chennai",
        "Delhi",
        "Hyderabad",
        "Kolkata",
        "Mumbai",
        "Pune",
    ],
    default=[
        "Ahmedabad",
        "Bangalore",
        "Chennai",
        "Delhi",
        "Hyderabad",
        "Kolkata",
        "Mumbai",
        "Pune",
    ],
)
st.sidebar.subheader("📅 Date Range")

min_date, max_date = get_transaction_date_range()

date_range = st.sidebar.date_input(
    "Transaction Date",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

if len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = end_date = date_range[0]
st.sidebar.divider()

st.sidebar.caption(
    "Filters apply to the transaction investigation table."
)
# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title("💳 Real-Time AI Transaction Intelligence")

st.markdown(
    """
    **FinTech transaction monitoring and anomaly intelligence platform**

    Monitor transaction activity, ML-detected anomalies, behavioral risk,
    and high-risk transactions from the PostgreSQL analytics warehouse.
    """
)

st.divider()


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

try:
    kpis = get_overall_kpis(
        risk_levels=selected_risk_levels,
        payment_methods=selected_payment_methods,
        merchant_categories=selected_categories,
        locations=selected_locations,
        start_date=start_date,
        end_date=end_date,
    ).iloc[0]

    risk_distribution = get_risk_distribution(
        risk_levels=selected_risk_levels,
        payment_methods=selected_payment_methods,
        merchant_categories=selected_categories,
        locations=selected_locations,
        start_date=start_date,
        end_date=end_date,
    )

    daily_risk = get_daily_risk(
        risk_levels=selected_risk_levels,
        payment_methods=selected_payment_methods,
        merchant_categories=selected_categories,
        locations=selected_locations,
        start_date=start_date,
        end_date=end_date,
    )

    top_customers = get_top_customers(
        risk_levels=selected_risk_levels,
        payment_methods=selected_payment_methods,
        merchant_categories=selected_categories,
        locations=selected_locations,
        start_date=start_date,
        end_date=end_date,
    )

    top_merchants = get_top_merchants(
        risk_levels=selected_risk_levels,
        payment_methods=selected_payment_methods,
        merchant_categories=selected_categories,
        locations=selected_locations,
        start_date=start_date,
        end_date=end_date,
    )

    filtered_transactions = get_filtered_transactions(
        risk_levels=selected_risk_levels,
        payment_methods=selected_payment_methods,
        merchant_categories=selected_categories,
        locations=selected_locations,
        start_date=start_date,
        end_date=end_date,
    )

except Exception as e:
    st.error(f"Unable to load dashboard data: {e}")
    st.stop()


# ---------------------------------------------------------
# KPI cards
# ---------------------------------------------------------

st.subheader("📊 Transaction Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Transactions",
        f"{int(kpis['total_transactions']):,}",
    )

with col2:
    st.metric(
        "Transaction Value",
        f"₹{kpis['total_transaction_value']:,.2f}",
    )

with col3:
    st.metric(
        "Average Risk",
        f"{kpis['average_risk_score']:.2f}",
    )

with col4:
    st.metric(
        "High-Risk Transactions",
        f"{int(kpis['high_risk_transactions']):,}",
    )


col5, col6, col7, col8 = st.columns(4)

with col5:
    st.metric(
        "Customers",
        f"{int(kpis['unique_customers']):,}",
    )

with col6:
    st.metric(
        "Merchants",
        f"{int(kpis['unique_merchants']):,}",
    )

with col7:
    st.metric(
        "ML Anomalies",
        f"{int(kpis['ml_anomalies']):,}",
    )

with col8:
    high_risk_rate = (
        kpis["high_risk_transactions"]
        / kpis["total_transactions"]
        * 100
    )

    st.metric(
        "High-Risk Rate",
        f"{high_risk_rate:.2f}%",
    )


st.divider()


# ---------------------------------------------------------
# Risk distribution
# ---------------------------------------------------------

st.subheader("🚨 Risk Overview")

col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(
        risk_distribution_chart(risk_distribution),
        use_container_width=True,
    )

with col2:
    st.plotly_chart(
        risk_value_chart(risk_distribution),
        use_container_width=True,
    )


# ---------------------------------------------------------
# Daily trends
# ---------------------------------------------------------

st.subheader("📈 Transaction & Risk Trends")

col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(
        daily_transaction_chart(daily_risk),
        use_container_width=True,
    )

with col2:
    st.plotly_chart(
        daily_risk_chart(daily_risk),
        use_container_width=True,
    )

st.divider()
st.subheader("⚡ Live Transaction Monitoring")


@st.fragment(run_every="2s")
def live_transaction_monitor():

    live_feed = get_live_risk_feed(50)

    if live_feed.empty:
        st.info("Waiting for incoming transactions...")
        return

    # ---------------------------------------------------------
    # Live metrics
    # ---------------------------------------------------------

    total_events = len(live_feed)
    anomaly_count = int(live_feed["is_anomaly"].sum())
    high_risk_count = int(
        live_feed["risk_level"].isin(["high", "critical"]).sum()
    )
    average_risk = live_feed["risk_score"].mean()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Events",
        f"{total_events:,}",
    )

    col2.metric(
        "Anomalies",
        f"{anomaly_count:,}",
    )

    col3.metric(
        "High Risk",
        f"{high_risk_count:,}",
    )

    col4.metric(
        "Avg Risk Score",
        f"{average_risk:.2f}",
    )

    st.caption(
        "Live feed • Auto-refreshing every 2 seconds • "
        f"Latest {total_events:,} processed transactions"
    )

    # ---------------------------------------------------------
    # Prepare display
    # ---------------------------------------------------------

    display_feed = live_feed.copy()

    display_feed["risk_score"] = display_feed["risk_score"].round(2)

    display_feed["is_anomaly"] = display_feed["is_anomaly"].map(
        {
            0: "No",
            1: "Yes",
        }
    )

    display_feed["processed_at"] = pd.to_datetime(
        display_feed["processed_at"]
    ).dt.strftime("%Y-%m-%d %H:%M:%S")

    display_feed = display_feed.rename(
        columns={
            "transaction_id": "Transaction ID",
            "processed_at": "Processed At",
            "risk_score": "Risk Score",
            "risk_level": "Risk Level",
            "is_anomaly": "Anomaly",
        }
    )

    # ---------------------------------------------------------
    # Display table
    # ---------------------------------------------------------

    st.dataframe(
        display_feed,
        use_container_width=True,
        hide_index=True,
        height=420,
    )


live_transaction_monitor()
st.divider()


# ---------------------------------------------------------
# Filtered transaction explorer
# ---------------------------------------------------------

st.subheader("🔎 Transaction Explorer")

st.caption(
    f"Showing {len(filtered_transactions):,} matching transactions "
    "(maximum 500 displayed)."
)

st.dataframe(
    filtered_transactions,
    use_container_width=True,
    hide_index=True,
)


# ---------------------------------------------------------
# Transaction investigation
# ---------------------------------------------------------

st.divider()

st.subheader("🔍 Transaction Investigation")

transaction_id = st.text_input(
    "Enter a Transaction ID",
    placeholder="Example: TXN_00007087",
)

if transaction_id:
    transaction_details = get_transaction_details(
        transaction_id.strip()
    )

    if transaction_details.empty:
        st.warning(
            f"No transaction found for ID: {transaction_id}"
        )

    else:
        transaction = transaction_details.iloc[0]

        st.success(
            f"Transaction {transaction['transaction_id']} found."
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Amount",
                f"₹{transaction['amount']:,.2f}",
            )

        with col2:
            st.metric(
                "Risk Score",
                f"{transaction['risk_score']:.2f}",
            )

        with col3:
            st.metric(
                "Risk Level",
                transaction["final_risk_level"].upper(),
            )

        with col4:
            anomaly_status = (
                "YES"
                if transaction["is_anomaly"] == 1
                else "NO"
            )

            st.metric(
                "ML Anomaly",
                anomaly_status,
            )

        st.markdown("### Transaction Details")

        detail_col1, detail_col2 = st.columns(2)

        with detail_col1:
            st.write(
                f"**Transaction ID:** "
                f"{transaction['transaction_id']}"
            )

            st.write(
                f"**Timestamp:** "
                f"{transaction['transaction_timestamp']}"
            )

            st.write(
                f"**Customer:** "
                f"{transaction['customer_id']}"
            )

            st.write(
                f"**Merchant:** "
                f"{transaction['merchant_id']}"
            )

            st.write(
                f"**Category:** "
                f"{transaction['merchant_category']}"
            )

            st.write(
                f"**Payment Method:** "
                f"{transaction['payment_method']}"
            )

        with detail_col2:
            st.write(
                f"**Location:** "
                f"{transaction['location']}"
            )

            st.write(
                f"**Device:** "
                f"{transaction['device_type']}"
            )

            st.write(
                f"**Behavioral Risk:** "
                f"{transaction['behavioral_risk_score']:.2f}"
            )

            st.write(
                f"**Anomaly Prediction:** "
                f"{transaction['anomaly_prediction']}"
            )

            st.write(
                f"**Anomaly Type:** "
                f"{transaction['anomaly_type']}"
            )

            st.write(
                f"**Injected Anomaly:** "
                f"{'Yes' if transaction['is_injected_anomaly'] == 1 else 'No'}"
            )

        st.markdown("### ⚠️ Risk Explanation")

        st.info(
            transaction["risk_explanation"]
        )

st.divider()


# ---------------------------------------------------------
# Customer & merchant intelligence
# ---------------------------------------------------------

st.subheader("👥 Customer Risk Intelligence")

st.dataframe(
    top_customers,
    use_container_width=True,
    hide_index=True,
)


st.subheader("🏪 Merchant Risk Intelligence")

st.dataframe(
    top_merchants,
    use_container_width=True,
    hide_index=True,
)


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.divider()

st.caption(
    "Real-Time AI Transaction Intelligence Platform • "
    "PostgreSQL + Python + ML + Streamlit"
)
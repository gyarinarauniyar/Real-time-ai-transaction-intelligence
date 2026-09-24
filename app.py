import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Transaction Intelligence Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PLATFORM HOME
# ============================================================

def home():

    st.title("🤖 AI Transaction Intelligence Platform")

    st.markdown(
        """
        ### Real-Time FinTech Intelligence

        A real-time transaction intelligence platform that combines
        streaming data, machine-learning anomaly detection, behavioral
        risk scoring, MLOps monitoring, and a local AI analyst.

        Use the applications below to monitor transactions or investigate
        transaction risk using natural language.
        """
    )

    st.divider()

    # ========================================================
    # APPLICATIONS
    # ========================================================

    st.header("Platform Applications")

    col1, col2 = st.columns(2, gap="large")

    # --------------------------------------------------------
    # TRANSACTION INTELLIGENCE
    # --------------------------------------------------------

    with col1:

        st.subheader("📊 AI Transaction Intelligence")

        st.write(
            """
            Real-time transaction monitoring and risk intelligence.

            Monitor incoming transaction events, filter and investigate
            transactions, analyze risk distributions, compare anomalous
            behavior, inspect ML predictions, and investigate individual
            transactions.
            """
        )

        st.write("")

        if st.button(
            "Open Transaction Intelligence",
            icon="📊",
            type="primary",
            use_container_width=True,
        ):
            st.switch_page("platform_pages/dashboard_page.py")

    # --------------------------------------------------------
    # AI ANALYST
    # --------------------------------------------------------

    with col2:

        st.subheader("🤖 AI Transaction Analyst")

        st.write(
            """
            Natural-language intelligence over transaction data and
            project knowledge.

            Ask questions about transactions, risk classifications,
            anomaly detection, model behavior, risk rules, monitoring,
            and platform architecture.
            """
        )

        st.write("")

        if st.button(
            "Open AI Transaction Analyst",
            icon="🤖",
            use_container_width=True,
        ):
            st.switch_page("platform_pages/analyst_page.py")

    st.divider()

    # ========================================================
    # PLATFORM SUMMARY
    # ========================================================

    st.header("What the Platform Does")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("⚡ Detect")
        st.write(
            "Processes transaction events and detects unusual behavioral patterns."
        )

    with col2:
        st.subheader("🎯 Assess")
        st.write(
            "Combines ML anomaly signals with deterministic behavioral risk scoring."
        )

    with col3:
        st.subheader("🤖 Explain")
        st.write(
            "Uses database facts, RAG knowledge, SQL and a local LLM to explain results."
        )

    st.divider()

    st.caption(
        "AI Transaction Intelligence Platform"
    )


# ============================================================
# STREAMLIT MULTI-PAGE NAVIGATION
# ============================================================

home_page = st.Page(
    home,
    title="Platform Home",
    icon="🏠",
    default=True,
)

dashboard_page = st.Page(
    "platform_pages/dashboard_page.py",
    title="AI Transaction Intelligence",
    icon="📊",
)

analyst_page = st.Page(
    "platform_pages/analyst_page.py",
    title="AI Transaction Analyst",
    icon="🤖",
)


navigation = st.navigation(
    {
        "Platform": [
            home_page,
        ],
        "Applications": [
            dashboard_page,
            analyst_page,
        ],
    }
)


navigation.run()
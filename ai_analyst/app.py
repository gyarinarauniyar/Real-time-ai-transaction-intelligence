import sys
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from ai_analyst.analyst import analyze


st.set_page_config(
    page_title="AI Transaction Analyst",
    page_icon="🤖",
    layout="wide"
)


st.title("🤖 AI Transaction Analyst")

st.caption(
    "Natural-language analytics for the "
    "Real-Time AI Transaction Intelligence Platform"
)

st.divider()


question = st.text_input(
    "Ask a question about your transaction intelligence data",
    placeholder="Example: How many high-risk transactions were detected?"
)


if st.button("Analyze", type="primary"):

    if not question.strip():

        st.warning("Please enter a question.")

    else:

        try:

            with st.spinner(
                "Analyzing transaction intelligence data..."
            ):

                result = analyze(question)


            # ------------------------------------------
            # Intent
            # ------------------------------------------

            st.caption(
                f"Detected intent: **{result['intent'].upper()}**"
            )


            # ------------------------------------------
            # Analysis
            # ------------------------------------------

            st.subheader("📊 Analysis")

            st.info(
                result["explanation"]
            )


            # ------------------------------------------
            # Database results
            # ------------------------------------------

            raw_results = result["results"]

            if isinstance(raw_results, list):

                dataframe = pd.DataFrame(
                    raw_results
                )

                if not dataframe.empty:

                    st.subheader("📋 Query Results")

                    st.dataframe(
                        dataframe,
                        use_container_width=True,
                        hide_index=True
                    )


            # ------------------------------------------
            # SQL
            # ------------------------------------------

            if result["sql"]:

                with st.expander(
                    "🔍 View Generated SQL"
                ):

                    st.code(
                        result["sql"],
                        language="sql"
                    )


            # ------------------------------------------
            # RAG context
            # ------------------------------------------

            with st.expander(
                "📚 View Retrieved Knowledge"
            ):

                st.text(
                    result["knowledge_context"]
                )


        except Exception as error:

            st.error(
                f"Analysis failed: {error}"
            )
import re
import requests

from .config import OLLAMA_URL, MODEL_NAME
from .database import execute_query
from .sql_generator import (
    generate_sql,
    generate_transaction_sql,
    extract_transaction_id
)

from ai_analyst.rag_context import get_relevant_context
from ai_analyst.sql_validator import validate_sql


# ============================================================
# QUESTION CLASSIFICATION
# ============================================================

def classify_question(question):
    """
    Classify the question into:

        knowledge
        data
        hybrid

    Deterministic routing is preferred.
    Ollama is used only for genuinely ambiguous questions.
    """

    q = question.lower().strip()

    transaction_id = extract_transaction_id(question)

    # --------------------------------------------------
    # KNOWLEDGE QUESTIONS
    # --------------------------------------------------

    knowledge_patterns = [
        "what features",
        "which features",
        "features used",
        "how does the model",
        "how does isolation forest",
        "what is isolation forest",
        "what is anomaly detection",
        "how does anomaly detection work",
        "how does the platform detect",
        "how does the platform detect unusual",
        "how does the system detect",
        "how are unusual transactions detected",
        "how are anomalies detected",
        "why does the platform detect",
        "how is risk calculated",
        "how is risk score calculated",
        "what are the risk rules",
        "what is the risk rule",
        "how does risk scoring work",
        "how does the risk engine work",
        "what is the risk score",
        "what does the anomaly score mean",
        "how does monitoring work",
        "what is data drift",
        "what is the production model",
        "how does the system work",
        "how does the platform work",
        "what is rag",
        "how does rag work",
        "how does mlops work",
        "what is mlops",
        "what is the ai analyst",
        "how does the ai analyst work",
        "what is the architecture",
        "how does the architecture work",
        "what is data quality",
        "how is data quality checked",
        "how does the pipeline work",
    ]

    for pattern in knowledge_patterns:
        if pattern in q:
            return "knowledge"

    # --------------------------------------------------
    # TRANSACTION-SPECIFIC QUESTIONS
    # --------------------------------------------------

    hybrid_patterns = [
        "why was",
        "why is",
        "why did",
        "explain this transaction",
        "explain the transaction",
        "what caused this transaction",
        "why was this transaction",
        "why is this transaction",
        "why did this transaction",
        "explain this",
    ]

    if transaction_id:

        for pattern in hybrid_patterns:
            if pattern in q:
                return "hybrid"

        # Any direct transaction-ID question requires
        # actual database information.
        return "hybrid"

    # --------------------------------------------------
    # DATA QUESTIONS
    # --------------------------------------------------

    data_patterns = [
        "how many",
        "count",
        "total",
        "average",
        "avg",
        "maximum",
        "minimum",
        "highest",
        "lowest",
        "show me",
        "show",
        "list",
        "transactions",
        "transaction records",
        "latest transactions",
        "recent transactions",
        "today",
        "yesterday",
        "this week",
        "this month",
        "by location",
        "by payment",
        "by merchant",
        "by category",
    ]

    for pattern in data_patterns:
        if pattern in q:
            return "data"

    # --------------------------------------------------
    # OLLAMA FALLBACK
    # --------------------------------------------------

    prompt = f"""
You are an intent classifier for a transaction intelligence platform.

Classify the user's question into exactly ONE category:

KNOWLEDGE
DATA
HYBRID

KNOWLEDGE:
Questions about how the platform works, anomaly detection,
risk scoring, model behavior, MLOps, RAG, architecture,
data quality, or definitions.

DATA:
Questions asking for actual transaction/database statistics,
counts, averages, totals, records, dates, locations,
payment methods, or categories.

HYBRID:
Questions about a specific transaction that require both
database facts and project knowledge.

USER QUESTION:
{question}

Return only:
KNOWLEDGE
DATA
or
HYBRID
"""

    try:

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0,
                    "num_predict": 5
                }
            },
            timeout=30
        )

        response.raise_for_status()

        intent = response.json()["response"].strip().upper()

        if "HYBRID" in intent:
            return "hybrid"

        if "DATA" in intent:
            return "data"

        return "knowledge"

    except Exception:

        # Safe fallback.
        return "knowledge"


# ============================================================
# RESULT FORMATTING
# ============================================================

def format_results(columns, rows):

    if not rows:
        return "No matching records were found."

    output = []

    for row in rows:

        record = {}

        for column, value in zip(columns, row):
            record[column] = value

        output.append(record)

    return output


# ============================================================
# KNOWLEDGE ANSWER
# ============================================================

def generate_knowledge_answer(
    question,
    knowledge_context
):
    """
    Answer documentation/model questions using RAG context.

    No database query is performed.
    """

    prompt = f"""
You are the AI Analyst for a
Real-Time AI Transaction Intelligence Platform.

Answer the user's question using ONLY the project knowledge
provided below.

PROJECT KNOWLEDGE:
{knowledge_context}

USER QUESTION:
{question}

Rules:

- Do not invent facts.
- Do not use unsupported general knowledge.
- Do not create database statistics.
- Do not invent implementation details.
- If the project knowledge does not contain enough information,
  say so clearly.
- Keep the answer concise and factual.
- Explain the platform in practical terms.

Answer:
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 180
            }
        },
        timeout=120
    )

    response.raise_for_status()

    return response.json()["response"].strip()


# ============================================================
# DATABASE ANSWER
# ============================================================

def generate_explanation(
    question,
    sql,
    results,
    knowledge_context
):
    """
    Convert verified database results into a concise user-facing answer.

    Simple aggregate results are formatted deterministically.
    More complex results can still use Ollama for summarization.
    """

    # --------------------------------------------------
    # No results
    # --------------------------------------------------

    if not results or results == "No matching records were found.":
        return "No matching records were found."

    # --------------------------------------------------
    # Simple aggregate results
    # --------------------------------------------------

    if isinstance(results, list) and len(results) == 1:
        record = results[0]

        if "transaction_count" in record:
            return (
                f"There are {record['transaction_count']} transactions "
                "in the current transaction event stream."
            )

        if "anomaly_count" in record:
            return (
                f"There are {record['anomaly_count']} transactions "
                "currently flagged as anomalies."
            )

        if "high_risk_count" in record:
            return (
                f"There are {record['high_risk_count']} high-risk "
                "transactions."
            )

        if "critical_count" in record:
            return (
                f"There are {record['critical_count']} critical-risk "
                "transactions."
            )

        if "average_transaction_amount" in record:
            value = record["average_transaction_amount"]

            if value is not None:
                return (
                    f"The average transaction amount is "
                    f"{float(value):.2f}."
                )

        if "total_transaction_amount" in record:
            value = record["total_transaction_amount"]

            if value is not None:
                return (
                    f"The total transaction amount is "
                    f"{float(value):.2f}."
                )

    # --------------------------------------------------
    # Complex results
    # --------------------------------------------------

    prompt = f"""
You are the AI Analyst for a Real-Time AI Transaction Intelligence Platform.

Answer the user's question using ONLY the verified database result below.

USER QUESTION:
{question}

VERIFIED DATABASE RESULT:
{results}

PROJECT KNOWLEDGE:
{knowledge_context}

Rules:

- Treat the database result as the only source of transaction facts.
- Do not invent statistics or transaction attributes.
- Do not mention SQL.
- Do not mention PostgreSQL.
- Do not mention "source of truth".
- Do not mention prompts, models, AI rules, or internal processing.
- Do not repeat the raw Python/database result.
- Do not claim fraud unless explicitly present in the data.
- Keep the answer concise and natural.
- Answer the user's question directly.

Return ONLY the final user-facing answer.

Answer:
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": 120
            }
        },
        timeout=120
    )

    response.raise_for_status()

    return response.json()["response"].strip()
def generate_hybrid_answer(
    question,
    sql,
    results,
    knowledge_context
):
    """
    Explain a specific transaction using verified database facts.

    Transaction risk and anomaly status are formatted deterministically
    from the database. No LLM is used to reinterpret these facts.
    """

    # --------------------------------------------------
    # No database record
    # --------------------------------------------------

    if (
        not results
        or results == "No matching records were found."
    ):
        transaction_id = extract_transaction_id(question)

        return (
            f"No transaction record was found for "
            f"{transaction_id}. "
            "I cannot determine its risk level or anomaly status "
            "without a corresponding database record."
        )

    # --------------------------------------------------
    # Extract verified transaction facts
    # --------------------------------------------------

    if not isinstance(results, list) or not results:
        return "No matching transaction record was found."

    verified = results[0]

    transaction_id = verified.get("transaction_id")
    risk_score = verified.get("risk_score")
    risk_level = verified.get("risk_level")
    anomaly_prediction = verified.get("anomaly_prediction")
    is_anomaly = verified.get("is_anomaly")

    # --------------------------------------------------
    # Normalize values
    # --------------------------------------------------

    try:
        if risk_score is not None:
            risk_score = float(risk_score)
    except (TypeError, ValueError):
        pass

    if risk_level is not None:
        risk_level = str(risk_level).lower()

    try:
        if anomaly_prediction is not None:
            anomaly_prediction = int(anomaly_prediction)
    except (TypeError, ValueError):
        pass

    try:
        if is_anomaly is not None:
            is_anomaly = int(is_anomaly)
    except (TypeError, ValueError):
        pass

    # --------------------------------------------------
    # Build risk statement
    # --------------------------------------------------

    if risk_level is not None and risk_score is not None:
        risk_statement = (
            f"{transaction_id} is classified as "
            f"{risk_level} risk with a risk score of "
            f"{risk_score:.2f}."
        )

    elif risk_level is not None:
        risk_statement = (
            f"{transaction_id} is classified as "
            f"{risk_level} risk."
        )

    elif risk_score is not None:
        risk_statement = (
            f"{transaction_id} has a risk score of "
            f"{risk_score:.2f}."
        )

    else:
        risk_statement = (
            f"The risk classification for {transaction_id} "
            "is unavailable."
        )

    # --------------------------------------------------
    # Build anomaly statement
    # --------------------------------------------------

    if anomaly_prediction == -1:
        anomaly_statement = (
            "The Isolation Forest classified the transaction "
            "as an anomaly."
        )

    elif anomaly_prediction == 1:
        anomaly_statement = (
            "The Isolation Forest did not classify the "
            "transaction as an anomaly."
        )

    else:
        anomaly_statement = (
            "The anomaly prediction is unavailable."
        )

    # --------------------------------------------------
    # Build explicit anomaly flag statement
    # --------------------------------------------------

    if is_anomaly == 1:
        flag_statement = (
            "The transaction is explicitly flagged as an anomaly."
        )

    elif is_anomaly == 0:
        flag_statement = (
            "The transaction is not explicitly flagged as an anomaly."
        )

    else:
        flag_statement = (
            "The anomaly flag is unavailable."
        )

    # --------------------------------------------------
    # Correct incorrect user assumptions
    # --------------------------------------------------

    correction = ""

    if (
        "high risk" in question.lower()
        and risk_level is not None
        and risk_level != "high"
    ):
        correction = (
            f"{transaction_id} is not classified as high risk. "
        )

    # --------------------------------------------------
    # Final response
    # --------------------------------------------------

    return (
        f"{correction}"
        f"{risk_statement} "
        f"{anomaly_statement} "
        f"{flag_statement}"
    )
def analyze(question):

    question = question.strip()

    if not question:
        raise ValueError("Please enter a question.")

    # --------------------------------------------------
    # STEP 1 — Determine intent FIRST
    # --------------------------------------------------

    intent = classify_question(question)

    # --------------------------------------------------
    # STEP 2 — Knowledge question
    # --------------------------------------------------

    if intent == "knowledge":

        knowledge_context = get_relevant_context(
            question,
            top_k=3
        )

        explanation = generate_knowledge_answer(
            question,
            knowledge_context
        )

        return {
            "question": question,
            "intent": intent,
            "sql": None,
            "results": None,
            "explanation": explanation,
            "knowledge_context": knowledge_context
        }

    # --------------------------------------------------
    # STEP 3 — Hybrid transaction question
    # --------------------------------------------------

    if intent == "hybrid":

        knowledge_context = get_relevant_context(
            question,
            top_k=3
        )

        sql = generate_transaction_sql(question)

        validate_sql(sql)

        columns, rows = execute_query(sql)

        results = format_results(
            columns,
            rows
        )

        # --------------------------------------------------
        # IMPORTANT:
        # If the transaction does not exist, return the
        # deterministic response from generate_hybrid_answer.
        #
        # Ollama will NOT be called in this case.
        # --------------------------------------------------

        explanation = generate_hybrid_answer(
            question,
            sql,
            results,
            knowledge_context
        )

        return {
            "question": question,
            "intent": intent,
            "sql": sql,
            "results": results,
            "explanation": explanation,
            "knowledge_context": knowledge_context
        }

    # --------------------------------------------------
    # STEP 4 — Data question
    # --------------------------------------------------

    if intent == "data":

        knowledge_context = get_relevant_context(
            question,
            top_k=2
        )

        sql = generate_sql(question)

        validate_sql(sql)

        columns, rows = execute_query(sql)

        results = format_results(
            columns,
            rows
        )

        explanation = generate_explanation(
            question,
            sql,
            results,
            knowledge_context
        )

        return {
            "question": question,
            "intent": intent,
            "sql": sql,
            "results": results,
            "explanation": explanation,
            "knowledge_context": knowledge_context
        }

    raise ValueError(
        f"Unsupported intent: {intent}"
    )
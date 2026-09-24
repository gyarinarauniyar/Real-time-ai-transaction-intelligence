import requests

from .config import OLLAMA_URL, MODEL_NAME
from .database import execute_query
from .sql_generator import (
    generate_sql,
    generate_transaction_sql
)
from ai_analyst.rag_context import get_relevant_context
from ai_analyst.sql_validator import validate_sql


def classify_question(question):
    """
    Classify the user question into:
    - knowledge
    - data
    - hybrid

    Deterministic rules are used first for common platform
    questions, then Ollama handles ambiguous questions.
    """

    q = question.lower().strip()

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
        "why is a transaction classified",
        "why was a transaction classified",
        "why is this transaction classified",
        "why was this transaction classified",
        "why high risk",
        "why is high risk",
        "what makes a transaction high risk",
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
    ]

    for pattern in knowledge_patterns:
        if pattern in q:
            return "knowledge"

    # --------------------------------------------------
    # HYBRID QUESTIONS
    # --------------------------------------------------

    hybrid_patterns = [
        "why was",
        "why is",
        "explain this transaction",
        "explain the transaction",
        "why did this transaction",
        "what caused this transaction",
        "why did this become",
    ]

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
        "list",
        "transactions",
        "transaction records",
        "latest transactions",
        "recent transactions",
        "today",
        "yesterday",
        "this week",
        "this month",
    ]

    # A "why" question involving a specific transaction
    # should use both database data and project knowledge.
    for pattern in hybrid_patterns:
        if pattern in q:
            for data_pattern in data_patterns:
                if data_pattern in q:
                    return "hybrid"

    # Explicit transaction ID → hybrid
    if "txn_" in q:
        for pattern in hybrid_patterns:
            if pattern in q:
                return "hybrid"

    # --------------------------------------------------
    # DATA QUESTIONS
    # --------------------------------------------------

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
Questions about:
- model features
- model behavior
- anomaly detection
- risk rules
- risk scoring
- monitoring
- data quality
- definitions
- project architecture
- how the system works

DATA
Questions requiring PostgreSQL transaction data:
- counts
- totals
- averages
- transaction records
- transaction statistics
- dates
- locations
- payment methods
- database information

HYBRID
Questions requiring BOTH:
- actual transaction data
AND
- project/model/risk-rule knowledge

USER QUESTION:
{question}

Return ONLY one word:
KNOWLEDGE
DATA
or
HYBRID
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": 10
            }
        },
        timeout=60
    )

    response.raise_for_status()

    intent = response.json()["response"].strip().upper()

    if "HYBRID" in intent:
        return "hybrid"

    if "DATA" in intent:
        return "data"

    return "knowledge"


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


def generate_knowledge_answer(question, knowledge_context):
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
- Do not use general knowledge that is not supported
  by the project knowledge.
- Do not create database statistics.
- Do not make business recommendations unless the
  project knowledge supports them.
- If the project knowledge does not contain enough
  information, say so clearly.
- Be concise and factual.

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
                "num_predict": 256
            }
        },
        timeout=180
    )

    response.raise_for_status()

    return response.json()["response"].strip()


def generate_explanation(
    question,
    sql,
    results,
    knowledge_context
):
    """
    Explain database results using database output
    and relevant project knowledge.
    """

    prompt = f"""
You are the AI Analyst for a
Real-Time AI Transaction Intelligence Platform.

Answer the user's question using:

1. The PostgreSQL query result.
2. The project knowledge provided below.

PROJECT KNOWLEDGE:
{knowledge_context}

USER QUESTION:
{question}

SQL USED:
{sql}

DATABASE RESULT:
{results}

Rules:
- Do not invent statistics.
- Do not claim information that is not present
  in the database result.
- Do not invent trends from a small sample.
- Do not claim that a transaction is fraudulent
  unless the database explicitly provides such a label.
- If the database result does not contain enough
  information, say so clearly.
- Keep the answer concise and business-oriented.

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
                "num_predict": 256
            }
        },
        timeout=180
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
    Explain a specific transaction using:
    - actual PostgreSQL transaction data
    - retrieved project knowledge
    """

    prompt = f"""
You are the AI Analyst for a
Real-Time AI Transaction Intelligence Platform.

The user wants an explanation of a transaction's risk.

Use BOTH:

1. ACTUAL DATABASE RESULT
2. PROJECT KNOWLEDGE

PROJECT KNOWLEDGE:
{knowledge_context}

USER QUESTION:
{question}

SQL USED:
{sql}

ACTUAL DATABASE RESULT:
{results}

Rules:

- Use only facts present in the database result.
- Use the project knowledge to explain risk rules
  and model behavior.
- Do not invent transaction attributes.
- Do not invent risk factors.
- Do not claim fraud unless the database explicitly
  provides a fraud label.
- Explain which available risk signals or rules
  are relevant.
- If the available information is insufficient
  to determine the exact reason, clearly say so.
- Do not confuse an anomaly signal with proof of fraud.
- Keep the answer concise and business-oriented.

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
                "num_predict": 300
            }
        },
        timeout=180
    )

    response.raise_for_status()

    return response.json()["response"].strip()


def analyze(question):

    # --------------------------------------------------
    # STEP 1 — Retrieve relevant project knowledge
    # --------------------------------------------------

    knowledge_context = get_relevant_context(
        question,
        top_k=3
    )

    # --------------------------------------------------
    # STEP 2 — Determine question intent
    # --------------------------------------------------

    intent = classify_question(question)

    # --------------------------------------------------
    # STEP 3 — Knowledge question
    # --------------------------------------------------

    if intent == "knowledge":

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
    # STEP 4 — Hybrid question
    # --------------------------------------------------

    if intent == "hybrid":

        sql = generate_transaction_sql(question)

        # Validate LLM-generated SQL before execution
        validate_sql(sql)

        columns, rows = execute_query(sql)

        results = format_results(
            columns,
            rows
        )

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
    # STEP 5 — Data question
    # --------------------------------------------------

    sql = generate_sql(question)

    # Validate LLM-generated SQL before execution
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
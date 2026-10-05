# Real-Time AI Transaction Intelligence Platform

A production-style FinTech platform for **real-time transaction monitoring, anomaly detection, risk scoring, AI-assisted investigation, and ML model monitoring**.

The platform combines streaming transaction processing, machine learning, rule-based risk scoring, PostgreSQL analytics, MLOps monitoring, and a RAG-powered AI Analyst into a single local application.

---

## 🚀 Overview

The platform processes transaction events through a real-time risk pipeline:

```text
Transaction Event
       │
       ▼
   Event Queue
       │
       ▼
Real-Time Consumer
       │
       ▼
Feature Engineering
       │
       ▼
Isolation Forest
       │
       ▼
Behavioral Risk Engine
       │
       ▼
Final Risk Score
       │
       ▼
PostgreSQL
       │
       ├──────────────► Live Dashboard
       │
       ├──────────────► MLOps Monitoring
       │
       └──────────────► AI Analyst
                              │
                         RAG + Ollama
```

The system is designed to distinguish between:

- **Anomaly detection** — identifying transactions that differ from learned behavior.
- **Risk scoring** — combining ML and behavioral signals into an operational risk level.
- **AI investigation** — allowing users to query transaction data and project knowledge using natural language.

---

## ✨ Key Features

### Real-Time Transaction Processing

- Generates and processes transaction events locally.
- Uses a producer/consumer streaming architecture.
- Stores transaction events and risk events in PostgreSQL.
- Supports continuous transaction risk evaluation.

### Machine Learning Anomaly Detection

- Uses **Isolation Forest** for unsupervised anomaly detection.
- Uses production-oriented transaction features.
- Stores trained models for inference.
- Produces anomaly predictions for individual transactions.

### Behavioral Risk Engine

Combines transaction behavior and ML signals to produce a risk score.

Risk levels are based on the platform's configured thresholds:

| Risk Score | Risk Level |
|---:|---|
| `< 30` | Low |
| `30 – 59.99` | Medium |
| `60 – 79.99` | High |
| `≥ 80` | Critical |

An anomaly signal is treated as a risk indicator and **not as proof of fraud**.

### Live Monitoring Dashboard

The Streamlit dashboard provides:

- Event monitoring
- Anomaly counts
- High-risk and critical-risk metrics
- Average risk
- Risk-level filtering
- Transaction tables
- Risk visualizations
- Transaction investigation

### AI Analyst

The AI Analyst supports three types of questions:

#### Knowledge

Questions about the platform itself.

Examples:

```text
What is Isolation Forest?
How does risk scoring work?
What is RAG?
How does MLOps work?
```

These use the project's knowledge base through **Retrieval-Augmented Generation (RAG)** and Ollama.

#### Data

Questions requiring actual transaction data.

Examples:

```text
How many transactions are there?
How many high risk transactions are there?
What is the average transaction amount?
```

These are resolved using PostgreSQL queries.

#### Hybrid / Transaction Investigation

Questions about specific transactions.

Examples:

```text
Explain TXN_00007657
Why is TXN_00008457 high risk?
```

Transaction risk and anomaly facts are taken directly from the database rather than allowing the LLM to invent or reinterpret them.

### RAG Knowledge Retrieval

The AI Analyst uses a local knowledge base containing project documentation such as:

- Feature definitions
- Risk rules
- Model documentation
- Monitoring documentation

Embeddings are generated using:

```text
all-MiniLM-L6-v2
```

### MLOps

The project includes components for:

- Model registry
- Prediction logging
- Monitoring windows
- Drift detection
- Alerts
- Prediction feedback
- Model evaluation
- Retraining and rollback workflows

---

## 🧠 Machine Learning

### Isolation Forest

Isolation Forest is used as the primary unsupervised anomaly detection model.

The model identifies transaction patterns that differ from learned transaction behavior.

The platform uses the following convention:

```text
anomaly_prediction = -1 → anomaly
anomaly_prediction =  1 → not classified as anomaly
```

The anomaly prediction is combined with behavioral signals to produce the final transaction risk assessment.

---

## 🏗️ Architecture

```text
                         ┌──────────────────────┐
                         │ Transaction Producer │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ PostgreSQL Event     │
                         │ Queue                │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Real-Time Consumer   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Feature Engineering  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Isolation Forest     │
                         │ Anomaly Detection    │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Behavioral Risk      │
                         │ Engine               │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ PostgreSQL           │
                         │ Transaction + Risk   │
                         │ Data                 │
                         └──────┬───────┬───────┘
                                │       │
                  ┌─────────────┘       └──────────────┐
                  ▼                                    ▼
        ┌──────────────────┐                 ┌──────────────────┐
        │ Streamlit        │                 │ MLOps Monitoring │
        │ Dashboard        │                 │                  │
        └────────┬─────────┘                 └──────────────────┘
                 │
                 ▼
        ┌──────────────────┐
        │ AI Analyst       │
        │                  │
        │ PostgreSQL       │
        │ + RAG            │
        │ + Ollama         │
        └──────────────────┘
```

---

## 📂 Project Structure

```text
real-time-ai-transaction-intelligence/
│
├── ai_analyst/
│   ├── analyst.py
│   ├── config.py
│   ├── database.py
│   ├── rag_context.py
│   ├── sql_generator.py
│   └── sql_validator.py
│
├── dashboard/
│   └── app.py
│
├── rag/
│   ├── knowledge/
│   │   ├── feature_dictionary.md
│   │   ├── model_documentation.md
│   │   ├── monitoring.md
│   │   └── risk_rules.md
│   └── retriever.py
│
├── streaming/
│   ├── producer.py
│   ├── realtime_consumer.py
│   └── ...
│
├── src/
│   ├── analytics/
│   ├── data_quality/
│   ├── database/
│   ├── feature_engineering/
│   ├── ingestion/
│   ├── ml/
│   │   ├── anomaly_detection/
│   │   ├── inference/
│   │   └── risk_engine/
│   ├── processing/
│   └── utils/
│
├── models/
│
├── data/
│
├── app.py
├── pyproject.toml
├── .gitignore
└── README.md
```

---

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Anomaly Detection | Isolation Forest |
| Database | PostgreSQL |
| Streaming | Python producer/consumer pipeline |
| API / Backend | FastAPI, Uvicorn |
| Dashboard | Streamlit |
| Visualization | Plotly |
| AI / LLM | Ollama |
| Embeddings | Sentence Transformers |
| RAG | Local vector retrieval |
| MLOps | MLflow, PostgreSQL-based monitoring |
| Data Generation | Faker |
| Testing | Pytest |
| Environment | Python virtual environment |
| Version Control | Git / GitHub |

---

## 💻 Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/gyarinarauniyar/Real-time-ai-transaction-intelligence.git

cd Real-time-ai-transaction-intelligence
```

### 2. Create a virtual environment

```powershell
python -m venv venv
```

Activate it:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

If the project is being installed from `pyproject.toml`, use the project's configured dependency installation instead.

### 4. Configure environment variables

Create a `.env` file containing the required PostgreSQL configuration:

```env
POSTGRES_USER=your_user
POSTGRES_PASSWORD=your_password
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=transaction_intelligence
```

Do not commit `.env` to GitHub.

### 5. Start PostgreSQL

Make sure PostgreSQL is running and the project database is available.

### 6. Start Ollama

Install Ollama and make sure the configured local model is available.

Example:

```powershell
ollama list
```

The AI Analyst is configured to use the local Ollama API.

---

## ▶️ Running the Platform

From the project root:

```powershell
streamlit run app.py
```

The application provides:

```text
Platform Home
├── Dashboard
└── AI Analyst
```

### Start the streaming producer

In another terminal:

```powershell
python -m streaming.producer
```

### Start the real-time consumer

In another terminal:

```powershell
python -m streaming.realtime_consumer
```

The consumer processes transaction events and writes risk results to PostgreSQL.

---

## 🔍 Example AI Analyst Queries

### Knowledge

```text
What is Isolation Forest?
```

```text
How does the platform detect unusual transactions?
```

```text
How does risk scoring work?
```

```text
What is RAG?
```

```text
How does MLOps work?
```

### Data

```text
How many transactions are there?
```

```text
How many high risk transactions are there?
```

```text
How many critical transactions are there?
```

```text
What is the average transaction amount?
```

### Transaction Investigation

```text
Explain TXN_00007657
```

```text
Why is TXN_00008457 high risk?
```

The analyst validates transaction information against PostgreSQL before producing a risk explanation.

---

## 🔐 Safety and Reliability

The AI Analyst is designed with several safeguards:

- Database transaction facts are treated as authoritative.
- LLM-generated SQL is not blindly executed.
- SQL validation restricts queries to approved project tables.
- Write operations are blocked by the SQL validator.
- Missing transactions do not trigger fabricated explanations.
- Risk levels are not inferred from user wording.
- Anomaly predictions are not treated as proof of fraud.
- The LLM is not allowed to override database risk classifications.
- Project knowledge is separated from live transaction data.

---

## 🧪 Testing

Python modules can be syntax-checked with:

```powershell
python -m compileall ai_analyst streaming dashboard rag src
```

Project tests can be executed with:

```powershell
pytest
```

The platform should also be tested across:

- Transaction ingestion
- Real-time risk scoring
- PostgreSQL persistence
- Dashboard updates
- AI Analyst queries
- RAG retrieval
- MLOps logging
- Missing transaction handling
- Invalid or failed events

---

## 📊 Risk Interpretation

The platform uses risk levels to prioritize transactions for monitoring:

```text
LOW
 │
 ├── score < 30
 │
MEDIUM
 │
 ├── score ≥ 30
 │
HIGH
 │
 ├── score ≥ 60
 │
CRITICAL
 │
 └── score ≥ 80
```

Risk classification is an operational signal and should not automatically be interpreted as confirmed fraud.

---

## 🎯 Project Goals

This project demonstrates practical experience with:

- Real-time data processing
- Transaction monitoring
- Data engineering
- Machine learning
- Unsupervised anomaly detection
- Feature engineering
- Risk scoring
- PostgreSQL data systems
- Streamlit dashboards
- RAG pipelines
- Local LLM integration
- AI-assisted data investigation
- MLOps concepts
- Model monitoring
- Git/GitHub workflows

---

## 📌 Current Status

The core platform is operational with:

- Real-time transaction processing
- Isolation Forest anomaly detection
- Behavioral risk scoring
- PostgreSQL persistence
- Live Streamlit monitoring
- AI Analyst
- RAG knowledge retrieval
- Local Ollama integration
- MLOps components

Further validation and integration testing are being performed across the RAG retrieval layer, dashboard transaction investigation, and MLOps monitoring workflows.

---

## 👤 Author

**Gyarina Rauniyar**

B.Tech — Computer Science Engineering

GitHub:  
https://github.com/gyarinarauniyar

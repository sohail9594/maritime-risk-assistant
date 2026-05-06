# Maritime Risk Intelligence Assistant

An AI-powered maritime risk analysis platform that combines Retrieval-Augmented Generation (RAG), domain-specific legal and insurance knowledge, and Large Language Models (LLMs) to assist in maritime operational decision-making.

---

# Overview

The Maritime Risk Intelligence Assistant is designed to analyze maritime incidents, operational disruptions, environmental hazards, and logistics risks using a hybrid AI pipeline.

The system combines:

- Maritime law documents
- Insurance policy datasets
- Synthetic maritime incident datasets
- Maritime jargon and terminology databases
- LLM-powered reasoning
- Vector similarity search

to generate intelligent risk assessments and operational recommendations.

The project was developed as an academic AI/LLM proof-of-concept demonstrating how modern RAG systems can be applied to real-world maritime operations.

---

# Key Features

## AI-Powered Risk Assessment
Analyzes maritime incidents and produces structured operational recommendations.

## Retrieval-Augmented Generation (RAG)
Retrieves relevant legal, insurance, and operational context before generating responses.

## Maritime Law Integration
Supports maritime compliance analysis using:
- UAE Federal Maritime Law documents
- International maritime regulations
- Operational policy references

## Insurance Intelligence
Incorporates maritime insurance datasets to evaluate:
- Liability exposure
- Claim implications
- Coverage-related operational risks

## Synthetic Maritime Dataset
Uses 150+ synthetic maritime incident scenarios covering:
- Port disruptions
- GPS jamming
- Weather events
- Cargo delays
- Smart container failures
- Cybersecurity incidents
- Vessel routing conflicts

## Vector Semantic Search
Embeddings-based retrieval for domain-specific context matching.

## Conversational Chat Interface
Interactive Streamlit-based interface for querying maritime incidents naturally.

---

# System Architecture

```text
                    ┌────────────────────┐
                    │  User Query/Input  │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │  Streamlit Frontend│
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │  LangChain Pipeline│
                    └─────────┬──────────┘
                              │
          ┌───────────────────┴───────────────────┐
          ▼                                       ▼
┌────────────────────┐              ┌────────────────────┐
│ Vector Database     │              │ LLM Reasoning Layer│
│ (Embeddings Search) │              │ (Generation)       │
└─────────┬──────────┘              └─────────┬──────────┘
          │                                     │
          ▼                                     ▼
 ┌──────────────────┐              ┌────────────────────┐
 │ Maritime Laws    │              │ Risk Recommendations│
 │ Insurance Docs   │              │ Operational Output  │
 │ Synthetic Data   │              └────────────────────┘
 └──────────────────┘
```

---

# Tech Stack

## Frontend
- Python
- Streamlit

## Backend / AI Pipeline
- LangChain
- Retrieval-Augmented Generation (RAG)
- Vector Embeddings
- Semantic Search Pipeline

## LLMs / AI Models
- OpenAI-compatible LLM workflows
- Embedding models for vector retrieval
- AI reasoning pipeline for maritime risk analysis

## Data Processing
- CSV processing
- Document chunking
- Text embeddings
- Semantic indexing

## Knowledge Sources
- UAE Maritime Law documents
- International Maritime Law datasets
- Maritime insurance datasets
- Synthetic maritime operational datasets
- Maritime terminology/jargon datasets

---

# How RAG Works in This Project

The assistant follows a Retrieval-Augmented Generation pipeline:

1. User submits a maritime incident or operational scenario
2. Query is converted into vector embeddings
3. Relevant maritime documents are retrieved
4. Context is passed into the LLM
5. The LLM generates:
   - Risk analysis
   - Operational impact
   - Recommended actions
   - Legal/insurance considerations

This improves:
- Factual grounding
- Domain specificity
- Reliability of responses
- Context-aware recommendations

---

# Example Use Case

## Scenario
A regional GPS/Satellite jamming event disrupts communication with automated smart containers.

## Assistant Output
The system:
- Identifies probable operational causes
- Retrieves related maritime operational guidelines
- Assesses insurance implications
- Provides lower-risk operational recommendations
- Explains possible recovery actions

---

# Project Structure

```text
maritime-risk-assistant/
│
├── app.py
├── requirements.txt
├── data/
│   ├── synthetic_datasets/
│   ├── maritime_laws/
│   ├── insurance_docs/
│   └── terminology/
│
├── embeddings/
├── vector_store/
├── rag_pipeline/
├── utils/
└── README.md
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/sohail9594/maritime-risk-assistant.git
cd maritime-risk-assistant
```

## Create Virtual Environment

```bash
python -m venv venv
```

## Activate Environment

### macOS/Linux

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Run the Application

```bash
streamlit run app.py
```

The application will launch locally in your browser.

---

# Future Improvements

- Real-time AIS vessel tracking integration
- Live maritime weather APIs
- Advanced multi-agent AI workflows
- Fine-tuned maritime-specific LLM
- Voice-based incident reporting
- Predictive risk forecasting
- Multi-language maritime support

---

# Academic Purpose

This project was developed as an academic proof-of-concept demonstrating:
- Retrieval-Augmented Generation (RAG)
- AI-assisted decision support systems
- Domain-specific LLM applications
- Semantic document retrieval
- Maritime operational intelligence

---

# Contributors

- Sohail Mohammed
- Adnan
- Sulaiman

---

# PDF Assistant RAG with LangGraph & Gemini

An advanced Retrieval-Augmented Generation (RAG) system built using **Streamlit**, **LangGraph**, and **Google Gemini** for high-accuracy document Q&A and summarization.

This project goes beyond simple vector search by implementing a sophisticated pipeline including:
- **LlamaParse** for high-fidelity PDF parsing (OCR & Tables)
- **Query Expansion** to generate search variations
- **Re-ranking** results for higher relevance
- **LangGraph** state machine for workflow orchestration
- **Streamlit** UI for easy file management and chatting

---

## 🚀 Features

- **Advanced PDF Parsing**: Uses `LlamaParse` to accurately extract tables and text, even from scanned documents.
- **Agentic Workflow**: Built on `LangGraph` to manage the flow between Query Transformation, Retrieval, and Generation.
- **Smart Retrieval**: 
  - **Query Expansion**: Generates 3 variations of the user's question to find better matches.
  - **Reranking**: Uses Gemini to score and re-order search results before answering.
- **Summarization Engine**: Dedicated mode for generating Detailed, Concise, Bullet-point, or Executive summaries.
- **Persistent Storage**: Uses ChromaDB to save vector embeddings locally.
---

## 🏗️ Architecture

The system is split into two distinct execution branches:
### 1. The QA Branch (Precision)
`Transform Query` → `Vector Search (Recall)` → `Rerank (Precision)` → `Generate Answer`

### 2. The Summary Branch (Context)
`Router` → `Check Token Length` → **Decision**:
  - **Path A (Small Docs):** `Stuff Node` (Single Prompts)
  - **Path B (Large Docs):** `Refine Node` (Iterative Loop)


## 📂 Project Structure

- `app.py`: The Streamlit frontend interface.
- `agent_graph.py`: The LangGraph workflow definition (Transform -> Retrieve -> Answer).
- `vector_engine.py`: Handles LlamaParse ingestion, ChromaDB storage, and Reranking logic.
- `config.py`: Central configuration for API keys, model names, and chunking settings.
- `requirements.txt`: Python dependencies.

---

## ▶️ How to Run

Follow these steps to set up the environment and run the application.

### **1. Clone the Repository**
```bash
git clone https://github.com/rak-esh/AgenticRAG.git
cd AgenticRAG

## Setup
```bash
uv venv
.venv/bin/activate 
uv pip install -r requirements.txt
```
## Environment variables
Create a .env file and provide the following:
```bash
LLAMA_CLOUD_API_KEY=<>
GOOGLE_API_KEY=<>
```
## Entrypoint
```bash
streamlit run app.py
```


import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- API Keys ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY", "")

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_FOLDER = os.path.join(BASE_DIR, "uploaded_pdfs")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

# Create directories
os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(CHROMA_PATH, exist_ok=True)

# --- Model Configurations ---
EMBEDDING_MODEL_NAME = "models/gemini-embedding-001"
LLM_MODEL_NAME = "gemini-2.0-flash"  # Updated to latest flash model for speed

# --- Retrieval Configs ---
INITIAL_RETRIEVAL_K = 15

# Chunking Configs
CHUNK_CONFIGS = {
    "qa": {"size": 1000, "overlap": 100}
}

# --- Prompts ---
RERANK_PROMPT = """You are a highly intelligent relevance ranker.
Your task is to evaluate the following list of documents based on their relevance to the user's query.

Query: {query}

Documents:
{docs}

Instructions:
1. Analyze the semantic meaning of the query.
2. Rank the provided documents from MOST relevant to LEAST relevant.
3. Return the ID numbers of the top {k} documents in order, separated by commas.
4. Do not output anything else (no explanations, no intro).
5. If a document is completely irrelevant, do not include it.

Example Output: 2, 5, 1, 3
"""


SUMMARY_TEMPLATES = {
    "detailed": """You are a detailed summarization expert. Create a comprehensive summary of the following text.
    Capture main ideas, key supporting points, and logical flow.
    
    Text: {text}
    
    Detailed Summary:""",
    
    "concise": """You are a concise summarization expert. Create a brief, focused summary extracting only essential information.
    
    Text: {text}
    
    Concise Summary:""",
    
    "bullet": """You are a bullet-point summarization expert. Create a clear, structured bullet-point summary.
    Use top-level bullets for main points and sub-bullets for details.
    
    Text: {text}
    
    Bullet Point Summary:""",
    
    "executive": """You are an executive summary expert. Create a high-level summary focusing on strategic points, business impact, and key findings.
    
    Text: {text}
    
    Executive Summary:"""
}

QUERY_REWRITE_TEMPLATE = """You are an AI assistant specialized in information retrieval. 
Your task is to generate 3 different versions of the given user question to retrieve relevant documents from a vector database. 
By generating multiple perspectives, your goal is to help the user overcome limitations of distance-based similarity search.

Provide these alternative questions separated by newlines. Do not number them. Do not add any other text.

Original question: {question}"""

QA_SYSTEM_PROMPT = """You are a precise and helpful PDF assistant. Use the provided context to answer the user's question.
If the answer is not in the context, politely state that you cannot find the information in the provided documents.
Do not hallucinate information.

Context:
{context}"""

REFINE_THRESHOLD_CHARS = 100000 

REFINE_TEMPLATE = """
You are refining an existing summary of a document.
We have an existing summary up to a certain point:
{existing_summary}

Below is the new context to add:
{new_context}

Given the new context, refine the original summary.
If the new context isn't useful, return the original summary.
Refined Summary:
"""
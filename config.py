import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "") 

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_FOLDER = os.path.join(BASE_DIR, "uploaded_pdfs")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

# Create directories
os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(CHROMA_PATH, exist_ok=True)

# --- Model Configurations (Online) ---
# Using Google's latest text embedding model
EMBEDDING_MODEL_NAME = "models/text-embedding-004"
LLM_MODEL_NAME = "gemini-2.5-flash" 

# --- Retrieval Configs ---
# How many "rough" matches to fetch before reranking
INITIAL_RETRIEVAL_K = 15

# Chunking Configs
# Increased chunk size slightly as Gemini handles larger contexts well
CHUNK_CONFIGS = {
    "qa": {"size": 1000, "overlap": 100},
    "summary": {"size": 8000, "overlap": 400}
}

# --- Prompts ---

# 1. Reranking Prompt (The Judge)
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

# 2. Summarization Prompts
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

# 3. Query Expansion Prompt
QUERY_REWRITE_TEMPLATE = """You are an AI assistant specialized in information retrieval. 
Your task is to generate 3 different versions of the given user question to retrieve relevant documents from a vector database. 
By generating multiple perspectives, your goal is to help the user overcome limitations of distance-based similarity search.

Provide these alternative questions separated by newlines. Do not number them. Do not add any other text.

Original question: {question}"""

# 4. QA Generation Prompt
QA_SYSTEM_PROMPT = """You are a precise and helpful PDF assistant. Use the provided context to answer the user's question.
If the answer is not in the context, politely state that you cannot find the information in the provided documents.
Do not hallucinate information.

Context:
{context}"""
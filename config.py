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
MODEL_CACHE_DIR = os.path.join(BASE_DIR, "model_cache")

# --- CRITICAL FIX: Set Cache Globally via Env Var ---
# This must be set before importing transformers/sentence_transformers
os.environ["HF_HOME"] = MODEL_CACHE_DIR
os.environ["SENTENCE_TRANSFORMERS_HOME"] = MODEL_CACHE_DIR

# Create directories
os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(CHROMA_PATH, exist_ok=True)
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# Model Configurations
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL_NAME = "gemini-2.5-flash" 

# Chunking Configs
CHUNK_CONFIGS = {
    "qa": {"size": 500, "overlap": 50},
    "summary": {"size": 4000, "overlap": 200}
}

# Prompts
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
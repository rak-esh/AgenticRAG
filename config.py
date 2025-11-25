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
EMBEDDING_MODEL_NAME = "models/text-embedding-004"
LLM_MODEL_NAME = "gemini-2.5-flash" 

# --- Retrieval Configs ---
INITIAL_RETRIEVAL_K = 15

# Chunking Configs
CHUNK_CONFIGS = {
    "qa": {"size": 1000, "overlap": 100},
    "summary": {"size": 8000, "overlap": 400}
}

# --- Prompts ---

# 1. Multimodal Summarization Prompt (For Tables/Images)
# Inspired by your notebook to create "searchable descriptions"
MULTIMODAL_SUMMARY_PROMPT = """You are an AI assistant specialized in summarizing rich document content for retrieval.
Your task is to create a concise, searchable description of the following content (Table or Image context).

Content:
{content}

Instructions:
1. Identify the key data points, trends, column headers, and relationships.
2. Generate a summary that includes the most important keywords a user might search for.
3. Keep it concise but comprehensive.

Searchable Summary:"""

# 2. Reranking Prompt
RERANK_PROMPT = """You are a highly intelligent relevance ranker.
Your task is to evaluate the following list of documents based on their relevance to the user's query.

Query: {query}

Documents:
{docs}

Instructions:
1. Analyze the semantic meaning of the query.
2. Rank the provided documents from MOST relevant to LEAST relevant.
3. Return the ID numbers of the top {k} documents in order, separated by commas.
4. Do not output anything else.

Example Output: 2, 5, 1, 3
"""

# 3. General Prompts
SUMMARY_TEMPLATES = {
    "detailed": "You are a detailed summarization expert. Summarize: {text}",
    "concise": "You are a concise summarization expert. Summarize: {text}",
    "bullet": "You are a bullet-point summarization expert. Summarize: {text}",
    "executive": "You are an executive summary expert. Summarize: {text}"
}

QUERY_REWRITE_TEMPLATE = """Generate 3 different search queries based on this user question to improve retrieval.
Original question: {question}"""

QA_SYSTEM_PROMPT = """You are a helpful assistant. Use the context provided to answer the question.
Context:
{context}"""
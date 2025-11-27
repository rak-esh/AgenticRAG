import os
import logging
from typing import List, Dict
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import chromadb
import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorEngine:
    def __init__(self):
        print("Initializing Vector Engine (Online Gemini + LLM Reranking)...")
        
        if not config.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not found in environment variables.")

        # 1. Embedding Model (Vector Search)
        self.embedding_model = GoogleGenerativeAIEmbeddings(
            model=config.EMBEDDING_MODEL_NAME,
            google_api_key=config.GOOGLE_API_KEY
        )

        # 2. LLM for Reranking (The Judge)
        # We use a low temperature for strict logical ranking
        self.rerank_llm = ChatGoogleGenerativeAI(
            model=config.LLM_MODEL_NAME,
            google_api_key=config.GOOGLE_API_KEY,
            temperature=0.0 
        )
        
        self.client = chromadb.PersistentClient(path=config.CHROMA_PATH)
        print("Vector Engine Initialized Successfully.")

    def _get_collection(self, name: str):
        return self.client.get_or_create_collection(name=name)

    def process_and_store_pdf(self, file_path: str, filename: str):
        text = self._extract_text(file_path)
        if not text:
            logger.warning(f"No text extracted from {filename}")
            return False

        # 1. Process QA Chunks
        self._chunk_and_store(text, filename, config.CHUNK_CONFIGS["qa"], "qa_chunks")

        # 2. Process Summary Chunks
        self._chunk_and_store(text, filename, config.CHUNK_CONFIGS["summary"], "summary_chunks")
        
        return True

    def _extract_text(self, path: str) -> str:
        try:
            reader = PdfReader(path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            logger.error(f"Error reading PDF {path}: {e}")
            return ""

    def _chunk_and_store(self, text: str, filename: str, config_dict: Dict, collection_name: str):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config_dict["size"],
            chunk_overlap=config_dict["overlap"],
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = splitter.split_text(text)
        
        if not chunks:
            return

        # Embed using Gemini API
        embeddings = self.embedding_model.embed_documents(chunks)
        ids = [f"{filename}_{collection_name}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": filename, "type": collection_name} for _ in chunks]

        collection = self._get_collection(collection_name)
        try:
            collection.delete(where={"source": filename})
        except:
            pass

        collection.add(
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        logger.info(f"Stored {len(chunks)} chunks in {collection_name} for {filename}")

    def retrieve_refined(self, original_query: str, generated_queries: List[str], k: int = 5, file_filter: str = None) -> List[Dict]:
        """
        1. Vector Search (Broad): Get top 15-20 candidates using Embedding Model.
        2. Deduplicate.
        3. LLM Reranking (Deep): Ask Gemini to rank them.
        """
        collection = self._get_collection("qa_chunks")
        
        all_queries = [original_query] + generated_queries
        
        # 1. Broad Vector Search
        initial_k = config.INITIAL_RETRIEVAL_K
        where_clause = {"source": file_filter} if file_filter and file_filter != "All PDFs" else None
        
        unique_docs = {} 
        
        # Gather candidates
        for q in all_queries:
            query_embedding = self.embedding_model.embed_query(q)
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=initial_k,
                where=where_clause
            )
            
            if results['documents']:
                for i, doc_content in enumerate(results['documents'][0]):
                    if doc_content not in unique_docs:
                        unique_docs[doc_content] = results['metadatas'][0][i]

        if not unique_docs:
            return []

        # Convert to list for reranking
        candidates = []
        for content, meta in unique_docs.items():
            candidates.append({
                "content": content,
                "source": meta['source']
            })

        # 2. LLM Reranking
        print(f"--- RERANKING {len(candidates)} CANDIDATES WITH GEMINI ---")
        try:
            reranked_results = self._rerank_with_gemini(original_query, candidates, k)
            return reranked_results
        except Exception as e:
            print(f"Reranking failed: {e}. Falling back to random vector order.")
            # Fallback: just return the first k unique vector results
            return candidates[:k]

    def _rerank_with_gemini(self, query: str, docs: List[Dict], k: int) -> List[Dict]:
        # Format documents for the prompt
        doc_text = ""
        for i, doc in enumerate(docs):
            # We give each doc an ID (0, 1, 2...)
            doc_text += f"[{i}] {doc['content']}\n\n"

        # Construct Prompt
        prompt = config.RERANK_PROMPT.format(
            query=query,
            docs=doc_text,
            k=k
        )

        # Call Gemini
        response = self.rerank_llm.invoke([HumanMessage(content=prompt)])
        content = response.content.strip()
        print(f"Gemini Rerank Output: {content}")

        # Parse Output (Expected format: "2, 0, 5, ...")
        try:
            # Extract numbers from comma-separated string
            top_indices_str = content.split(',')
            top_indices = [int(idx.strip()) for idx in top_indices_str if idx.strip().isdigit()]
            
            final_results = []
            for idx in top_indices:
                if 0 <= idx < len(docs):
                    final_results.append(docs[idx])
            
            # If model returned fewer than k (or 0), fill with remaining vector results
            if len(final_results) < k:
                seen_contents = set(d['content'] for d in final_results)
                for doc in docs:
                    if len(final_results) >= k:
                        break
                    if doc['content'] not in seen_contents:
                        final_results.append(doc)

            return final_results

        except Exception as e:
            print(f"Error parsing rerank response: {e}")
            return docs[:k]

    def get_all_summary_chunks(self, filename: str) -> str:
        collection = self._get_collection("summary_chunks")
        results = collection.get(where={"source": filename})
        if results and results['documents']:
            return "\n\n".join(results['documents'])
        return ""

    def get_existing_files(self) -> List[str]:
        return [f for f in os.listdir(config.PDF_FOLDER) if f.endswith('.pdf')]
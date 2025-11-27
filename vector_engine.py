import os
import logging
import nest_asyncio
import pandas as pd
import json
from typing import List, Dict
import chromadb
from concurrent.futures import ThreadPoolExecutor # <--- Added for Parallelization
import config

# --- IMPORTS ---
from llama_parse import LlamaParse
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# Apply nest_asyncio
nest_asyncio.apply()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorEngine:
    def __init__(self):
        print("Initializing Vector Engine (Online Gemini + Multimodal LlamaParse)...")
        
        # Check Keys
        if not config.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not found.")
        if not config.LLAMA_CLOUD_API_KEY:
             logger.warning("⚠️ LLAMA_CLOUD_API_KEY not found. PDF processing will fail.")

        # 1. Embedding Model
        self.embedding_model = GoogleGenerativeAIEmbeddings(
            model=config.EMBEDDING_MODEL_NAME,
            google_api_key=config.GOOGLE_API_KEY
        )

        # 2. Multimodal LLM (Gemini 1.5/2.0 Flash is Vision-capable)
        self.llm = ChatGoogleGenerativeAI(
            model=config.LLM_MODEL_NAME,
            google_api_key=config.GOOGLE_API_KEY,
            temperature=0.0 
        )
        
        # 3. Initialize LlamaParse
        self.parser = LlamaParse(
            api_key=config.LLAMA_CLOUD_API_KEY,
            result_type="json", 
            verbose=True,
            parse_mode="parse_page_with_lvm",
            model="openai-gpt-4o-mini",       
            adaptive_long_table=True,
            outlined_table_extraction=True,
            output_tables_as_HTML=True,
        )
        
        self.client = chromadb.PersistentClient(path=config.CHROMA_PATH)
        print("Vector Engine Initialized Successfully.")

    def _get_collection(self, name: str):
        return self.client.get_or_create_collection(name=name)

    def process_and_store_pdf(self, file_path: str, filename: str):
        text = self._extract_content_with_multimodal(file_path)
        
        if not text:
            logger.warning(f"No content extracted from {filename}")
            return False

        # 1. Store Vectors for Q&A (ChromaDB)
        self._chunk_and_store(text, filename, config.CHUNK_CONFIGS["qa"], "qa_chunks")

        # 2. Store Full Text for Summarization (Local Markdown File)
        # This skips embedding costs for the summary track
        md_path = os.path.join(config.PDF_FOLDER, f"{filename}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(text)
        
        logger.info(f"Saved full markdown for summarization to {md_path}")
        return True

    def _extract_content_with_multimodal(self, file_path: str) -> str:
        """
        Extracts Text, Tables, AND Images.
        """
        logger.info(f"Sending {file_path} to LlamaParse (Multimodal)...")
        
        try:
            # 1. Get JSON Result
            json_result = self.parser.get_json_result(file_path)
            full_document_markdown = ""

            for doc in json_result:
                pages = doc.get("pages", [])
                for page in pages:
                    items = page.get("items", [])
                    
                    for item in items:
                        item_type = item.get("type", "").lower()
                        
                        # --- A. TABLES (Reconstruct using Pandas) ---
                        if item_type == "table":
                            rows = item.get("rows")
                            if rows:
                                try:
                                    df = pd.DataFrame(rows[1:], columns=rows[0])
                                    table_md = df.to_markdown(index=False)
                                    full_document_markdown += f"\n\n[TABLE]\n{table_md}\n\n"
                                except:
                                    # Fallback if pandas fails
                                    full_document_markdown += f"\n\n{item.get('md', '')}\n\n"
                        else:
                            md_content = item.get("md", "")
                            full_document_markdown += md_content + "\n"
                    
                    full_document_markdown += "\n---\n" # Page separator

            return full_document_markdown

        except Exception as e:
            logger.error(f"LlamaParse Error: {e}")
            return ""

    def _chunk_and_store(self, text: str, filename: str, config_dict: Dict, collection_name: str):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config_dict["size"],
            chunk_overlap=config_dict["overlap"],
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = splitter.split_text(text)
        
        if not chunks: return

        embeddings = self.embedding_model.embed_documents(chunks)
        ids = [f"{filename}_{collection_name}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": filename, "type": collection_name} for _ in chunks]

        collection = self._get_collection(collection_name)
        try: collection.delete(where={"source": filename})
        except: pass

        collection.add(documents=chunks, embeddings=embeddings, metadatas=metadatas, ids=ids)
        logger.info(f"Stored {len(chunks)} chunks in {collection_name} for {filename}")

    def retrieve_refined(self, original_query: str, generated_queries: List[str], k: int = 5, file_filter: str = None) -> List[Dict]:
        """
        PARALLELIZED RETRIEVAL
        Fires 4 embedding + search calls simultaneously using ThreadPoolExecutor.
        """
        collection = self._get_collection("qa_chunks")
        all_queries = [original_query] + generated_queries
        initial_k = config.INITIAL_RETRIEVAL_K
        where_clause = {"source": file_filter} if file_filter and file_filter != "All PDFs" else None
        unique_docs = {} 

        # --- Helper for Parallel Execution ---
        def search_single_query(query_text):
            # 1. Network Call (Embed)
            query_embedding = self.embedding_model.embed_query(query_text)
            # 2. DB Call (Search)
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=initial_k,
                where=where_clause
            )
            return results

        # --- Execute in Parallel ---
        # We use max_workers=5 to safely cover our ~4 queries
        print(f"--- PARALLEL SEARCHING FOR {len(all_queries)} QUERIES ---")
        with ThreadPoolExecutor(max_workers=5) as executor:
            # .map() maintains the order of results corresponding to all_queries
            results_list = list(executor.map(search_single_query, all_queries))

        # --- Aggregation & Deduplication ---
        for results in results_list:
            if results['documents']:
                for i, doc_content in enumerate(results['documents'][0]):
                    if doc_content not in unique_docs:
                        unique_docs[doc_content] = results['metadatas'][0][i]

        if not unique_docs: return []

        candidates = [{"content": content, "source": meta['source']} for content, meta in unique_docs.items()]
        
        # Rerank
        print(f"--- RERANKING {len(candidates)} CANDIDATES WITH GEMINI ---")
        return self._rerank_with_gemini(original_query, candidates, k)

    def _rerank_with_gemini(self, query: str, docs: List[Dict], k: int) -> List[Dict]:
        doc_text = ""
        for i, doc in enumerate(docs):
            doc_text += f"[{i}] {doc['content']}\n\n"
        
        prompt = config.RERANK_PROMPT.format(query=query, docs=doc_text, k=k)
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content.strip()
            # Robust parsing of LLM output "1, 3, 5"
            top_indices = [int(idx.strip()) for idx in content.split(',') if idx.strip().isdigit()]
            final_results = []
            
            for idx in top_indices:
                if 0 <= idx < len(docs):
                    final_results.append(docs[idx])
            
            # Backfill if LLM is too strict
            if len(final_results) < k:
                seen_contents = set(d['content'] for d in final_results)
                for doc in docs:
                    if len(final_results) >= k: break
                    if doc['content'] not in seen_contents: final_results.append(doc)
            return final_results
        except Exception:
            # Fallback to vector ranking if LLM fails
            return docs[:k]

    def get_all_summary_chunks(self, filename: str) -> str:
        """
        Reads the full Markdown content directly from the local file.
        This is zero-latency and cost-free compared to vector retrieval.
        """
        md_path = os.path.join(config.PDF_FOLDER, f"{filename}.md")
        
        if os.path.exists(md_path):
            with open(md_path, "r", encoding="utf-8") as f:
                return f.read()
        else:
            logger.warning(f"Summary markdown file not found for {filename}")
            return ""

    def get_existing_files(self) -> List[str]:
        return [f for f in os.listdir(config.PDF_FOLDER) if f.endswith('.pdf')]
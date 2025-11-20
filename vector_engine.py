import os
import logging
from typing import List, Dict
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
import chromadb
import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorEngine:
    def __init__(self):
        print("Initializing Vector Engine...")
        # 1. Embedding Model (Bi-Encoder)
        # Relies on os.environ["HF_HOME"] from config.py
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL_NAME
        )
        
        # 2. Reranker Model (Cross-Encoder)
        # FIXED: No extra arguments passed here to avoid TypeError
        self.reranker = CrossEncoder(
            config.RERANKER_MODEL_NAME
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
        collection = self._get_collection("qa_chunks")
        
        all_queries = [original_query] + generated_queries
        initial_k = k * 3
        where_clause = {"source": file_filter} if file_filter and file_filter != "All PDFs" else None
        
        unique_docs = {} 
        
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

        doc_contents = list(unique_docs.keys())
        doc_metadatas = list(unique_docs.values())
        
        # Reranking
        pairs = [[original_query, doc] for doc in doc_contents]
        scores = self.reranker.predict(pairs)
        
        scored_docs = sorted(
            zip(doc_contents, doc_metadatas, scores), 
            key=lambda x: x[2], 
            reverse=True
        )
        
        final_results = []
        for doc, meta, score in scored_docs[:k]:
            final_results.append({
                "content": doc,
                "source": meta['source'],
                "score": float(score)
            })
            
        return final_results

    def get_all_summary_chunks(self, filename: str) -> str:
        collection = self._get_collection("summary_chunks")
        results = collection.get(where={"source": filename})
        if results and results['documents']:
            return "\n\n".join(results['documents'])
        return ""

    def get_existing_files(self) -> List[str]:
        return [f for f in os.listdir(config.PDF_FOLDER) if f.endswith('.pdf')]
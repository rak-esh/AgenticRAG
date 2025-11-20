This system implements a sophisticated pipeline designed to minimize hallucinations and maximize retrieval accuracy.
1. Data Ingestion (Dual-Chunking)
	• QA Chunks (Small): ~500 characters. Optimized for precise fact retrieval.
	• Summary Chunks (Large): ~4000 characters. Optimized for capturing global context.
	• Vector Store: ChromaDB (Persistent storage).
2. The Intelligence Loop (LangGraph)
When a user asks a question, the State Graph executes the following workflow:
	1. Router: Determines if the intent is Question Answering or Summarization.
	2. Query Translation: Uses LLM to generate 3 variations of the user's query (improves recall).
	3. Broad Retrieval: Fetches top ~15 documents across all query variations.
	4. Reranking (Cross-Encoder): Uses ms-marco-MiniLM-L-6-v2 to score the relevancy of retrieved chunks against the original question.
	5. Generation: Sends only the top 5 verified chunks to Gemini 2.5 Flash for the final answer.
<img width="895" height="401" alt="image" src="https://github.com/user-attachments/assets/d93667d8-8951-4a47-8d27-90f4c9e260d0" />

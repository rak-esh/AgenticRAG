import os
import sys
from typing import TypedDict, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
import config

# Import VectorEngine with Error Handling
try:
    from vector_engine import VectorEngine
    print("Initializing Vector Engine in Graph...")
    vector_engine = VectorEngine()
except Exception as e:
    print(f"🔥 CRITICAL ERROR: Could not initialize VectorEngine: {e}")
    # We allow the script to continue so the import doesn't fail entirely, 
    # but the graph will fail if run.
    vector_engine = None

# Setup Gemini
llm = ChatGoogleGenerativeAI(
    model=config.LLM_MODEL_NAME,
    google_api_key=config.GOOGLE_API_KEY,
    temperature=0.3,
    convert_system_message_to_human=True
)

# --- State Definitions ---
class AgentState(TypedDict):
    question: str
    generated_queries: List[str]
    file_filter: str
    summary_type: str
    mode: str 
    context: List[str]
    answer: str
    source_docs: List[str]

# --- Nodes ---

def transform_query_node(state: AgentState):
    print("--- TRANSFORMING QUERY ---")
    question = state["question"]
    
    # Use LLM to generate variations
    prompt = config.QUERY_REWRITE_TEMPLATE.format(question=question)
    msg = [HumanMessage(content=prompt)]
    
    try:
        response = llm.invoke(msg)
        queries = [q.strip() for q in response.content.split('\n') if q.strip()]
    except Exception as e:
        print(f"Query translation failed: {e}")
        queries = []

    return {"generated_queries": queries}

def retrieve_node(state: AgentState):
    print("--- RETRIEVING DOCUMENTS ---")
    if vector_engine is None:
        return {"context": [], "source_docs": []}

    question = state["question"]
    generated_queries = state.get("generated_queries", [])
    file_filter = state.get("file_filter")
    
    results = vector_engine.retrieve_refined(
        original_query=question,
        generated_queries=generated_queries,
        k=5,
        file_filter=file_filter
    )
    
    context_text = [r["content"] for r in results]
    sources = list(set([r["source"] for r in results]))
    
    return {"context": context_text, "source_docs": sources}

def generate_qa_node(state: AgentState):
    print("--- GENERATING ANSWER ---")
    question = state["question"]
    context = "\n\n".join(state["context"])
    
    if not context:
        return {"answer": "I couldn't find any relevant information in the uploaded documents."}

    system_prompt = config.QA_SYSTEM_PROMPT.format(context=context)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Question: {question}")
    ]
    
    response = llm.invoke(messages)
    return {"answer": response.content}

def summarize_node(state: AgentState):
    print("--- GENERATING SUMMARY ---")
    if vector_engine is None:
        return {"answer": "Vector Engine failed to initialize."}

    filename = state["file_filter"]
    summary_type = state["summary_type"]
    
    full_text = vector_engine.get_all_summary_chunks(filename)
    
    if not full_text:
        return {"answer": "Could not find processed text for this file."}

    template = config.SUMMARY_TEMPLATES.get(summary_type, config.SUMMARY_TEMPLATES["detailed"])
    prompt = template.format(text=full_text)
    
    messages = [HumanMessage(content=prompt)]
    
    try:
        response = llm.invoke(messages)
        return {"answer": response.content}
    except Exception as e:
        return {"answer": f"Error generating summary: {str(e)}"}

# --- Graph Construction ---

def route_workflow(state: AgentState):
    if state["mode"] == "summarize":
        return "summarize"
    return "transform"

workflow = StateGraph(AgentState)

workflow.add_node("transform", transform_query_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate_qa", generate_qa_node)
workflow.add_node("summarize", summarize_node)

workflow.add_edge("transform", "retrieve")
workflow.add_edge("retrieve", "generate_qa")
workflow.add_edge("generate_qa", END)
workflow.add_edge("summarize", END)

workflow.set_conditional_entry_point(
    route_workflow,
    {
        "transform": "transform",
        "summarize": "summarize"
    }
)

# Compile the graph
app_graph = workflow.compile()
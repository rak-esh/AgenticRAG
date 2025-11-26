import sys
from typing import TypedDict, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
import config

# --- INITIALIZE ENGINE (Singleton) ---
try:
    from vector_engine import VectorEngine
    vector_engine = VectorEngine() # <--- Created ONCE here
except Exception as e:
    print(f"Engine Init Error: {e}")
    vector_engine = None

# Setup Gemini for QA
llm = ChatGoogleGenerativeAI(
    model=config.LLM_MODEL_NAME,
    google_api_key=config.GOOGLE_API_KEY,
    temperature=0
)

# --- State ---
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
    print("--- TRANSFORM ---")
    question = state["question"]
    prompt = config.QUERY_REWRITE_TEMPLATE.format(question=question)
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        queries = [q.strip() for q in response.content.split('\n') if q.strip()]
    except:
        queries = []
    return {"generated_queries": queries}

def retrieve_node(state: AgentState):
    print("--- RETRIEVE ---")
    if not vector_engine: return {"context": [], "source_docs": []}

    results = vector_engine.retrieve_refined(
        original_query=state["question"],
        generated_queries=state.get("generated_queries", []),
        k=5,
        file_filter=state.get("file_filter")
    )
    return {
        "context": [r["content"] for r in results],
        "source_docs": list(set([r["source"] for r in results]))
    }

def generate_qa_node(state: AgentState):
    print("--- GENERATE ---")
    context = "\n\n".join(state["context"])
    if not context:
        return {"answer": "No relevant info found in documents."}

    system_prompt = config.QA_SYSTEM_PROMPT.format(context=context)
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=state["question"])
    ])
    return {"answer": response.content}

def summarize_node(state: AgentState):
    print("--- SUMMARIZE ---")
    if not vector_engine: return {"answer": "Engine Error"}
    
    text = vector_engine.get_all_summary_chunks(state["file_filter"])
    if not text: return {"answer": "No text found for this file."}

    template = config.SUMMARY_TEMPLATES.get(state["summary_type"], config.SUMMARY_TEMPLATES["detailed"])
    response = llm.invoke([HumanMessage(content=template.format(text=text))])
    return {"answer": response.content}

def route_workflow(state: AgentState):
    return "summarize" if state["mode"] == "summarize" else "transform"

# --- Graph ---
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
    {"transform": "transform", "summarize": "summarize"}
)

app_graph = workflow.compile()
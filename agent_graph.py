import sys
from typing import TypedDict, List, Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter # <--- Vital for Refine Node
from langgraph.graph import StateGraph, END
import config

# --- INITIALIZE ENGINE (Singleton) ---
try:
    from vector_engine import VectorEngine
    vector_engine = VectorEngine() 
except Exception as e:
    print(f"Engine Init Error: {e}")
    vector_engine = None

# Setup Gemini
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

# --- QA NODES (Left Branch) ---

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

# --- SUMMARY NODES (Right Branch - Adaptive) ---

def route_summary_logic(state: AgentState) -> Literal["stuff_summary", "refine_summary"]:
    """
    Router Logic: Decides strategy based on document length.
    """
    print("--- CHECKING DOCUMENT SIZE ---")
    if not vector_engine: return "stuff_summary"

    # Fetches the Markdown text directly from disk (Fast)
    full_text = vector_engine.get_all_summary_chunks(state["file_filter"])
    
    # Check length against threshold (e.g., 100k chars)
    if len(full_text) > config.REFINE_THRESHOLD_CHARS:
        print(f"--- ROUTING TO REFINE (Length: {len(full_text)} chars) ---")
        return "refine_summary"
    else:
        print(f"--- ROUTING TO STUFF (Length: {len(full_text)} chars) ---")
        return "stuff_summary"

def stuff_summary_node(state: AgentState):
    """
    Strategy 1: Stuff (Fast)
    """
    print("--- SUMMARIZE (STUFF) ---")
    if not vector_engine: return {"answer": "Engine Error"}
    
    text = vector_engine.get_all_summary_chunks(state["file_filter"])
    if not text: return {"answer": "No text found for this file."}

    template = config.SUMMARY_TEMPLATES.get(state["summary_type"], config.SUMMARY_TEMPLATES["detailed"])
    
    try:
        response = llm.invoke([HumanMessage(content=template.format(text=text))])
        return {"answer": response.content}
    except Exception as e:
        return {"answer": f"Error: {str(e)}"}

def refine_summary_node(state: AgentState):
    """
    Strategy 2: Refine Chain (Robust for Large Docs)
    """
    print("--- SUMMARIZE (REFINE CHAIN) ---")
    if not vector_engine: return {"answer": "Engine Error"}
    
    full_text = vector_engine.get_all_summary_chunks(state["file_filter"])
    if not full_text: return {"answer": "No text found for this file."}
    
    # 1. Split text into 20k char chunks (Safe for Gemini Flash)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=20000, 
        chunk_overlap=1000,
        separators=["\n## ", "\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_text(full_text)
    print(f"--- SPLIT DOC INTO {len(chunks)} CHUNKS FOR REFINEMENT ---")

    # 2. Step 1: Initial Summary
    initial_template = config.SUMMARY_TEMPLATES.get(state["summary_type"], config.SUMMARY_TEMPLATES["detailed"])
    print("--- REFINING: CHUNK 1 ---")
    current_summary = llm.invoke([
        HumanMessage(content=initial_template.format(text=chunks[0]))
    ]).content

    # 3. Step 2..N: Iterative Refinement
    refine_template = config.REFINE_TEMPLATE
    for i, chunk in enumerate(chunks[1:]):
        print(f"--- REFINING: CHUNK {i+2}/{len(chunks)} ---")
        prompt = refine_template.format(
            existing_summary=current_summary,
            new_context=chunk
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        current_summary = response.content
        
    return {"answer": current_summary}

# --- GRAPH CONSTRUCTION ---

def route_main_workflow(state: AgentState):
    """
    Primary Router: QA vs Summarize
    """
    if state["mode"] == "summarize":
        return route_summary_logic(state) # <--- Calls the adaptive router
    else:
        return "transform"

workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("transform", transform_query_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate_qa", generate_qa_node)
workflow.add_node("stuff_summary", stuff_summary_node)
workflow.add_node("refine_summary", refine_summary_node)

# Edges: QA
workflow.add_edge("transform", "retrieve")
workflow.add_edge("retrieve", "generate_qa")
workflow.add_edge("generate_qa", END)

# Edges: Summary
workflow.add_edge("stuff_summary", END)
workflow.add_edge("refine_summary", END)

# Entry Point (The "Start" Node in your diagram)
workflow.set_conditional_entry_point(
    route_main_workflow,
    {
        "transform": "transform",
        "stuff_summary": "stuff_summary",
        "refine_summary": "refine_summary"
    }
)

app_graph = workflow.compile()
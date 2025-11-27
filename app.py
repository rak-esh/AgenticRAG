import streamlit as st
import os
import time
import config

# --- IMPORT GRAPH & ENGINE ---
# We import vector_engine from agent_graph so we share the SAME instance
from agent_graph import app_graph, vector_engine

st.set_page_config(page_title="Agentic RAG Assistant", layout="wide")

st.markdown("""
    <style>
    .stButton>button { width: 100%; height: 3em; }
    .chat-message { padding: 1.5rem; margin-bottom: 1rem; border-radius: 0.5rem; }
    .chat-message.user { background-color: #2b313e; color: #fff; }
    .chat-message.bot { background-color: #f0f2f6; }
    </style>
""", unsafe_allow_html=True)

def main():
    if not vector_engine:
        st.error("Vector Engine failed to initialize. Check API keys.")
        st.stop()

    with st.sidebar:
        st.title("🎛️ Controls")
        mode = st.radio("Mode", ["Chat Q&A", "Summarization", "Manage Files"])
        st.markdown("---")
        st.markdown("### 📂 Files")
        files = vector_engine.get_existing_files()
        for f in files: st.text(f"• {f}")

    # --- UPLOAD ---
    if mode == "Manage Files":
        uploaded_files = st.file_uploader("Upload PDFs", type=['pdf'], accept_multiple_files=True)
        if uploaded_files and st.button("Process Files"):
            bar = st.progress(0)
            start_time = time.time() # <--- Start Timer
            
            for i, file in enumerate(uploaded_files):
                path = os.path.join(config.PDF_FOLDER, file.name)
                with open(path, "wb") as f: f.write(file.getvalue())
                
                # Use the shared engine instance
                vector_engine.process_and_store_pdf(path, file.name)
                bar.progress((i + 1) / len(uploaded_files))
            
            end_time = time.time() # <--- End Timer
            st.success(f"Processing Complete in {end_time - start_time:.2f}s!")
            time.sleep(1)
            st.rerun()

    # --- CHAT ---
    elif mode == "Chat Q&A":
        file_options = ["All PDFs"] + files
        selected_file = st.selectbox("Search Context:", file_options)
        
        if "messages" not in st.session_state: st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "sources" in msg: st.caption(f"Sources: {msg['sources']}")
                if "time" in msg: st.caption(f"⏱️ {msg['time']}s") # <--- Display stored time

        if prompt := st.chat_input("Ask a question..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    start_ts = time.time() # <--- Start Timer
                    
                    result = app_graph.invoke({
                        "question": prompt,
                        "file_filter": selected_file,
                        "mode": "qa",
                        "summary_type": "", 
                        "context": [], "answer": "", "source_docs": []
                    })
                    
                    end_ts = time.time() # <--- End Timer
                    elapsed = round(end_ts - start_ts, 2)

                    response = result["answer"]
                    sources = result.get("source_docs", [])
                    
                    st.markdown(response)
                    if sources: st.caption(f"Sources: {', '.join(sources)}")
                    st.caption(f"⏱️ Response generated in {elapsed} seconds") # <--- Display Time
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response, 
                        "sources": str(sources),
                        "time": elapsed # <--- Save to history
                    })

    # --- SUMMARY ---
    elif mode == "Summarization":
        if not files: st.warning("Upload files first.")
        else:
            target = st.selectbox("Document", files)
            sType = st.selectbox("Type", ["detailed", "concise", "bullet", "executive"])
            
            if st.button("Summarize"):
                with st.spinner("Summarizing..."):
                    start_ts = time.time() # <--- Start Timer
                    
                    res = app_graph.invoke({
                        "question": "", "file_filter": target, 
                        "mode": "summarize", "summary_type": sType,
                        "context": [], "answer": "", "source_docs": []
                    })
                    
                    end_ts = time.time() # <--- End Timer
                    elapsed = round(end_ts - start_ts, 2)
                    
                    st.info(res["answer"])
                    st.success(f"✅ Summary generated in {elapsed} seconds") # <--- Display Time

if __name__ == "__main__":
    main()
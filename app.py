import streamlit as st
import os
import time  # Ensure time is imported
import config
from agent_graph import app_graph
from vector_engine import VectorEngine

# Initialize Engine
# Note: In Streamlit, it's often better to cache this resource.
engine = VectorEngine()

st.set_page_config(page_title="Gemini PDF Assistant", page_icon="🤖", layout="wide")

# CSS for production look
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
    }
    .reportview-container {
        background: #f0f2f6;
    }
    .chat-message {
        padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
    }
    .chat-message.user {
        background-color: #2b313e; color: #ffffff;
    }
    .chat-message.bot {
        background-color: #f0f2f6; border: 1px solid #dee2e6;
    }
    </style>
""", unsafe_allow_html=True)

def main():
    # Sidebar Configuration
    with st.sidebar:
        st.title("🎛️ Controls")
        
        # API Key Check
        if not config.GOOGLE_API_KEY:
            st.error("⚠️ Google API Key not found!")
            st.info("Set GOOGLE_API_KEY in .env or config.py")
            st.stop()

        mode = st.radio("Select Mode", ["Chat Q&A", "Document Summarization", "Manage Files"])
        
        st.markdown("---")
        st.markdown("### 📂 Available Files")
        
        files = engine.get_existing_files()
        if files:
            for f in files:
                st.text(f"• {f}")
        else:
            st.text("No files uploaded.")

    # --- Mode: Manage Files ---
    if mode == "Manage Files":
        st.header("📤 Document Management")
        
        uploaded_files = st.file_uploader(
            "Upload PDFs", 
            type=['pdf'], 
            accept_multiple_files=True
        )
        
        if uploaded_files:
            if st.button("Process Uploaded Files"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, file in enumerate(uploaded_files):
                    status_text.text(f"Processing {file.name}...")
                    
                    # Save file locally
                    file_path = os.path.join(config.PDF_FOLDER, file.name)
                    with open(file_path, "wb") as f:
                        f.write(file.getvalue())
                    
                    # Process into Vector DB
                    success = engine.process_and_store_pdf(file_path, file.name)
                    
                    if success:
                        st.success(f"✅ Processed: {file.name}")
                    else:
                        st.error(f"❌ Failed: {file.name}")
                    
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                status_text.text("All files processed!")
                time.sleep(1)
                st.rerun()

    # --- Mode: Chat Q&A ---
    elif mode == "Chat Q&A":
        st.header("💬 Ask Questions")
        
        # File Filter
        file_options = ["All PDFs"] + (files if files else [])
        selected_file = st.selectbox("Search Context:", file_options)
        
        # Initialize Chat History
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display Chat
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "sources" in msg and msg["sources"]:
                    st.caption(f"Sources: {', '.join(msg['sources'])}")
                if "time_taken" in msg:
                    st.caption(f"⏱️ Time taken: {msg['time_taken']:.2f} seconds")

        # User Input
        if prompt := st.chat_input("Ask a question about your documents..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate Response
            with st.chat_message("assistant"):
                with st.spinner("Thinking (Searching & Reranking)..."):
                    # --- Start Timer ---
                    start_time = time.time()
                    
                    # Invoke LangGraph
                    inputs = {
                        "question": prompt,
                        "file_filter": selected_file,
                        "mode": "qa",
                        "summary_type": "",
                        "context": [],
                        "answer": "",
                        "source_docs": []
                    }
                    
                    result = app_graph.invoke(inputs)
                    
                    # --- End Timer ---
                    end_time = time.time()
                    time_taken = end_time - start_time
                    
                    response = result["answer"]
                    sources = result.get("source_docs", [])

                    st.markdown(response)
                    
                    # Display Time and Sources
                    if sources:
                        st.caption(f"📚 Sources: {', '.join(sources)}")
                    st.caption(f"⏱️ Time taken: {time_taken:.2f} seconds")
                    
                    # Add assistant message to history with time
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response,
                        "sources": sources,
                        "time_taken": time_taken
                    })

    # --- Mode: Summarization ---
    elif mode == "Document Summarization":
        st.header("📝 Document Summarization")
        
        if not files:
            st.warning("Please upload files first.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                target_file = st.selectbox("Select Document", files)
            with col2:
                sum_type = st.selectbox(
                    "Summary Type", 
                    ["detailed", "concise", "bullet", "executive"]
                )

            if st.button("Generate Summary", type="primary"):
                with st.spinner("Reading and Summarizing... (This might take a moment)"):
                    # --- Start Timer ---
                    start_time = time.time()
                    
                    inputs = {
                        "question": "", # Not needed for summary
                        "file_filter": target_file,
                        "mode": "summarize",
                        "summary_type": sum_type,
                        "context": [],
                        "answer": "",
                        "source_docs": []
                    }
                    
                    result = app_graph.invoke(inputs)
                    
                    # --- End Timer ---
                    end_time = time.time()
                    time_taken = end_time - start_time
                    
                    summary = result["answer"]
                    
                    st.markdown("### Summary Result")
                    st.info(summary)
                    
                    # Show time taken
                    st.success(f"⏱️ Summary generated in {time_taken:.2f} seconds")
                    
                    st.download_button(
                        "Download Summary",
                        summary,
                        file_name=f"{target_file}_summary.txt"
                    )

if __name__ == "__main__":
    main()
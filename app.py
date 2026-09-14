"""
app.py
------
Streamlit front-end for the Corrective RAG assistant.

Run:
    streamlit run app.py
"""

import os
import streamlit as st

from ingest import build_vectorstore, DATA_DIR
from graph import build_graph

st.set_page_config(page_title="Corrective RAG Assistant", page_icon="🔎", layout="centered")

st.title("🔎 Corrective RAG Assistant")
st.caption("LangChain + LangGraph · self-grading retrieval with a web-search fallback")

# ---------------------------------------------------------------------------
# Sidebar: document ingestion
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("📄 Knowledge base")
    uploaded_files = st.file_uploader(
        "Upload PDFs to index", type=["pdf"], accept_multiple_files=True
    )

    if st.button("Build / rebuild index", use_container_width=True):
        if not uploaded_files:
            st.warning("Upload at least one PDF first.")
        else:
            os.makedirs(DATA_DIR, exist_ok=True)
            for f in uploaded_files:
                with open(os.path.join(DATA_DIR, f.name), "wb") as out:
                    out.write(f.getbuffer())
            with st.spinner("Chunking, embedding, and indexing..."):
                build_vectorstore()
            st.success("Index built. You can start asking questions.")
            st.session_state.pop("graph_app", None)  # force reload of graph/retriever

    st.divider()
    st.caption(
        "How it works: your question is first checked against the indexed "
        "PDFs. If the retrieved chunks don't actually answer it, the app "
        "automatically falls back to a live web search instead of guessing."
    )

# ---------------------------------------------------------------------------
# Main: chat interface
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask something about your documents...")

if question:
    if not os.path.isdir("chroma_db"):
        st.error("No index found yet. Upload PDFs and click 'Build / rebuild index' first.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        if "graph_app" not in st.session_state:
            st.session_state.graph_app = build_graph()

        with st.chat_message("assistant"):
            with st.spinner("Retrieving and grading context..."):
                result = st.session_state.graph_app.invoke({"question": question})
            badge = "📚 vector store" if result["source"] == "vector_store" else "🌐 web search"
            st.caption(f"Answered using: {badge}")
            st.markdown(result["generation"])

        st.session_state.messages.append(
            {"role": "assistant", "content": result["generation"]}
        )

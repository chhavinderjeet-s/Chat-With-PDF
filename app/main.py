import json
import os
import sys
from dotenv import load_dotenv

# Ensure project root is on sys.path so "app" package is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Load environment variables (e.g., OPENAI_API_KEY) from .env at project root
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

import streamlit as st

from app.utils import (
    pdf_read,
    get_chunks,
    build_vector_store,
    load_vector_store
)
from app.rag_chain import run_rag
from evaluation.evaluate import evaluate_rag
from rate_limit.limiter import rate_limiter
from deployment.deploy import conditional_deploy

# --------------------------------------------------
# Streamlit Config
# --------------------------------------------------
st.set_page_config("RAG Chat PDF")
st.header("📄 RAG-based Continuous Chat with PDF (MLOps)")

# --------------------------------------------------
# Session State Initialization
# --------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "db_ready" not in st.session_state:
    st.session_state.db_ready = False

# --------------------------------------------------
# Sidebar – Upload PDFs
# --------------------------------------------------
with st.sidebar:
    st.title("Upload PDFs")

    pdfs = st.file_uploader(
        "Upload PDF files",
        accept_multiple_files=True
    )

    if st.button("Submit & Process"):
        if not pdfs:
            st.warning("Please upload at least one PDF.")
        else:
            with st.spinner("Processing PDFs..."):
                raw_text = pdf_read(pdfs)
                chunks = get_chunks(raw_text)
                build_vector_store(chunks)

                st.session_state.db_ready = True
                st.session_state.chat_history = []
                st.success("PDFs processed. You can start chatting!")

# --------------------------------------------------
# Chat Interface
# --------------------------------------------------
if st.session_state.db_ready:

    # Show chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Chat input
    user_question = st.chat_input("Ask a question about your PDFs")

    if user_question:
        try:
            # Rate & usage limit
            rate_limiter()

            # Load vector DB
            db = load_vector_store()
            retriever = db.as_retriever()

            # Run RAG
            answer, context = run_rag(
                retriever,
                user_question,
                st.session_state.chat_history
            )

            # Store messages
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_question
            })
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer
            })

            # Display answer
            with st.chat_message("assistant"):
                st.write(answer)

            # --------------------------------------------------
            # Evaluation + Artifact Logging
            # --------------------------------------------------
            ground_truth = context[:500]  # proxy GT
            metrics = evaluate_rag(answer, context, ground_truth)

            with open("artifacts/metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)

            # --------------------------------------------------
            # Conditional Deployment Gate
            # --------------------------------------------------
            success, message = conditional_deploy()

            with st.expander("📊 Evaluation Metrics"):
                st.json(metrics)

            with st.expander("🚦 Deployment Decision"):
                st.write(message)

        except Exception as e:
            st.error(str(e))

else:
    st.info("👈 Upload PDFs from the sidebar to start chatting.")

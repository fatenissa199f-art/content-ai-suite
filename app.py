# ============================================================
# app.py – Dubizzle Group AI Content Lab
# FULL CLEAN VERSION (WORKING RAG + CHAT + FILE UPLOAD)
# ============================================================

import os
import shutil
import tempfile
import pandas as pd
import streamlit as st
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import HuggingFaceHub


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Dubizzle Group AI Content Lab", layout="wide")

st.markdown("""
<h1 style='text-align:center; font-weight:900;'>
Dubizzle Group <span style="color:#D92C27">AI Content Lab</span>
</h1>
<p style='text-align:center; color:#555;'>
Internal AI-powered content platform for Bayut & Dubizzle teams
</p>
""", unsafe_allow_html=True)


# ============================================================
# LOAD RAG INDEX
# ============================================================

DATA_DIR = "data"
VECTOR_DB = "data/faiss_store"

# Embeddings model
emb_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load vector DB
if os.path.exists(VECTOR_DB):
    retriever = FAISS.load_local(
        VECTOR_DB,
        emb_model,
        allow_dangerous_deserialization=True
    ).as_retriever(search_kwargs={"k": 6})
    st.success("✅ Index loaded")
else:
    st.error("❌ No FAISS index found. Upload documents first.")
    retriever = None


# ============================================================
# RAG PROMPT
# ============================================================

prompt = PromptTemplate.from_template("""
You are an internal assistant for Bayut & Dubizzle.

Use ONLY the provided context from internal documents.
If the answer is not found in the documents, say:
"No relevant data found in internal documents."

Context:
{context}

Question:
{question}

Answer:
""")


# ============================================================
# BUILD CONTEXT FUNCTION
# ============================================================

def build_context(question):
    if retriever is None:
        return "NO INDEX LOADED"
    docs = retriever.get_relevant_documents(question)
    return "\n\n".join([d.page_content for d in docs])


# ============================================================
# LLM
# ============================================================

llm = HuggingFaceHub(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    model_kwargs={"temperature": 0.1, "max_length": 1024}
)


# ============================================================
# RAG CHAIN
# ============================================================

chain = (
    {
        "context": RunnablePassthrough(func=build_context),
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
)


# ============================================================
# CHAT UI
# ============================================================

st.subheader("Ask your question:")

query = st.text_input("Enter your question here", "")

if st.button("Ask"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            answer = chain.invoke(query)  # ✅ FIXED — MUST SEND STRING ONLY
        st.markdown("### ✅ Answer")
        st.write(answer)


# ============================================================
# DOCUMENT UPLOAD (OPTIONAL)
# ============================================================

st.divider()
st.subheader("Upload documents & rebuild index")

uploaded_files = st.file_uploader(
    "Upload PDF, DOCX, or text files", 
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

if st.button("Rebuild RAG Index"):
    if uploaded_files:
        # Delete old index
        if os.path.exists(VECTOR_DB):
            shutil.rmtree(VECTOR_DB)

        # Temp directory for processing
        tmp = tempfile.mkdtemp()

        docs = []

        for file in uploaded_files:
            file_path = os.path.join(tmp, file.name)
            with open(file_path, "wb") as f:
                f.write(file.read())

            if file.name.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif file.name.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
            else:
                loader = TextLoader(file_path)

            docs.extend(loader.load())

        # Create new index
        db = FAISS.from_documents(docs, emb_model)
        db.save_local(VECTOR_DB)

        st.success("✅ Index rebuilt successfully!")
    else:
        st.warning("Please upload files first.")


"""
ingest.py
---------
Loads every PDF in ./data, splits it into chunks, embeds the chunks,
and persists them into a local Chroma vector store at ./chroma_db.

Run this once (and again whenever you add new documents):
    python ingest.py
"""

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DATA_DIR = "data"
PERSIST_DIR = "chroma_db"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def load_documents():
    docs = []
    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(f"'{DATA_DIR}/' not found. Create it and add PDFs first.")

    pdf_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in '{DATA_DIR}/'. Add at least one PDF.")

    for filename in pdf_files:
        path = os.path.join(DATA_DIR, filename)
        print(f"Loading {filename} ...")
        loader = PyPDFLoader(path)
        docs.extend(loader.load())

    return docs


def build_vectorstore():
    docs = load_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print("Embedding and persisting to Chroma ...")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
    )
    vectordb.persist()
    print(f"Done. Vector store saved to '{PERSIST_DIR}/'.")


if __name__ == "__main__":
    build_vectorstore()

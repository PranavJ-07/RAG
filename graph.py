"""
graph.py
--------
The Corrective RAG (CRAG) state machine.

Flow:
    retrieve -> grade_documents -> (relevant?) -> generate
                                 -> (not relevant?) -> web_search -> generate

Swap LLM providers by editing `get_llm()` below (OpenAI is the default;
a Groq/Llama option is included but commented out).
"""

import os
from typing import List, TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

from langgraph.graph import StateGraph, END

load_dotenv()

PERSIST_DIR = "chroma_db"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
TOP_K = 4


# ---------------------------------------------------------------------------
# LLM + retriever setup
# ---------------------------------------------------------------------------

def get_llm():
    """Central place to swap LLM providers."""
    return ChatGroq(model="openai/gpt-oss-20b", temperature=0)

    # --- OpenAI alternative ---
    # return ChatOpenAI(model="gpt-4o-mini", temperature=0)


def get_retriever():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    return vectordb.as_retriever(search_kwargs={"k": TOP_K})


llm = get_llm()
retriever = get_retriever()
tavily_tool = TavilySearchResults(k=5)


# ---------------------------------------------------------------------------
# 1. Shared graph state
# ---------------------------------------------------------------------------

class GraphState(TypedDict):
    question: str
    documents: List[str]
    generation: str
    needs_search: bool
    source: str  # "vector_store" or "web_search" — surfaced in the UI


# ---------------------------------------------------------------------------
# 2. Structured output schema for the relevance grader
# ---------------------------------------------------------------------------

class GradeDocuments(BaseModel):
    binary_score: str = Field(
        description="Are the retrieved documents relevant to the question? 'yes' or 'no'."
    )


GRADE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a grader assessing whether retrieved document chunks are "
            "relevant to a user question. Give a binary score 'yes' or 'no'. "
            "'yes' means the chunks contain enough information to answer the "
            "question; 'no' means they do not.",
        ),
        ("human", "Retrieved chunks:\n\n{documents}\n\nUser question: {question}"),
    ]
)

GENERATE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer the user's question using ONLY "
            "the provided context. If the context does not contain the answer, "
            "say you don't know. Cite which source type you used (documents or "
            "web search) at the end of your answer.",
        ),
        ("human", "Context:\n\n{context}\n\nQuestion: {question}"),
    ]
)


# ---------------------------------------------------------------------------
# 3. Nodes
# ---------------------------------------------------------------------------

def retrieve_node(state: GraphState) -> dict:
    docs = retriever.invoke(state["question"])
    return {"documents": [d.page_content for d in docs], "source": "vector_store"}


def grade_documents_node(state: GraphState) -> dict:
    structured_llm = llm.with_structured_output(GradeDocuments)
    chain = GRADE_PROMPT | structured_llm

    doc_text = "\n\n".join(state["documents"]) if state["documents"] else "(no documents retrieved)"
    result = chain.invoke({"documents": doc_text, "question": state["question"]})

    needs_search = result.binary_score.strip().lower() != "yes"
    return {"needs_search": needs_search}


def web_search_node(state: GraphState) -> dict:
    results = tavily_tool.invoke({"query": state["question"]})
    # TavilySearchResults returns a list of dicts with a "content" field
    snippets = [r.get("content", "") for r in results] if isinstance(results, list) else [str(results)]
    return {"documents": snippets, "source": "web_search"}


def generate_node(state: GraphState) -> dict:
    context = "\n\n".join(state["documents"])
    chain = GENERATE_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": state["question"]})
    return {"generation": answer}


# ---------------------------------------------------------------------------
# 4. Conditional routing
# ---------------------------------------------------------------------------

def route_after_grading(state: GraphState) -> str:
    return "web_search" if state["needs_search"] else "generate"


# ---------------------------------------------------------------------------
# 5. Build the graph
# ---------------------------------------------------------------------------

def build_graph():
    builder = StateGraph(GraphState)

    builder.add_node("retrieve", retrieve_node)
    builder.add_node("grade_documents", grade_documents_node)
    builder.add_node("web_search", web_search_node)
    builder.add_node("generate", generate_node)

    builder.set_entry_point("retrieve")
    builder.add_edge("retrieve", "grade_documents")
    builder.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {"web_search": "web_search", "generate": "generate"},
    )
    builder.add_edge("web_search", "generate")
    builder.add_edge("generate", END)

    return builder.compile()


if __name__ == "__main__":
    # Quick CLI smoke test
    app = build_graph()
    question = input("Ask a question: ")
    result = app.invoke({"question": question})
    print(f"\n[source: {result['source']}]\n")
    print(result["generation"])
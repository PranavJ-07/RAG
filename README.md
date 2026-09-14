# Corrective RAG (CRAG) Document Assistant

A retrieval-augmented assistant that self-checks its own retrieval. Instead of
blindly trusting whatever the vector store returns, a LangGraph node grades
the retrieved chunks for relevance — if they're not good enough, the graph
routes to a live web search before generating an answer.

```
[User Query]
     │
     ▼
[1. Retrieve]  ── ChromaDB similarity search
     │
     ▼
[2. Grade]     ── LLM judges: do these chunks actually answer the question?
     │
     ├── relevant ────────────────► [4. Generate]
     │
     └── not relevant ── [3. Web Search] ──► [4. Generate]
```

## Project layout

```
crag-assistant/
├── data/            # put source PDFs here
├── ingest.py        # chunk + embed + persist to Chroma
├── graph.py          # the LangGraph CRAG state machine
├── app.py            # Streamlit UI
├── requirements.txt
└── .env.example
```

## Setup

1. **Install dependencies**

   ```bash
   python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Set API keys**

   ```bash
   cp .env.example .env
   # then fill in OPENAI_API_KEY and TAVILY_API_KEY
   ```

   - OpenAI key: https://platform.openai.com/api-keys
   - Tavily key (free tier, 1,000 searches/month): https://tavily.com

3. **Add documents and build the index**
   Put one or more PDFs in `data/`, then:

   ```bash
   python ingest.py
   ```

4. **Run the app**
   ```bash
   streamlit run app.py
   ```
   Or test the graph directly from the command line:
   ```bash
   python graph.py
   ```

## Swapping the LLM

`graph.py` defaults to `gpt-4o-mini`. To use Groq's free/fast Llama models
instead, install `langchain-groq`, uncomment the Groq block in `get_llm()`,
and set `GROQ_API_KEY` in `.env`.

## Why this is a good portfolio project

Plain RAG (chunk a PDF, embed it, do similarity search) is on a lot of
resumes. What makes this one stand out is the small addition of a **cyclic,
conditional workflow**: the graph makes a runtime decision about its own
retrieval quality instead of trusting top-k results unconditionally. That's
the core idea behind "agentic" RAG systems, and it's a natural jumping-off
point to talk about state machines, self-correction, and reducing
hallucination.

### Resume bullets

- Built an agentic RAG pipeline with LangChain and LangGraph implementing
  Corrective RAG (CRAG), where an LLM grades retrieved-chunk relevance before
  generation.
- Added conditional state-machine routing that falls back to the Tavily web
  search API when retrieval confidence is low, reducing hallucinated answers.
- Indexed documents with Hugging Face embeddings and ChromaDB, and shipped an
  interactive Streamlit interface for end users.

### How to talk about it in an interview

- **The problem:** a standard RAG chain is linear — if the vector store
  returns weak or irrelevant chunks, the model either hallucinates or gives a
  generic non-answer.
- **The fix:** replace the linear chain with a LangGraph `StateGraph`. Each
  node does one job, and a conditional edge decides the next step based on
  the current state.
- **The mechanism:** a grading node uses structured LLM output to score
  chunk relevance. Relevant → generate directly. Not relevant → fall back to
  a live web search, then generate from that instead.
- **Why it matters:** the system checks its own work at runtime instead of
  assuming the top-k nearest neighbors are always good enough — a small step
  toward production-grade reliability.

## 🧠 Agentic RAG with Web Search

This project implements an **agentic Retrieval-Augmented Generation (RAG)** system with integrated **web search capabilities** (Google / DuckDuckGo) via langgraph and local llm model via ollama/gemma4:26b.

---

## 🏗️ System Architecture

The system consists of five core agents:

### 1. Manager Agent (Orchestrator)

Responsible for:
- Handling user queries
- Routing tasks to appropriate agents
- Combining outputs into a final response

#### Responsibilities:
- Detect query type:
  - Knowledge-based → RAG
  - Real-time / external → Web Search
  - Complex → Both
  - Vision
- Manage user actions:
  - Document upload
  - Database reset

---

### 2. RAG Agent (Vector Retrieval)

Handles:
- Document ingestion
- Embedding + storage
- Semantic retrieval

#### Workflow:
1. Receive query from Manager
2. Embed query using `mxbai-embed-large` via Ollama
3. Perform similarity search in ChromaDB
4. Return Top-K relevant chunks

#### Document ingestion:
- Split → Embed → Store in ChromaDB
- Utilizes MarkitDown to ingest documents and convert them into markdowns

---

### 3. Web Search Agent

Handles:
- External knowledge retrieval
- Query decomposition
- Summarization

### 4. Vision Agent

Handles:
- image upload queries

### 5. Aggregrator Agent

Handles:
- Final responses from all agents
- Compares their responses and check for consistences and accuracies
- Summarizes all the agent's responses into a final response to output


#### Workflow:
1. Receive query
2. Decompose into sub-queries (if complex)
3. Perform web search (DuckDuckGo / Google) / Reads the image
4. Retrieve content
5. Summarize into structured output

---

## 🔀 Routing Logic

The Manager Agent determines which agents to invoke:

| Query Type | Action |
|-----------|--------|
| Local knowledge | RAG only |
| Real-time info | Web Search only |
| Image Understanding | Vision only |
| Complex / uncertain | More than one agent |
| Simple query | Direct|

Fallback:
- If RAG returns low relevance → trigger Web Search
- If Web Search fails → fallback to RAG

---

## 🔗 Agent Interfaces

### RAG Agent

```json
{
  "input": "query string",
  "output": {
    "documents": ["chunk1", "chunk2"],
    "scores": [0.92, 0.87]
  }
}

---

## Frontend UI

### Use react chat ui to build the interface

## Communication protocol

### Uses FastApi to communicate between backend and frontend


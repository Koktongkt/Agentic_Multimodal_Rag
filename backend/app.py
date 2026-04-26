from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
import glob
import json
import requests
import time

# ChromaDB
import chromadb
from langgraph.graph import START, StateGraph
from typing import TypedDict, Annotated
from operator import or_

# Import configuration
from .config import (
    DOCS_DIR, CHROMA_DB_PATH, OLLAMA_BASE_URL, EMBEDDING_MODEL, LLM_MODEL,
    CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RESULTS, LLM_TEMPERATURE, LLM_MAX_TOKENS,
    WEB_SEARCH_MAX_RESULTS
)

app = FastAPI(title="Agentic RAG Backend (Chroma + Ollama + LangGraph)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Chroma client (using default Client for in-memory or auto-managed persistence)
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="docs")

# Web search availability
try:
    from duckduckgo_search import ddg
    DDG_AVAILABLE = True
except Exception:
    DDG_AVAILABLE = False

class ChatRequest(BaseModel):
    message: str


def ollama_embed(texts):
    """Call local Ollama embed endpoint. Try several possible endpoints and payload shapes.
    Returns list of embeddings or raises RuntimeError."""
    if not isinstance(texts, list):
        texts = [texts]
    endpoints = [
        f"{OLLAMA_BASE_URL}/embed",
        f"{OLLAMA_BASE_URL}/embeddings",
        f"{OLLAMA_BASE_URL}/api/embed",
        f"{OLLAMA_BASE_URL}/api/embeddings",
    ]
    payloads = [
        {"model": EMBEDDING_MODEL, "input": texts},
        {"model": EMBEDDING_MODEL, "texts": texts},
        {"model": EMBEDDING_MODEL, "data": texts},
    ]
    last_err = None
    for url in endpoints:
        for payload in payloads:
            try:
                resp = requests.post(url, json=payload, timeout=30)
                if resp.status_code >= 400:
                    last_err = f"{url} -> {resp.status_code}: {resp.text}"
                    continue
                data = resp.json()
                # common shapes
                if isinstance(data, dict) and 'embeddings' in data:
                    return data['embeddings']
                if isinstance(data, dict) and 'data' in data:
                    out = []
                    for item in data['data']:
                        if isinstance(item, dict) and 'embedding' in item:
                            out.append(item['embedding'])
                        elif isinstance(item, dict) and 'embeddings' in item:
                            out.append(item['embeddings'])
                    if len(out) > 0:
                        return out
                if isinstance(data, list) and all(isinstance(x, list) for x in data):
                    return data
                # some APIs return {'result': [...]}
                if isinstance(data, dict) and 'result' in data and isinstance(data['result'], list):
                    if all(isinstance(x, list) for x in data['result']):
                        return data['result']
                # try to extract embedding if single object
                if isinstance(data, dict) and 'embedding' in data and isinstance(data['embedding'], list):
                    return [data['embedding']]
            except Exception as e:
                last_err = str(e)
                continue
    raise RuntimeError(f"Ollama embedding failed; last error: {last_err}")


def llm_generate(prompt, model=LLM_MODEL, temperature=LLM_TEMPERATURE, max_tokens=LLM_MAX_TOKENS):
    url = f"{OLLAMA_BASE_URL}/api/generate"

    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "stream": False
    }

    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()

        if "response" in data:
            return data["response"]

        raise RuntimeError(f"Unexpected response format: {data}")

    except Exception as e:
        raise RuntimeError(f"Ollama generate failed: {e}")


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into chunks of chunk_size chars with overlap."""
    if not text:
        return []
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == L:
            break
        start = end - overlap
    return chunks


@app.post('/ingest')
def ingest(force: bool = False):
    """Ingest all .md files from docs/ into Chroma using Ollama embeddings.
    If force=True, existing collection will be cleared and rebuilt."""
    if not os.path.exists(DOCS_DIR):
        return {"status": "docs folder not found", "path": DOCS_DIR}

    if force:
        try:
            chroma_client.delete_collection(name="docs")
        except Exception:
            pass
    # recreate collection
    coll = chroma_client.get_or_create_collection(name="docs")

    # If collection already has items and not force, skip
    try:
        if coll.count() > 0 and not force:
            return {"status": "collection already populated", "count": coll.count()}
    except Exception:
        pass

    files = glob.glob(os.path.join(DOCS_DIR, '**', '*.md'), recursive=True)
    total_chunks = 0
    for fpath in files:
        try:
            with open(fpath, 'r', encoding='utf-8') as fh:
                text = fh.read()
        except Exception:
            continue
        chunks = chunk_text(text)
        if not chunks:
            continue
        # create ids and metadatas
        ids = [f"{os.path.basename(fpath)}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": os.path.relpath(fpath, DOCS_DIR), "chunk_index": i} for i in range(len(chunks))]
        # embed in batches to avoid too large requests
        batch_size = 16
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            try:
                embs = ollama_embed(batch)
            except Exception as e:
                return {"status": "embedding_failed", "error": str(e)}
            batch_ids = ids[i:i+batch_size]
            batch_meta = metadatas[i:i+batch_size]
            coll.add(documents=batch, metadatas=batch_meta, ids=batch_ids, embeddings=embs)
            total_chunks += len(batch)
            time.sleep(0.1)

    # persist if needed (chromadb client handles persistence)
    return {"status": "ingested", "files": len(files), "chunks": total_chunks}


@app.post('/clear')
def clear():
    """Delete the Chroma collection named 'docs' to clear the document database."""
    try:
        chroma_client.delete_collection(name="docs")
    except Exception:
        # ignore errors during delete
        pass
    return {"status": "cleared"}


def rag_query(query, top_k=TOP_K_RESULTS):
    # embed query
    try:
        q_emb = ollama_embed([query])[0]
    except Exception as e:
        return {"answer": f"Embedding failed: {e}", "documents": []}
    coll = chroma_client.get_collection(name="docs")
    try:
        res = coll.query(query_embeddings=[q_emb], n_results=top_k, include=["documents", "metadatas", "distances"])
        docs = res.get('documents', [[]])[0]
        metas = res.get('metadatas', [[]])[0]
        dists = res.get('distances', [[]])[0]
        # prepare context for LLM
        parts = []
        for i, m in enumerate(metas):
            src = m.get('source', 'unknown')
            idx = m.get('chunk_index', i)
            snippet = docs[i]
            parts.append(f"Source: {src} (chunk {idx})\n{snippet}")
        context = "\n\n".join(parts)
        if len(context) > 3000:
            context = context[:3000]
        prompt = f"You are a helpful assistant. Use the following document chunks to answer the question. Cite sources by filename and chunk index.\n\nCONTEXT:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        try:
            llm_ans = llm_generate(prompt)
        except Exception as e:
            llm_ans = f"LLM generation failed: {e}\n\nFallback raw snippets:\n\n{context[:1000]}"
        return {"answer": llm_ans, "documents": docs, "metadatas": metas, "distances": dists}
    except Exception as e:
        return {"answer": f"Chroma query failed: {e}", "documents": []}


# keep web search from before
def web_search(query, max_results=WEB_SEARCH_MAX_RESULTS):
    if not DDG_AVAILABLE:
        return {"answer": "Web search unavailable: install duckduckgo_search package.", "results": []}
    try:
        results = ddg(query, max_results=max_results)
        summary = []
        for r in results:
            title = r.get('title') or r.get('body') or ''
            snippet = r.get('body') or ''
            href = r.get('href') or r.get('url') or ''
            summary.append({'title': title, 'snippet': snippet, 'href': href})
        agg = "\n\n".join([f"{i+1}. {s['title']} - {s['href']}\n{s['snippet']}" for i, s in enumerate(summary)])
        # ask LLM to summarize search results
        prompt = f"Summarize the following search results concisely and list key findings:\n\n{agg}\n\nQuestion: {query}\n\nSummary:"
        try:
            summary_text = llm_generate(prompt)
        except Exception as e:
            summary_text = f"LLM summarization failed: {e}\n\nRaw results:\n{agg}"
        return {"answer": summary_text, "results": summary}
    except Exception as e:
        return {"answer": f"Web search failed: {e}", "results": []}


def route_query(message: str):
    m = message.lower()
    keywords_web = ['recent', 'today', 'news', 'search', 'google', 'duckduckgo', 'latest']
    use_web = any(k in m for k in keywords_web)
    keywords_rag = ['document', 'file', 'docs', 'local', 'definition', 'explain', 'what is', 'who is']
    use_rag = any(k in m for k in keywords_rag)
    if use_web and not use_rag:
        return 'web'
    if use_rag and not use_web:
        return 'rag'
    return 'both'

def route_query_llm(message: str) -> str:
    prompt = f"""
You are a routing agent.

Decide how to answer the user's query:
- "rag" → if the query should be answered using local documents
- "web" → if the query needs recent or external information
- "both" → if both sources may help

Return ONLY one word: rag, web, or both.

Query: {message}
Answer:
"""
    try:
        route = llm_generate(prompt, temperature=0).strip().lower()
        if route in ("rag", "web", "both"):
            return route
    except:
        pass

    # fallback
    return route_query(message)


# Build a LangGraph manager graph to orchestrate RAG and Web Search with separate agents
GRAPH_AVAILABLE = True

class GraphState(TypedDict, total=False):
    message: str
    route: str
    rag: Annotated[dict, lambda a, b:b] # overwrite
    web: Annotated[dict, lambda a, b:b] # overwrite
    response: Annotated[dict, lambda a, b:b] 


try:
    graph = StateGraph(GraphState)

    def manager_node(state: GraphState) -> GraphState:
        """Decide routing based on message and record decision in state."""
        message = state.get('message', '')
        route = route_query_llm(message)
        return {"route": route}

    def rag_agent_node(state: GraphState) -> GraphState:
        """Run RAG agent when route requires it, store result under 'rag'."""
        route = state.get('route', 'both')
        if route not in ('rag', 'both'):
            return {}
        message = state.get('message', '')
        rag_res = rag_query(message)
        return {"rag": rag_res}

    def web_agent_node(state: GraphState) -> GraphState:
        """Run Web Search agent when route requires it, store result under 'web'."""
        route = state.get('route', 'both')
        if route not in ('web', 'both'):
            return {}
        message = state.get('message', '')
        web_res = web_search(message)
        return {"web": web_res}

    def aggregator_node(state: GraphState) -> GraphState:
        route = state.get('route', 'both')
        rag = state.get('rag')
        web = state.get('web')

        # simple cases first (no LLM needed)
        if route == 'rag' and rag:
            return {"response": rag}
        if route == 'web' and web:
            return {"response": web}

        # both → use LLM to decide + synthesize
        prompt = f"""
    You are an expert assistant combining multiple information sources.

    User query:
    {state.get('message')}

    RAG result:
    {rag.get('answer') if rag else "None"}

    Web result:
    {web.get('answer') if web else "None"}

    Instructions:
    - Decide which source is more reliable
    - If both are useful, combine them
    - Prefer accurate and grounded answers
    - Keep answer concise

    Final Answer:
    """
        try:
            final_answer = llm_generate(prompt)
            return {
                "response": {
                    "route": route,
                    "answer": final_answer,
                    "sources": {
                        "rag_used": rag is not None,
                        "web_used": web is not None
                    }
                }
            }
        except Exception as e:
            return {
                "response": {
                    "route": route,
                    "answer": f"Aggregation failed: {e}",
                    "rag": rag,
                    "web": web
                }
            }

    # add nodes and edges: manager -> rag_agent & web_agent -> aggregator
    graph.add_node('manager', manager_node)
    graph.add_node('rag_agent', rag_agent_node)
    graph.add_node('web_agent', web_agent_node)
    graph.add_node('aggregator', aggregator_node)

    graph.add_edge(START, 'manager')
    graph.add_edge('manager', 'rag_agent')
    graph.add_edge('manager', 'web_agent')
    graph.add_edge('rag_agent', 'aggregator')
    graph.add_edge('web_agent', 'aggregator')

    compiled_graph = graph.compile()
except Exception as e:
    compiled_graph = None
    GRAPH_AVAILABLE = False


@app.post('/chat')
def chat(req: ChatRequest):
    message = req.message
    if compiled_graph is not None:
        try:
            out = compiled_graph.invoke({'message': message})
            # compiled graph places final output under 'response'
            return out.get('response', {"status": "no_response"})
        except Exception as e:
            print(f"LangGraph invocation failed: {e}")
    # fallback: run previous routing logic
    route = route_query(message)
    response = {"route": route}
    if route == 'web':
        ws = web_search(message)
        response.update(ws)
    elif route == 'rag':
        rag = rag_query(message)
        response.update(rag)
    else:
        rag = rag_query(message)
        low_confidence = False
        if isinstance(rag.get('distances'), list) and len(rag.get('distances'))>0:
            low_confidence = all(d > 0.5 for d in rag.get('distances'))
        if low_confidence:
            ws = web_search(message)
            response.update({'rag': rag, 'web': ws})
        else:
            response.update({'rag': rag})
    return response


@app.get('/')
def root():
    return {
        "status": "ok",
        "ollama_base_url": OLLAMA_BASE_URL,
        "embedding_model": EMBEDDING_MODEL,
        "llm_model": LLM_MODEL,
        "web_search_available": DDG_AVAILABLE
    }

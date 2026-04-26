from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
import glob
import re
import json
import requests
import time
from markitdown import MarkItDown

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
    """Ingest all documents from DOCS_DIR using markitdown to convert files to text,
    then chunk and store embeddings in Chroma using Ollama.
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

    # Use DocumentIngestor for robust multi-format ingestion
    try:
        from .document_ingestor import DocumentIngestor
    except Exception as e:
        return {"status": "document_ingestor_unavailable", "error": str(e)}

    ingestor = DocumentIngestor(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    documents = ingestor.load_all_documents()
    if not documents:
        return {"status": "no_documents_found"}

    chunks = ingestor.chunk_documents(documents)
    if not chunks:
        return {"status": "no_chunks", "documents": len(documents)}

    total_chunks = 0
    batch_size = 16

    # Add chunks to Chroma in batches
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i+batch_size]
        docs_batch = [c['content'] for c in batch_chunks]
        ids_batch = []
        metas_batch = []
        for c in batch_chunks:
            try:
                rel_source = os.path.relpath(c.get('source', ''), DOCS_DIR)
            except Exception:
                rel_source = c.get('source', '')
            ids_batch.append(f"{rel_source}_{c.get('chunk_id')}")
            metas_batch.append({"source": rel_source, "chunk_index": c.get('chunk_id')})

        try:
            embs = ollama_embed(docs_batch)
        except Exception as e:
            return {"status": "embedding_failed", "error": str(e)}

        coll.add(documents=docs_batch, metadatas=metas_batch, ids=ids_batch, embeddings=embs)
        total_chunks += len(docs_batch)
        time.sleep(0.1)

    return {"status": "ingested", "files": len(documents), "chunks": total_chunks}


@app.post('/clear')
def clear():
    """Delete the Chroma collection named 'docs' to clear the document database."""
    try:
        chroma_client.delete_collection(name="docs")
    except Exception:
        # ignore errors during delete
        pass
    return {"status": "cleared"}


def extract_json(text: str):
    """Layer 2: robust JSON extraction"""

    # 1. direct parse
    try:
        return json.loads(text)
    except:
        pass

    # 2. extract JSON block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass

    return None

def rag_agent(query, max_iters=3, top_k=TOP_K_RESULTS):
    current_query = query
    last_context = []
    
    for i in range(max_iters):

        # 1. Retrieve
        q_emb = ollama_embed([current_query])[0]
        coll = chroma_client.get_collection(name="docs")

        res = coll.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]

        context = [
            f"[{m.get('source')} | {m.get('chunk_index')}]\n{docs[j]}"
            for j, m in enumerate(metas)
        ]

        last_context = context

        # 2. Judge retrieval quality (LLM critic)
        judge_prompt = f"""
        You are a retrieval evaluation agent.

        Your job:
        1. Decide if retrieved context is sufficient to answer the question
        2. If NOT sufficient, propose a better search query. Break down the query into more specific sub-questions if needed
        or suggest different keywords to improve retrieval precision or coverage.

        IMPORTANT RULES:
        - If sufficient = true → improved_query MUST be ""
        - If sufficient = false → improved_query MUST be a better search query
        - improved_query must improve retrieval precision or coverage
        - Do NOT repeat the same query unless necessary

        Return ONLY valid JSON:

        {{
        "sufficient": true or false,
        "reason": "short explanation",
        "improved_query": "string or empty string"
        }}

        Question:
        {query}

        Retrieved context:
        {context[:4]}
        """

        result = llm_generate(judge_prompt, temperature=0)

        data = extract_json(result)

        if data and data.get("sufficient") == True:
            break

        # Obtain improved query from LLM feedback, if provided, otherwise fallback to a simple refinement prompt
        improved_query = (data or {}).get("improved_query")

        if improved_query and improved_query != current_query:
            current_query = improved_query
        else:
            # fallback refinement (important safeguard)
            refine_prompt = f"""
            Improve this search query for better document retrieval.

            Query: {current_query}
            Return ONLY the improved query.
            """
            current_query = llm_generate(refine_prompt, temperature=0).strip()

    # 4. Final synthesis
    final_prompt = f"""
    You are a RAG assistant.

    Use the context to answer the question. If there are no context, say you don't know. Be concise and clear.

    Context:
    {last_context[:8]}

    Question:
    {query}

    Return JSON:
    {{
    "answer": "...",
    "key_points": ["..."],
    "sources": ["..."]
    }}
    """

    result = llm_generate(final_prompt)
    structured = extract_json(result) or {}

    return {
        "answer": structured.get("answer", ""),
        "key_points": structured.get("key_points", []),
        "sources": structured.get("sources", []),
        "final_query": current_query,
        "context": last_context
    }


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
    direct_answer: str
    rag: Annotated[dict, lambda a, b:b] # overwrite
    web: Annotated[dict, lambda a, b:b] # overwrite
    response: Annotated[dict, lambda a, b:b] 


try:
    graph = StateGraph(GraphState)

    def manager_node(state: GraphState) -> GraphState:
        message = state.get("message", "")

        prompt = f"""
    You are a smart assistant router.

    Decide how to handle the query:

    Options:
    1. "direct" → You can answer it yourself without external data
    2. "rag" → Needs internal documents
    3. "web" → Needs external/recent info
    4. "both" → Needs both sources

    Return ONLY JSON in this format:
    {{
    "mode": "...",
    "answer": "only if mode=direct, otherwise empty string"
    }}

    Query: {message}
    """

        try:
            import json

            result = llm_generate(prompt, temperature=0)

            data = json.loads(result)

            mode = data.get("mode", "both")
            answer = data.get("answer", "")

            return {
                "route": mode,
                "direct_answer": answer
            }

        except Exception:
            # fallback to old router
            return {
                "route": route_query_llm(message),
                "direct_answer": ""
            }

    def rag_agent_node(state: GraphState) -> GraphState:
        """Run RAG agent when route requires it, store result under 'rag'."""
        route = state.get('route', 'both')
        if route not in ('rag', 'both'):
            return {}
        message = state.get('message', '')
        rag_res = rag_agent(message)
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
        route = state.get("route")
        direct = state.get("direct_answer")

        # ✅ FAST PATH: no agents needed
        if route == "direct" and direct:
            return {
                "response": {
                    "route": "direct",
                    "answer": direct
                }
            }

        rag = state.get("rag")
        web = state.get("web")

        # FINAL AGGREGATION PROMPT, if rag/web no results, their portion will be none/empty, LLM should learn to handle that gracefully
        prompt = f"""
        You are a final response formatter.

        Rules:
        - Be concise and clear
        - Do NOT repeat full documents
        - Do NOT use bullet lists unless necessary
        - If both RAG and Web info are present, compare them first for any conflicting information and synthesize them into a single answer.

        User Query:
        {state.get('message')}

        RAG Answer:
        {rag.get('answer') if rag else "None"}
        {rag.get('key_points') if rag else "None"}
        {rag.get('sources') if rag else "None"}

        Web Answer:
        {web.get('answer') if web else "None"}

        Final concise answer:
        """

        try:
            final = llm_generate(prompt)
            return {
                "response": {
                    "route": route,
                    "answer": final
                }
            }
        except Exception as e:
            return {
                "response": {
                    "answer": f"Aggregation failed: {e}"
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
        rag = rag_agent(message)
        response.update(rag)
    else:
        rag = rag_agent(message)
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

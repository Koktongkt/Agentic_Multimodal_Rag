import chunk
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
import re
import json
import requests
import time
import hashlib # file hashing for better rag

from typing import List, Dict, Optional
from websockets import route
from .document_ingestor import DocumentIngestor
from .web_search import web_search_agent

# ChromaDB
import chromadb
from langgraph.graph import START, StateGraph
from typing import TypedDict, Annotated
from operator import or_

# vision endpoint
from .vision import call_ollama_vision, router as vision_router

# Import configuration
from .config import (
    DOCS_DIR, OLLAMA_BASE_URL, EMBEDDING_MODEL, LLM_MODEL,
    CHUNK_SIZE, CHUNK_OVERLAP, WEB_SEARCH_MAX_RESULTS, TOP_K_RESULTS, LLM_TEMPERATURE, LLM_MAX_TOKENS
)

# Import chromadb collection
from .db import collection as coll, chroma_client

app = FastAPI(title="Agentic RAG Backend (Chroma + Ollama + LangGraph)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(vision_router, prefix="/api")


# Web search availability
from ddgs import DDGS


class ChatRequest(BaseModel):
    message: str
    image_b64: str | None = None
    history: list[dict] | None = None


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

def file_hash(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def chunk_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@app.post('/ingest')
def ingest(force: bool = False, files: Optional[List[UploadFile]] = File(None)):
    # Ensure docs dir exists or create it if files are uploaded
    if not os.path.exists(DOCS_DIR):
        if not files:
            return {"status": "docs folder not found"}
        os.makedirs(DOCS_DIR, exist_ok=True)

    # Save uploaded files to DOCS_DIR if any were provided
    if files:
        saved = []
        for up in files:
            try:
                filename = os.path.basename(up.filename)
                dest = os.path.join(DOCS_DIR, filename)
                with open(dest, "wb") as f:
                    f.write(up.file.read())
                saved.append(filename)
            except Exception as e:
                print("Failed saving uploaded file:", up.filename, e)
        print("Saved uploaded files:", saved)

    if force:
        try:
            chroma_client.delete_collection(name="docs")
        except Exception:
            pass
        # Recreate the collection object so subsequent calls use a fresh, empty collection
        global coll
        try:
            coll = chroma_client.get_or_create_collection("docs")
        except Exception:
            pass

    # --- Load existing file hashes ---
    existing_files = {}
    try:
        results = coll.get(include=["metadatas"])
        for m in results.get("metadatas", []):
            if m and m.get("source") and m.get("file_hash"):
                existing_files[m["source"]] = m["file_hash"]
    except Exception:
        pass

    # --- Load documents ---
    ingestor = DocumentIngestor(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    documents = ingestor.load_all_documents()

    normalized_docs = []
    for d in documents:
        source_path = d.get("path")

        if not source_path:
            #print("Skipping doc with no source:", d)
            continue

        d["source"] = source_path  # enforce consistency
        normalized_docs.append(d)

    documents = normalized_docs
    print("Loaded docs:", len(documents))

    filtered_docs = []
    for d in documents:
        source = os.path.relpath(d["source"], DOCS_DIR)
        f_hash = file_hash(d["source"])

        # Skip unchanged
        if existing_files.get(source) == f_hash:
            continue

        # Delete old chunks if updated, source exists but hash key is different in existing_files
        if source in existing_files:
            coll.delete(where={"source": source})

        d["file_hash"] = f_hash
        filtered_docs.append(d)

    if not filtered_docs:
        return {"status": "no new or updated documents"}

    # --- Chunking ---
    chunks = ingestor.chunk_documents(filtered_docs)

    # Attach file_hash to chunks
    doc_hash_map = {
        os.path.relpath(d["source"], DOCS_DIR): d["file_hash"]
        for d in filtered_docs
    }

    for c in chunks:
        rel_source = os.path.relpath(c.get("source", ""), DOCS_DIR)
        c["file_hash"] = doc_hash_map.get(rel_source)

    # --- Insert into Chroma ---
    batch_size = 16
    total_chunks = 0

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]

        docs_batch = [c["content"] for c in batch]
        ids_batch = [
            f"{os.path.relpath(c['source'], DOCS_DIR)}_{chunk_hash(c['content'])}"
            for c in batch
        ]
        metas_batch = [
            {
                "source": os.path.relpath(c["source"], DOCS_DIR),
                "chunk_index": c.get("chunk_id"),
                "file_hash": c.get("file_hash"),
            }
            for c in batch
        ]

        embs = ollama_embed(docs_batch)
        if not embs or len(embs) != len(docs_batch):
            continue

        coll.add(
            documents=docs_batch,
            metadatas=metas_batch,
            ids=ids_batch,
            embeddings=embs
        )

        total_chunks += len(docs_batch)
        time.sleep(0.05)

    return {
        "status": "ingested",
        "files": len(filtered_docs),
        "chunks": total_chunks
    }


@app.post('/clear')
def clear():
    """Delete the Chroma collection named 'docs' to clear the document database."""
    try:
        chroma_client.delete_collection(name="docs")
    except Exception:
        # ignore errors during delete
        pass

    # Re-create the in-process collection object so subsequent ingest calls use a valid, empty collection
    global coll
    try:
        coll = chroma_client.get_or_create_collection("docs")
    except Exception:
        pass

    return {"status": "cleared"}


@app.post('/upload')
async def upload_endpoint(store: bool = False, query: str = "", files: Optional[List[UploadFile]] = File(None)):
    """Unified upload endpoint used by the frontend.
    Accepts image and document files. If `store` is true files are saved to DOCS_DIR and an ingest is triggered.
    For images, route through the manager (LangGraph) when available so routing/aggregation applies.
    """
    results = []
    try:
        if not files:
            return {"results": [], "error": "no files uploaded"}

        # ensure docs dir exists if storing
        if store and not os.path.exists(DOCS_DIR):
            os.makedirs(DOCS_DIR, exist_ok=True)

        for up in files:
            filename = os.path.basename(up.filename)
            entry = {"filename": filename}
            try:
                # read content (support async UploadFile)
                try:
                    content = await up.read()
                except Exception:
                    content = up.file.read()

                # simple image detection
                ext = os.path.splitext(filename)[1].lower()
                is_image = (up.content_type and up.content_type.startswith("image")) or ext in [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff"]

                if is_image:
                    import base64 as _b64
                    image_b64 = _b64.b64encode(content).decode("utf-8")
                    # If the LangGraph manager is available, invoke it so vision runs through manager_node and aggregator
                    try:
                        if compiled_graph is not None:
                            out = compiled_graph.invoke({
                                "message": query or "",
                                "image_b64": image_b64,
                                "history": []
                            })
                            # The compiled graph returns a dict with 'response' key when successful
                            resp = out.get("response", out)
                            entry["vision_manager"] = resp
                        else:
                            # fallback: call vision directly
                            vis = call_ollama_vision(query or "Describe the image in detail.", image_b64, LLM_MODEL)
                            entry["vision"] = vis["message"]["content"] if isinstance(vis, dict) and "message" in vis else vis
                    except Exception as e:
                        entry["error"] = str(e)
                else:
                    # non-image: save if requested, otherwise just acknowledge
                    if store:
                        dest = os.path.join(DOCS_DIR, filename)
                        try:
                            with open(dest, "wb") as f:
                                f.write(content)
                            entry["stored"] = True
                        except Exception as e:
                            entry["error"] = f"Failed saving file: {e}"
                    else:
                        entry["note"] = "file received"

            except Exception as e:
                entry["error"] = str(e)

            results.append(entry)

        ingest_result = None
        if store:
            # Trigger ingest to index saved files. Call ingest() directly so it reads DOCS_DIR
            try:
                ingest_result = ingest(False)
            except Exception as e:
                ingest_result = {"error": str(e)}

        return {"results": results, "ingest": ingest_result}

    except Exception as e:
        return {"results": results, "error": str(e)}


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

def rag_agent(query, history=None, max_iters=3, top_k=TOP_K_RESULTS):
    
    last_context = []
    if history:
        history_text = "\n".join(
            [f"{h['role']}: {h['content']}" for h in history[-3:]]  # last 3 turns
        )
        rewrite_prompt = f"""
        Rewrite the query to be fully self-contained using the conversation history for context. If the query is already self-contained, return it as is without any changes. 
        If the query does not contain clear intent or subjects/objects, rewrite it to be more specific and clear. Use the conversation history for context.
        
        Conversation history:
        {history_text}

        Query:
        {query}

        Return ONLY the query. Do not add explanation or extra text.
        Preserve wording unless clarification is needed for context resolution.
        """

        current_query = llm_generate(rewrite_prompt, temperature=0).strip()
    else:
        current_query = query

    for i in range(max_iters):

        # 1. Retrieve
        q_emb = ollama_embed([current_query])[0]

        res = coll.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        docs = res["documents"][0]
        metas = res["metadatas"][0]
        #dists = res["distances"][0]

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
        - If sufficient = True → improved_query MUST be ""
        - If sufficient = False → improved_query MUST be a better search query
        - improved_query must improve retrieval precision or coverage
        - Do NOT repeat the same query unless necessary

        Return ONLY valid JSON:

        {{
        "sufficient": True or False,
        "reason": "short explanation",
        "improved_query": "string or empty string"
        }}

        Question:
        {current_query}

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
    
    Refer to the conversation history for additional context, but do not rely on it too much as it may be incomplete. Focus on the retrieved documents as primary context.

    Conversation history:
    {history_text}

    Context:
    {last_context[:8]}

    Question:
    {current_query}

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


# hardcoded heuristic routing as fallback when LangGraph is not available or fails
def route_query(message: str):
    m = message.lower()
    keywords_web = ['recent', 'news', 'search', 'google', 'web', 'current', 'latest', 'trending']
    use_web = any(k in m for k in keywords_web)
    keywords_rag = ['document', 'file', 'docs', 'local', 'definition', 'explain', 'what is', 'who is']
    use_rag = any(k in m for k in keywords_rag)
    if use_web and not use_rag:
        return 'web'
    if use_rag and not use_web:
        return 'rag'
    return 'both'


# Build a LangGraph manager graph to orchestrate RAG and Web Search with separate agents
GRAPH_AVAILABLE = True

class GraphState(TypedDict):
    message: str
    image_b64: str | None 
    history: List[Dict[str, str]]  # optional conversation history for context
    route: dict
    direct_answer: str
    rag: dict
    web: dict
    vision: dict
    response: dict


try:
    graph = StateGraph(GraphState)

    def manager_node(state: GraphState) -> GraphState:
        message = state.get("message", "")
        image = state.get("image_b64")
        history = state.get("history", [])

        history_text = "\n".join(
            [f"{h['role']}: {h['content']}" for h in history[-10:]]  # last 10 turns
        )

        prompt = f"""
        You are a routing system.

        Conversation history:
        {history_text}

        Image present: {"yes" if image else "no"}

        You must decide ALL applicable routes. Multiple routes CAN be true at the same time.
        Take note of the Conversation history as it may provide important context for routing decisions.

        Return ONLY valid JSON in this format:

        {{
        "vision": true or false,
        "rag": true or false,
        "web": true or false,
        "direct": true or false,
        "direct_answer": "ONLY if direct=true, return the answer to the question, otherwise empty string"
        }}

        Rules:
        - vision = true if image is present or question about image
        - rag = true if question requires documents or definitions on topic of covid19
        - web = true if question requires recent / external information, or if it is a general knowledge question that can be answered better with web search
        - direct = true if general knowledge question that is considered simple and no image present.

        Query:
        {message}
        """

        result = llm_generate(prompt, temperature=0)
        data = extract_json(result) or {}

        route = {
            "vision": data.get("vision", False),
            "rag": data.get("rag", False),
            "web": data.get("web", False),
            "direct": data.get("direct", False),
        }
        print(f"Routing decision: {route}")
        direct_answer = data.get("direct_answer", "")

        # fallback safety
        if not any(route.values()):
            route = {
                "vision": image is not None,
                "rag": False,
                "web": False,
                "direct": True
            }

        return {
            "route": route,
            "direct_answer": direct_answer
        }

    def rag_agent_node(state: GraphState) -> GraphState:
        """Run RAG agent when route requires it, store result under 'rag'."""
        route = state["route"]
        if not route.get("rag"):
            return {}
        message = state.get('message', '')
        history = state.get('history', [])

        rag_response = rag_agent(message, history=history)
        return {"rag": rag_response}

    def web_agent_node(state: GraphState) -> GraphState:
        """Run Web Search agent when route requires it, store result under 'web'."""
        route = state["route"]
        if not route.get("web"):
            return {}
        message = state.get('message', '')
        history = state.get('history', [])
        
        web_response = web_search_agent(
            query=message, 
            history=history,
            llm_generate=llm_generate,
            k=WEB_SEARCH_MAX_RESULTS,
            extract_json=extract_json
        )
        return {"web": web_response}

    def vision_agent_node(state: GraphState) -> GraphState:
        route = state.get("route", {})

        if not route.get("vision"):
            return {}

        message = state.get("message", "")
        history = state.get("history", [])
        image_b64 = state.get("image_b64")

        history_text = ""
        if history:
            history_text = "\n".join(
                [f"{h['role']}: {h['content']}" for h in history[-3:]]
            )

        if not image_b64:
            return {"vision": {"error": "No image provided"}}

        prompt = message if message else "Describe this image in detail."
        system_prompt = """
        You are a vision AI.

        Use the conversation history for additional context if required and if user's request is ambiguous, but do not rely on it too much as it may be incomplete. Focus on analyzing the image.

        Return ONLY JSON:
        {
        "summary": "",
        "objects": [],
        "scene": "",
        "text_in_image": "",
        "user_question_answer": "",
        "safety_notes": ""
        }
        """
        final_prompt = f"""
        {system_prompt}

        User request:
        {prompt}
        """

        result = call_ollama_vision(
            prompt=final_prompt,
            image_b64=image_b64,
            model=LLM_MODEL
        )
        raw = result["message"]["content"]
        data = extract_json(raw)

        if not data:
            data = {
                "summary": raw,
                "objects": [],
                "scene": "",
                "text_in_image": "",
                "user_question_answer": "",
                "safety_notes": ""
            }

        return {
            "vision": data
        }

    def aggregator_node(state: GraphState) -> GraphState:
        route = state.get("route")
        direct = state.get("direct_answer")
        history = state.get("history", [])

        # ✅ FAST PATH: no agents needed
        if route.get("direct") and direct:
            return {
                "response": {
                    "route": "direct",
                    "answer": direct
                }
            }

        rag = state.get("rag")
        web = state.get("web")
        vision = state.get("vision")

        # FINAL AGGREGATION PROMPT, if rag/web no results, their portion will be none/empty, LLM should learn to handle that gracefully
        prompt = f"""
        You are a final response formatter.

        Rules:
        - Be concise and clear
        - Do NOT repeat full documents
        - Do NOT use bullet lists unless necessary
        - If both RAG and Web info are present, compare them first for any conflicting information and synthesize them into a single answer.
        - If Vision info is present, include it in the final answer.
        - Remove any special characters or formatting from the sources.

        User Query:
        {state.get('message')}

        RAG Answer:
        {rag.get('answer') if rag else "None"}
        {rag.get('key_points') if rag else "None"}
        {rag.get('sources') if rag else "None"}

        Web Answer:
        {web.get('answer') if web else "None"}
        {web.get('key_points') if web else "None"}
        {web.get('sources') if web else "None"}

        Vision Answer:
        {vision.get('summary') if vision else "None"}
        {vision.get('objects') if vision else "None"}
        {vision.get('scene') if vision else "None"}
        {vision.get('text_in_image') if vision else "None"}
        {vision.get('user_question_answer') if vision else "None"}

        Final concise answer:
        """

        try:
            final = llm_generate(prompt)
            return {
                "response": {
                    "route": route,
                    "answer": final,
                    "history": history
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
    graph.add_node("vision_agent", vision_agent_node)
    graph.add_node('rag_agent', rag_agent_node)
    graph.add_node('web_agent', web_agent_node)
    graph.add_node('aggregator', aggregator_node)

    graph.add_edge(START, 'manager')
    graph.add_edge("manager", "vision_agent")
    graph.add_edge('manager', 'rag_agent')
    graph.add_edge('manager', 'web_agent')
    graph.add_edge("vision_agent", "aggregator")
    graph.add_edge('rag_agent', 'aggregator')
    graph.add_edge('web_agent', 'aggregator')

    compiled_graph = graph.compile()
except Exception as e:
    compiled_graph = None
    GRAPH_AVAILABLE = False


@app.post('/chat')
def chat(req: ChatRequest):

    if compiled_graph is not None:
        try:
            out = compiled_graph.invoke({
                "message": req.message,
                "image_b64": req.image_b64,
                "history": req.history or []
            })

            return out.get("response", {
                "status": "no_response"
            })

        except Exception as e:
            print(f"LangGraph invocation failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    return {
        "status": "error",
        "error": "LangGraph not initialized"
    }


@app.get('/')
def root():
    return {
        "status": "ok",
        "ollama_base_url": OLLAMA_BASE_URL,
        "embedding_model": EMBEDDING_MODEL,
        "llm_model": LLM_MODEL
    }

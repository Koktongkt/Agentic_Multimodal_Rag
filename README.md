## 🧠 Agentic RAG with Web Search

This project implements an **agentic Retrieval-Augmented Generation (RAG)** system with integrated **web search capabilities** (Google / DuckDuckGo) via langgraph and local llm model via ollama/gemma4:26b. The goal is to run an agentic multimodal rag in a local system.

---

## 🏗️ System Architecture

The system consists of five core agents:

### 1. Manager Agent (Orchestrator)

Responsible for:
- Handling user queries
- Routing tasks to appropriate agents by providing a boolean route output

#### Responsibilities:
- Detect query type:
  - Knowledge-based → RAG (manager agent prompt need to mention what kind of topic e.g. currently it is grounded in covid19)
  - Real-time / external knowledge / complex queries → Web Search
  - Complex → Both
  - Vision when Image is attached
  - Direct answer if simple query
  - If document, 

- Manage user actions:
  - receiving Document / Image upload
  - deciphering User's query and determine the route to downstream agent

- Prompt tuning:
  - Rules are defined for how each route are determined, with an output of either True or False. Tune further based on your requirement.
  - If the user query is very simple / conversational, the manager agent will otuput direct: true and provide a direct answer under direct_answer output.
  - Tune further and include more routes if scaling with more agents (remember to add in the GraphState dict as well for more required inputs). Note: For more complicated use case/business logic, the current method may not be efficient or even feasible.

---

### 2. RAG Agent (Vector Retrieval)
Handles:
- Semantic retrieval via query embedding

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

### 5. Document Agent
Handles:
- Document upload queries
- If no queries provided, it will summarize the document as default
- If user's query contain intents to store the document in RAG database, it will perform duplication of doc in RAG database by file hash check. It the doc file hash does not exist, it will then store the doc in the storage folder and perform chunking + embedding.

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
| Document Understanding | Document only |
| Complex / uncertain | More than one agent |
| Simple query, documents attachment | Direct|

Fallback:
- Hardcoded rules to decide to use RAG or Web Search using keywords based

---

## Frontend UI

### Use React Chat as Frontend UI to build the interface:
 - /CHAT via SEND button for user to send a query
 - /INGEST via Upload & Ingest Docs button for user to ingest documents from the local database
  - User can upload docs to ingest new docs, or skip and reingest current docs (if docs already exist in vector db, then no reingestion)
 - /CLEAR via clear database button for user to clear the vector database (chromadb)
 - /CLEAR and FORCE=TRUE via Force Clear and rebuilt button to rebuilt database entirely

## Communication protocol
 Uses FastApi to communicate between backend and frontend
/chat to send query
/ingest to ingest new docs in database
/ingest and Force=True to rebult database entirely
/vision to read image and output response in json format

## RAG ingestion method
- Uses MarkItDown to convert files of multiple forms (docx, xlsx, pdf, etc) into markdowns
- Chunking selection (adjustable in config.py) with ollama embed model mxbai-embed-large (also adjustable in config.py)
- Chunks embeddings are stored in Chroma_db
- Hashlib is used together to store the source (docs path) together with the files in chroma_db
  - Hash_file helper function takes the path file as input, opens the file and hashes the content
- For new documents / updated parts of same documents, the hash key will be used to check whether the same chunks of the doc in chroma db needs to be replaced anot

## Memory
- currently conversation are stored in-memory as history for the last 3 conversation [-3]. You can extend this, but it eats away the context window.
- SessionStorage is initalized from frontend to backend for session_id via uuid generation. This is for protoype/local development only. For production grade, switch to localStorage to possibly cloud DB storage for actual user sessions storage.
- SessionStorage stores mainly uploaded document and its content in the event a user chooses to requests to store this in RAG database.

### Future updates
- Graph RAG option for user for more complicated / extensive knowledge
---
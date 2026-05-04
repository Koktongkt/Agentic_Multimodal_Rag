from ddgs import DDGS
from googlesearch import search as google_search
import requests
from bs4 import BeautifulSoup
from .config import WEB_SEARCH_MAX_RESULTS


# SEARCH LAYERS
def ddg_search(*, query, k=None):
    k = int(k or WEB_SEARCH_MAX_RESULTS)

    results = DDGS().text(query, max_results=k)

    return [
        {
            "title": r.get("title", ""),
            "snippet": r.get("body", ""),
            "url": r.get("href", "")
        }
        for r in results
    ]


def google_search_urls(*, query, k=None):
    k = int(k or WEB_SEARCH_MAX_RESULTS)
    return list(google_search(query, num_results=k))


# FETCHING
def fetch_page(url):
    try:
        r = requests.get(
            url,
            timeout=5,
            headers={"User-Agent": "Mozilla/5.0"}
        )

        soup = BeautifulSoup(r.text, "html.parser")

        for tag in soup(["script", "style"]):
            tag.decompose()

        text = soup.get_text(" ", strip=True)
        return text[:2500]

    except:
        return ""


# FORMATTERS
def format_ddg(ddg_results):
    return "\n".join(
        f"{r['title']}\n{r['snippet']}\n{r['url']}"
        for r in ddg_results
    )


def format_google(pages):
    return "\n".join(
        f"{p['url']}\n{p['content']}"
        for p in pages
    )


# OPTIONAL COMBINED RAW SEARCH
def combined_web_search(*, query, k=None):
    k = int(k or WEB_SEARCH_MAX_RESULTS)

    ddg_results = ddg_search(query=query, k=k)
    google_urls = google_search_urls(query=query, k=k)

    google_results = [fetch_page(url) for url in google_urls]

    return {
        "ddg": ddg_results,
        "google": google_results
    }


# MAIN AGENT
def web_search_agent(*, query, history=None, llm_generate, k=None, extract_json):
    
    k = int(k or WEB_SEARCH_MAX_RESULTS)
    if history:
        history_text = "\n".join(
            [f"{h['role']}: {h['content']}" for h in history[-3:]]
        )

        rewrite_prompt = f"""
        Rewrite the query to be self-contained using the conversation, if the query is not already self-contained. If the query is already self-contained, return it as is without any changes.
        If the query does not contain clear intent or subjects/objects, rewrite it to be more specific and clear. Use the conversation history for context.
        Conversation:
        {history_text}

        Query:
        {query}

        Return ONLY the query. Do not add explanation or extra text.
        Preserve wording unless clarification is needed for context resolution.
        """

        current_query = llm_generate(rewrite_prompt, temperature=0).strip()
    else:
        current_query = query

        # 1. Retrieve
    ddg_results = ddg_search(query=current_query, k=k)
    google_urls = google_search_urls(query=current_query, k=k)

    # 2. Fetch
    google_pages = []
    for url in google_urls:
        content = fetch_page(url)
        if content:
            google_pages.append({
                "url": url,
                "content": content
            })

    # 3. Format evidence
    ddg_text = format_ddg(ddg_results)
    google_text = format_google(google_pages)

    # 4. LLM synthesis
    prompt = f"""
    You are a web research agent.

    DUCKDUCKGO RESULTS:
    {ddg_text}

    GOOGLE PAGE CONTENT:
    {google_text}

    TASK:
    - Extract key facts
    - Identify consensus
    - Identify contradictions
    - Ignore duplicates
    - Be precise
    - Remove any special characters or formatting from the sources.

    Return ONLY JSON:
    {{
    "answer": "...",
    "key_points": ["..."],
    "sources": ["..."]
    }}

    QUESTION:
    {current_query}
    """

    result = llm_generate(prompt, temperature=0)
    structured = extract_json(result)

    return {
        "answer": structured.get("answer", ""),
        "key_points": structured.get("key_points", []),
        "sources": structured.get("sources", []),
        "raw": {
            "ddg": ddg_results,
            "google": google_pages
        }
    }
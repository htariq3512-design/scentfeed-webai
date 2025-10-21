import os, re, json, requests
from typing import List, Dict
from bs4 import BeautifulSoup
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- Config via environment variables ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

app = FastAPI(title="ScentFeed Web AI")

# Allow iOS app (and test tools) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later (e.g., your domains)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskReq(BaseModel):
    query: str
    max_sources: int = 3

@app.get("/health")
def health():
    return {"ok": True}

def tavily_search(query: str, k: int = 3) -> List[Dict]:
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "search_depth": "advanced",
        "max_results": k
    }
    r = requests.post(url, json=payload, timeout=20)
    r.raise_for_status()
    data = r.json()
    results = data.get("results", [])[:k]
    return [{"title": r.get("title",""), "url": r.get("url","")} for r in results]

def fetch_text(url: str, limit_chars: int = 7000) -> str:
    try:
        r = requests.get(url, timeout=20, headers={"User-Agent":"Mozilla/5.0"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]): tag.decompose()
        text = soup.get_text(separator=" ")
        text = re.sub(r"\s+", " ", text).strip()
        return text[:limit_chars]
    except Exception:
        return ""

def openai_answer(query: str, docs: List[Dict]) -> str:
    context_blocks = []
    for i, d in enumerate(docs, 1):
        context_blocks.append(f"[Source {i}] {d['url']}\n{d['text']}\n")
    context = "\n\n".join(context_blocks) if context_blocks else "No sources fetched."

    prompt = f"""
You are a careful, precise research assistant for fragrance topics.
Answer the user's question using ONLY the context below.
Cite sources inline like [1], [2], etc, corresponding to URLs at the end.

Question: {query}

Context:
{context}

Requirements:
- Be concise and accurate (no hallucinations).
- If uncertain, say what's missing.
- If (and only if) the user asks for recommendations, include 2–4 bullet points.
- End with: "Sources: [1] URL1 · [2] URL2 · [3] URL3"
"""

    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type":"application/json"}
    body = {
        "model": "gpt-4o-mini",
        "temperature": 0.2,
        "messages": [
            {"role":"system","content":"You write reliable answers with citations."},
            {"role":"user","content": prompt}
        ]
    }
    rr = requests.post(url, headers=headers, json=body, timeout=60)
    rr.raise_for_status()
    data = rr.json()
    return data["choices"][0]["message"]["content"].strip()

@app.post("/ask")
def ask(req: AskReq):
    if not OPENAI_API_KEY:
        return {"ok": False, "error": "Missing OPENAI_API_KEY"}
    if not TAVILY_API_KEY:
        return {"ok": False, "error": "Missing TAVILY_API_KEY"}

    # 1) Search
    results = tavily_search(req.query, k=max(1, min(req.max_sources, 5)))

    # 2) Fetch
    docs = []
    for r in results:
        text = fetch_text(r["url"])
        if len(text) > 300:
            docs.append({"url": r["url"], "text": text})

    # 3) Answer
    answer = openai_answer(req.query, docs)
    return {"ok": True, "answer": answer, "sources": [d["url"] for d in docs]}





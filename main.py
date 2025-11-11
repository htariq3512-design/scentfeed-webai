# main.py
import os, json, time, logging, re, hashlib, threading
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from openai import OpenAI

# ------------------------------
# Bootstrap
# ------------------------------
load_dotenv(override=True)
APP_VERSION = "2.1.0-cache+trace+facets"

# Firestore (optional)
FIRESTORE_READY = False
db = None
try:
    from firebase_admin import credentials, initialize_app
    from google.cloud import firestore
    FIREBASE_IMPORT_OK = True
except Exception:
    FIREBASE_IMPORT_OK = False

if FIREBASE_IMPORT_OK:
    try:
        svc_json = os.getenv("FIREBASE_SERVICE_ACCOUNT")
        if svc_json:
            cred = credentials.Certificate(json.loads(svc_json))
            initialize_app(cred)
            db = firestore.Client()
            FIRESTORE_READY = True
        elif os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            initialize_app()
            db = firestore.Client()
            FIRESTORE_READY = True
    except Exception as e:
        logging.warning("Firestore init skipped: %s", e)

# Postgres (optional)
POOL = None
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
if DATABASE_URL:
    try:
        from psycopg_pool import ConnectionPool
        POOL = ConnectionPool(conninfo=DATABASE_URL, max_size=5, kwargs={"connect_timeout": 5})
        logging.info("Postgres pool enabled.")
    except Exception as e:
        logging.warning("Postgres pool disabled: %s", e)

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing.")
oai = OpenAI(api_key=OPENAI_API_KEY)

# ------------------------------
# FastAPI + CORS
# ------------------------------
app = FastAPI(title="ScentFeed Web AI", version=APP_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------
# Models
# ------------------------------
class PreferencePayload(BaseModel):
    likes: List[str] = Field(default_factory=list)
    dislikes: List[str] = Field(default_factory=list)
    owned: List[str] = Field(default_factory=list)
    wishlist: List[str] = Field(default_factory=list)

class RecommendRequest(BaseModel):
    uid: Optional[str] = None
    goal: str = Field(default="Suggest 3 fragrances.")
    max_results: int = Field(default=3, ge=1, le=5)
    prefs: Optional[PreferencePayload] = None

class Recommendation(BaseModel):
    name: str
    reason: str
    match_score: int = Field(ge=0, le=100)
    notes: Optional[str] = None

class RecommendResponse(BaseModel):
    items: List[Recommendation]
    used_profile: PreferencePayload
    facets: Dict[str, Any] = Field(default_factory=dict)
    request_id: str

class AnalyzeRequest(BaseModel):
    uid: Optional[str] = None
    prefs: Optional[PreferencePayload] = None
    max_tags: int = Field(default=6, ge=3, le=12)

class AnalyzeResponse(BaseModel):
    summary: str
    dominant_notes: List[str] = Field(default_factory=list)
    style_tags: List[str] = Field(default_factory=list)
    occasions: List[str] = Field(default_factory=list)

# ------------------------------
# Health
# ------------------------------
@app.get("/health")
async def health():
    return {"ok": True, "status": "healthy", "version": APP_VERSION, "model": OPENAI_MODEL}

# ------------------------------
# Firestore helpers
# ------------------------------
def _merge_lists(a: Optional[List[str]], b: Optional[List[str]]) -> List[str]:
    out, seen = [], set()
    for lst in (a or []), (b or []):
        for x in lst:
            k = x.strip().lower()
            if k and k not in seen:
                seen.add(k); out.append(x.strip())
    return out

def _collect_ids(uid: str, sub: str) -> List[str]:
    if not (FIRESTORE_READY and db and uid):
        return []
    try:
        docs = db.collection("users").document(uid).collection(sub).stream()
        return [d.id for d in docs]
    except Exception as e:
        logging.warning("Firestore subcollection read failed (%s): %s", sub, e)
        return []

def load_user_prefs(uid: Optional[str], fallback: Optional[PreferencePayload]) -> PreferencePayload:
    likes, dislikes, owned, wishlist = [], [], [], []
    if FIRESTORE_READY and db and uid:
        try:
            doc = db.collection("users").document(uid).get()
            if doc.exists:
                d = doc.to_dict() or {}
                likes = _merge_lists(likes, d.get("likes", []))
                dislikes = _merge_lists(dislikes, d.get("dislikes", []))
                owned = _merge_lists(owned, d.get("owned", []))
                wishlist = _merge_lists(wishlist, d.get("wishlist", []))
            pref_doc = db.collection("users").document(uid).collection("preferences").document("default").get()
            if pref_doc.exists:
                d2 = pref_doc.to_dict() or {}
                likes = _merge_lists(likes, d2.get("likes", []))
                dislikes = _merge_lists(dislikes, d2.get("dislikes", []))
                owned = _merge_lists(owned, d2.get("owned", []))
                wishlist = _merge_lists(wishlist, d2.get("wishlist", []))
            likes = _merge_lists(likes, _collect_ids(uid, "likes"))
            dislikes = _merge_lists(dislikes, _collect_ids(uid, "dislikes"))
            owned = _merge_lists(owned, _collect_ids(uid, "owned"))
            wishlist = _merge_lists(wishlist, _collect_ids(uid, "wishlist"))
        except Exception as e:
            logging.warning("Firestore read failed: %s", e)
    if fallback:
        likes = _merge_lists(likes, fallback.likes)
        dislikes = _merge_lists(dislikes, fallback.dislikes)
        owned = _merge_lists(owned, fallback.owned)
        wishlist = _merge_lists(wishlist, fallback.wishlist)
    return PreferencePayload(likes=likes, dislikes=dislikes, owned=owned, wishlist=wishlist)

# ------------------------------
# Facet extraction (same as before)
# ------------------------------
_COUNTRY_OR_REGION = [
    r"middle\s*east", r"arab(ic|ian)?", r"gulf", r"french", r"italian", r"japanese",
    r"indian", r"pakistan(i)?", r"turkish", r"korean", r"british", r"american",
    r"saudi", r"emirati|uae", r"kuwait(i)?", r"qatar(i)?", r"oman(i)?", r"egypt(ian)?",
]
_OCCASIONS = [
    r"eid", r"ramadan", r"wedding", r"anniversary", r"birthday", r"date", r"office|work",
    r"school|class", r"gym|workout", r"interview", r"party", r"christmas|xmas|holiday",
]
_SEASONS = [r"spring", r"summer", r"fall|autumn", r"winter", r"rain|monsoon"]
_BUDGET = r"(?:under|below|<=?|less than)\s*\$?(\d{2,4})"
_PROJECTION = [r"beast\s*mode", r"strong\s*projection", r"soft\s*projection", r"intimate|skin\s*scent", r"moderate\s*projection"]
_LONGEVITY = [r"long\s*lasting|12\+?\s*h", r"8\s*h", r"all\s*day", r"short\s*lasting|2-3\s*h"]
_NOTES_HINT = [r"vanilla", r"oud|oudh|agarwood", r"rose", r"amber|ambery", r"musk",
               r"citrus|bergamot|lemon|orange|grapefruit|lime|neroli", r"leather|suede",
               r"iris|orris", r"vetiver", r"sandalwood|santal", r"jasmine|jasmin", r"patchouli",
               r"incense|olibanum|frankincense", r"gourmand|chocolate|caramel|praline",
               r"coconut", r"almond|heliotrope", r"cherry|apple|peach|pear",
               r"woody|cedar|guaiac|oak", r"aquatic|marine|ozonic", r"green|fig", r"spicy|pepper|cardamom|cinnamon|clove"]

_BRAND_OR_HOUSE_HINT = r"(?:brand|house)\s*[:\-]\s*([A-Za-z0-9&\.\-\s]{2,})"
_PERFUMER_HINT = r"(?:perfumer|by)\s*[:\-]?\s*([A-Z][A-Za-z\.\-\s]{2,})"

def _extract_facets(goal: str) -> Dict[str, Any]:
    g = goal.lower()
    facets: Dict[str, Any] = {}

    regions = {re.search(p, g, re.IGNORECASE).group(0)
               for p in _COUNTRY_OR_REGION if re.search(p, g, re.IGNORECASE)}
    if regions: facets["regions"] = sorted(regions)

    occs = {re.search(p, g, re.IGNORECASE).group(0)
            for p in _OCCASIONS if re.search(p, g, re.IGNORECASE)}
    if occs: facets["occasions"] = sorted(occs)

    seasons = {re.search(p, g, re.IGNORECASE).group(0)
               for p in _SEASONS if re.search(p, g, re.IGNORECASE)}
    if seasons: facets["seasons"] = sorted(seasons)

    b = re.search(_BUDGET, g, re.IGNORECASE)
    if b:
        try: facets["budget_max"] = int(b.group(1))
        except: pass

    proj = [p for p in _PROJECTION if re.search(p, g, re.IGNORECASE)]
    if proj: facets["projection"] = proj

    longv = [p for p in _LONGEVITY if re.search(p, g, re.IGNORECASE)]
    if longv: facets["longevity"] = longv

    notes = {re.search(p, g, re.IGNORECASE).group(0)
             for p in _NOTES_HINT if re.search(p, g, re.IGNORECASE)}
    if notes: facets["notes"] = sorted(notes)

    bh = re.search(_BRAND_OR_HOUSE_HINT, goal, re.IGNORECASE)
    if bh: facets["brand_or_house"] = bh.group(1).strip()

    pf = re.search(_PERFUMER_HINT, goal, re.IGNORECASE)
    if pf: facets["perfumer"] = pf.group(1).strip()

    return facets

# ------------------------------
# Prompts
# ------------------------------
TOOL_SCHEMA_RECOMMEND = [{
    "type": "function",
    "function": {
        "name": "propose_recommendations",
        "description": "Return 3–5 fragrance recommendations tailored to the user's taste and facets.",
        "parameters": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "reason": {"type": "string"},
                            "match_score": {"type": "integer"},
                            "notes": {"type": "string"}
                        },
                        "required": ["name","reason","match_score"]
                    }
                }
            },
            "required": ["items"]
        }
    }
}]

SYSTEM_PROMPT_RECOMMEND = """You are ScentFeed's fragrance AI.

You MUST interpret user goals as free text possibly containing notes, occasions, regions/countries, brands/houses, perfumers, budget, projection, longevity, season.

REQUIREMENTS:
- Identify all relevant facets and reflect them in selection AND reasoning.
- If region/country is present (e.g., Pakistan, Middle East), prioritize authentic houses/brands or regional styles and EXPLAIN that link in each reason.
- If brand/house or perfumer is named, prioritize direct matches; explain involvement.
- If an occasion exists (Eid, interview, gym, wedding), justify suitability.
- 3 results by default (5 max). match_score 0–100. Keep reasons specific and useful. Include concise 'notes' if helpful.
- Avoid recommending already-owned scents unless exceptions are justified.
- Output ONLY via the 'propose_recommendations' tool.
"""

def _build_user_prompt(goal: str, prefs: PreferencePayload, facets: Dict[str, Any]) -> str:
    def fmt_list(x): return ", ".join(x) if x else "—"
    return (
        f"User goal: {goal}\n\n"
        f"Extracted facets (must guide selection and be stated in reasons): {json.dumps(facets, ensure_ascii=False)}\n\n"
        f"User profile:\n"
        f"- Likes: {fmt_list(prefs.likes)}\n"
        f"- Dislikes: {fmt_list(prefs.dislikes)}\n"
        f"- Owned: {fmt_list(prefs.owned)}\n"
        f"- Wishlist: {fmt_list(prefs.wishlist)}\n\n"
        "Rules:\n"
        "- Return 3 items unless the user asked otherwise.\n"
        "- Reasons must explicitly tie back to facets when present.\n"
        "- If region/country present, call out brand origin/house identity/regional style.\n"
    )

def _reask_note(facets: Dict[str, Any]) -> str:
    asks = []
    if "regions" in facets:
        asks.append("Explicitly state the region/country link for each pick.")
    if "brand_or_house" in facets:
        asks.append("Favor the named brand/house or directly related lines; explain relation.")
    if "perfumer" in facets:
        asks.append("Favor the named perfumer; state authorship.")
    if "occasions" in facets:
        asks.append("Explain why each pick suits the stated occasion(s).")
    if "budget_max" in facets:
        asks.append(f"Respect budget under ${facets['budget_max']} and mention pricing tier.")
    if "notes" in facets:
        asks.append("Mention requested notes explicitly in the reason.")
    return " ".join(asks) or "Tighten reasons to match the goal phrasing."

# ------------------------------
# Request ID + Cache (TTL)
# ------------------------------
_CACHE_TTL_SECONDS = int(os.getenv("RECOMMEND_CACHE_TTL", "600"))  # 10 minutes default
_cache_lock = threading.Lock()
_cache: Dict[str, Dict[str, Any]] = {}  # key -> {"exp": datetime, "payload": dict}

def _make_cache_key(uid: Optional[str], goal: str, prefs: PreferencePayload, max_results: int) -> str:
    blob = json.dumps({
        "uid": uid or "",
        "goal": goal,
        "prefs": prefs.model_dump(),
        "max_results": max_results,
        "model": OPENAI_MODEL,
        "ver": APP_VERSION
    }, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()

def _read_cache(key: str) -> Optional[dict]:
    now = datetime.utcnow()
    with _cache_lock:
        entry = _cache.get(key)
        if not entry:
            return None
        if entry["exp"] < now:
            _cache.pop(key, None)
            return None
        return entry["payload"]

def _write_cache(key: str, payload: dict, ttl: int = _CACHE_TTL_SECONDS):
    with _cache_lock:
        _cache[key] = {"exp": datetime.utcnow() + timedelta(seconds=ttl), "payload": payload}

def _get_request_id(req: Request) -> str:
    hdr = req.headers.get("x-request-id")
    if hdr:
        return hdr.strip()
    # fallback: derive from time
    return f"req-{int(time.time()*1000)}"

# ------------------------------
# Core LLM call
# ------------------------------
async def call_openai_with_tools(goal: str, prefs: PreferencePayload, max_results: int) -> Dict[str, Any]:
    facets = _extract_facets(goal)

    def msgs(extra: str = ""):
        content = _build_user_prompt(goal, prefs, facets)
        if extra:
            content += ("\n\n" + extra)
        return [
            {"role": "system", "content": SYSTEM_PROMPT_RECOMMEND},
            {"role": "user", "content": content}
        ]

    attempts = [
        "",
        _reask_note(facets),
        "STRICT: Every reason must explicitly reference the key facets; if a facet cannot be satisfied, pick a closer alternative and say why."
    ]

    last_err: Optional[Exception] = None
    for note in attempts:
        try:
            resp = oai.chat.completions.create(
                model=OPENAI_MODEL,
                temperature=0.0,
                messages=msgs(note),
                tools=TOOL_SCHEMA_RECOMMEND,
                tool_choice={"type": "function", "function": {"name": "propose_recommendations"}},
            )
            choice = resp.choices[0]
            if not choice.message.tool_calls:
                raise RuntimeError("Model did not call the recommend tool.")
            args = json.loads(choice.message.tool_calls[0].function.arguments or "{}")
            raw_items = args.get("items", [])[:max_results]
            items = [{
                "name": (it.get("name") or "").strip(),
                "reason": (it.get("reason") or "").strip(),
                "match_score": int(it.get("match_score", 0)),
                "notes": ((it.get("notes") or "").strip() or None),
            } for it in raw_items]
            if items:
                return {"items": items, "facets": facets}
        except Exception as e:
            last_err = e
            time.sleep(0.25)
    raise HTTPException(status_code=422, detail=f"Recommendation generation failed. Last error: {last_err}")

# ------------------------------
# Analyze (unchanged)
# ------------------------------
TOOL_SCHEMA_ANALYZE = [{
    "type": "function",
    "function": {
        "name": "propose_profile",
        "description": "Summarize the user's fragrance taste from likes/dislikes/owned/wishlist.",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "dominant_notes": {"type": "array", "items": {"type": "string"}},
                "style_tags": {"type": "array", "items": {"type": "string"}},
                "occasions": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["summary","dominant_notes","style_tags","occasions"]
        }
    }
}]
ANALYZE_SYSTEM_PROMPT = """You are ScentFeed's taste analyst.
- Read likes, dislikes, owned, wishlist.
- Infer a concise taste profile (notes and styles).
- Avoid repeating the raw lists; interpret them.
- Be specific and helpful.
- Return results via the propose_profile tool.
"""

async def call_openai_analyze(prefs: PreferencePayload, max_tags: int) -> AnalyzeResponse:
    user_msg = f"""
Likes: {', '.join(prefs.likes) or '—'}
Dislikes: {', '.join(prefs.dislikes) or '—'}
Owned: {', '.join(prefs.owned) or '—'}
Wishlist: {', '.join(prefs.wishlist) or '—'}

Rules:
- Provide 2–4 sentence summary.
- Suggest concise dominant_notes (<= {max_tags}), style_tags (<= {max_tags}), occasions (<= {max_tags}).
"""
    last_err: Optional[Exception] = None
    for _ in range(2):
        try:
            resp = oai.chat.completions.create(
                model=OPENAI_MODEL,
                temperature=0.4,
                messages=[
                    {"role": "system", "content": ANALYZE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg}
                ],
                tools=TOOL_SCHEMA_ANALYZE,
                tool_choice={"type": "function", "function": {"name": "propose_profile"}}
            )
            choice = resp.choices[0]
            if not choice.message.tool_calls:
                raise RuntimeError("Model did not call the analysis tool.")
            args = json.loads(choice.message.tool_calls[0].function.arguments or "{}")
            summary = str(args.get("summary","")).strip() or "We analyzed your recent activity to identify your core scent preferences."
            dominant_notes = [str(x).strip() for x in args.get("dominant_notes", [])][:max_tags]
            style_tags = [str(x).strip() for x in args.get("style_tags", [])][:max_tags]
            occasions = [str(x).strip() for x in args.get("occasions", [])][:max_tags]
            return AnalyzeResponse(summary=summary, dominant_notes=dominant_notes, style_tags=style_tags, occasions=occasions)
        except Exception as e:
            last_err = e
            time.sleep(0.5)
    raise HTTPException(status_code=503, detail=f"AI analyze failed: {last_err}")

# ------------------------------
# Endpoints with cache + tracing
# ------------------------------
@app.post("/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest, request: Request):
    request_id = _get_request_id(request)
    profile = load_user_prefs(req.uid, req.prefs)
    cache_key = _make_cache_key(req.uid, req.goal, profile, req.max_results)

    cached = _read_cache(cache_key)
    if cached:
        payload = RecommendResponse(
            items=[Recommendation(**it) for it in cached["items"]],
            used_profile=profile,
            facets=cached["facets"],
            request_id=request_id
        )
        return payload

    result = await call_openai_with_tools(req.goal, profile, req.max_results)
    _write_cache(cache_key, result)

    payload = RecommendResponse(
        items=[Recommendation(**it) for it in result["items"]],
        used_profile=profile,
        facets=result["facets"],
        request_id=request_id
    )
    return payload

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest, request: Request):
    # request_id included only in logs (no schema change)
    request_id = _get_request_id(request)
    profile = load_user_prefs(req.uid, req.prefs)
    logging.info("analyze request_id=%s uid=%s", request_id, req.uid or "-")
    return await call_openai_analyze(profile, req.max_tags)




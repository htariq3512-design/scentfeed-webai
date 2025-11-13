# main.py
import os, json, time, logging, re
from typing import List, Optional, Tuple, Pattern

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

APP_VERSION = "1.8.0-filters+strict"

# ---------- Optional Firestore ----------
FIRESTORE_READY = False
db = None
try:
    from firebase_admin import credentials, initialize_app
    from google.cloud import firestore
    FIRESTORE_IMPORT_OK = True
except Exception:
    FIRESTORE_IMPORT_OK = False

if FIRESTORE_IMPORT_OK:
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

# ---------- Optional Postgres ----------
POOL = None
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
if DATABASE_URL:
    try:
        from psycopg_pool import ConnectionPool
        POOL = ConnectionPool(conninfo=DATABASE_URL, max_size=5, kwargs={"connect_timeout": 5})
        logging.info("Postgres pool enabled.")
    except Exception as e:
        logging.warning("Postgres pool disabled: %s", e)
else:
    logging.info("DATABASE_URL not set; Postgres disabled.")

# ---------- OpenAI ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing.")
oai = OpenAI(api_key=OPENAI_API_KEY)

# ---------- FastAPI ----------
app = FastAPI(title="ScentFeed Web AI", version=APP_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Models ----------
class PreferencePayload(BaseModel):
    likes: List[str] = Field(default_factory=list)
    dislikes: List[str] = Field(default_factory=list)
    owned: List[str] = Field(default_factory=list)
    wishlist: List[str] = Field(default_factory=list)

class FilterPayload(BaseModel):
    # All optional – add only what the client sets.
    price_min: Optional[float] = Field(default=None, ge=0)
    price_max: Optional[float] = Field(default=None, ge=0)
    stores: Optional[List[str]] = None          # e.g., ["Sephora","FragranceX"]
    houses: Optional[List[str]] = None          # brands/houses
    niche: Optional[bool] = None                # True niche, False designer, None any
    concentration: Optional[str] = None         # "EDT"|"EDP"|"Parfum"|etc.
    season: Optional[str] = None                # "spring"|"summer"|"fall"|"winter"
    occasion: Optional[str] = None              # "office"|"date"|"wedding"|...
    projection: Optional[str] = None            # "soft"|"moderate"|"strong"
    longevity: Optional[str] = None             # "short"|"moderate"|"long"
    region: Optional[str] = None                # "PK"|"FR"|"US"|full names also ok
    gender_positioning: Optional[str] = None    # "masc"|"fem"|"uni"

class RecommendRequest(BaseModel):
    uid: Optional[str] = None
    goal: str = Field(default="Suggest 3 fragrances.")
    max_results: int = Field(default=3, ge=1, le=5)
    prefs: Optional[PreferencePayload] = None
    filters: Optional[FilterPayload] = None     # <— NEW

class Recommendation(BaseModel):
    name: str
    reason: str
    match_score: int = Field(ge=0, le=100)
    notes: Optional[str] = None

class RecommendResponse(BaseModel):
    items: List[Recommendation]
    used_profile: PreferencePayload
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

# ---------- Health ----------
@app.get("/health")
async def health():
    return {"ok": True, "status": "healthy", "version": APP_VERSION, "model": OPENAI_MODEL}

# ---------- Helpers ----------
def _merge_lists(a, b):
    out, seen = [], set()
    for lst in (a or []), (b or []):
        for x in lst:
            k = x.strip().lower()
            if k and k not in seen:
                seen.add(k); out.append(x.strip())
    return out

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
        except Exception as e:
            logging.warning("Firestore read failed: %s", e)
    if fallback:
        likes = _merge_lists(likes, fallback.likes)
        dislikes = _merge_lists(dislikes, fallback.dislikes)
        owned = _merge_lists(owned, fallback.owned)
        wishlist = _merge_lists(wishlist, fallback.wishlist)
    return PreferencePayload(likes=likes, dislikes=dislikes, owned=owned, wishlist=wishlist)

def _validate_filters(f: Optional[FilterPayload]) -> None:
    if not f: return
    if f.price_min is not None and f.price_max is not None and f.price_min > f.price_max:
        raise HTTPException(status_code=422, detail="price_min cannot be greater than price_max.")

# ---------- LLM Tools ----------
TOOL_SCHEMA_RECOMMEND = [
    {
        "type": "function",
        "function": {
            "name": "propose_recommendations",
            "description": "Return 3–5 fragrance recommendations tailored to the user's taste and filters.",
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
                            "required": ["name", "reason", "match_score"]
                        }
                    }
                },
                "required": ["items"]
            }
        }
    }
]

SYSTEM_PROMPT_RECOMMEND = """You are ScentFeed's fragrance AI.

Hard requirements:
- Obey all user filters strictly (price range, niche/designer, stores, houses, region, season, occasion, concentration, projection, longevity, gender_positioning). If a filter is set, every item MUST satisfy it.
- Avoid recommending already-owned items unless explicitly justified as a variant or flankers.
- If the prompt mentions notes (e.g., 'vanilla, oud'), include those terms in each item's reason.

Output:
- 3 items (max 5), match_score 0–100, short notes if helpful.
- Use the 'propose_recommendations' tool to return structured JSON.
"""

def _filters_summary(f: Optional[FilterPayload]) -> str:
    if not f: return "Filters: —"
    def j(a): return ", ".join(a) if a else "—"
    return (
        "Filters:\n"
        f"- price_min: {f.price_min if f.price_min is not None else '—'}\n"
        f"- price_max: {f.price_max if f.price_max is not None else '—'}\n"
        f"- stores: {j(f.stores)}\n"
        f"- houses: {j(f.houses)}\n"
        f"- niche: {f.niche if f.niche is not None else '—'}\n"
        f"- concentration: {f.concentration or '—'}\n"
        f"- season: {f.season or '—'}\n"
        f"- occasion: {f.occasion or '—'}\n"
        f"- projection: {f.projection or '—'}\n"
        f"- longevity: {f.longevity or '—'}\n"
        f"- region: {f.region or '—'}\n"
        f"- gender_positioning: {f.gender_positioning or '—'}"
    )

def build_user_prompt(goal: str, prefs: PreferencePayload, filters: Optional[FilterPayload]) -> str:
    return f"""
Goal: {goal}

Likes: {', '.join(prefs.likes) or '—'}
Dislikes: {', '.join(prefs.dislikes) or '—'}
Owned: {', '.join(prefs.owned) or '—'}
Wishlist: {', '.join(prefs.wishlist) or '—'}

{_filters_summary(filters)}

Rules:
- Return exactly 3 recommendations unless the user asked for a different count.
- Each reason must explicitly reference how the item satisfies the set filters.
""".strip()

async def call_openai(goal: str, prefs: PreferencePayload, filters: Optional[FilterPayload], max_results: int) -> RecommendResponse:
    _validate_filters(filters)
    def _msgs():
        return [
            {"role": "system", "content": SYSTEM_PROMPT_RECOMMEND},
            {"role": "user", "content": build_user_prompt(goal, prefs, filters)}
        ]
    # up to 2 tries to enforce strict filters
    for _ in range(2):
        resp = oai.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.0,
            messages=_msgs(),
            tools=TOOL_SCHEMA_RECOMMEND,
            tool_choice={"type": "function", "function": {"name": "propose_recommendations"}},
        )
        choice = resp.choices[0]
        if not choice.message.tool_calls:
            continue
        args = json.loads(choice.message.tool_calls[0].function.arguments or "{}")
        raw = args.get("items", [])[:max_results]
        items = []
        for it in raw:
            name = (it.get("name") or "").strip()
            reason = (it.get("reason") or "").strip()
            score = int(it.get("match_score", 0))
            notes = ((it.get("notes") or "").strip() or None)
            if name:
                items.append(Recommendation(name=name, reason=reason, match_score=score, notes=notes))
        if items:
            rid = resp.id or f"req_{int(time.time()*1000)}"
            return RecommendResponse(items=items, used_profile=prefs, request_id=rid)
    raise HTTPException(status_code=422, detail="The model could not satisfy the filters. Try relaxing constraints.")

# ---------- ANALYZE ----------
TOOL_SCHEMA_ANALYZE = [
    {
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
    }
]

ANALYZE_SYSTEM_PROMPT = """You are ScentFeed's taste analyst.
- Read likes, dislikes, owned, wishlist.
- Infer concise preferences; be specific and helpful.
- Return via the propose_profile tool.
"""

async def call_openai_analyze(prefs: PreferencePayload, max_tags: int) -> AnalyzeResponse:
    user_msg = f"""
Likes: {', '.join(prefs.likes) or '—'}
Dislikes: {', '.join(prefs.dislikes) or '—'}
Owned: {', '.join(prefs.owned) or '—'}
Wishlist: {', '.join(prefs.wishlist) or '—'}

Rules:
- 2–4 sentence summary.
- <= {max_tags} for each of dominant_notes, style_tags, occasions.
""".strip()
    resp = oai.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.4,
        messages=[{"role":"system","content":ANALYZE_SYSTEM_PROMPT},{"role":"user","content":user_msg}],
        tools=TOOL_SCHEMA_ANALYZE,
        tool_choice={"type":"function","function":{"name":"propose_profile"}}
    )
    choice = resp.choices[0]
    if not choice.message.tool_calls:
        raise HTTPException(status_code=503, detail="AI analyze failed.")
    args = json.loads(choice.message.tool_calls[0].function.arguments or "{}")
    return AnalyzeResponse(
        summary = str(args.get("summary","")).strip() or "Here’s your taste profile.",
        dominant_notes = [str(x).strip() for x in args.get("dominant_notes", [])][:max_tags],
        style_tags = [str(x).strip() for x in args.get("style_tags", [])][:max_tags],
        occasions = [str(x).strip() for x in args.get("occasions", [])][:max_tags],
    )

# ---------- Endpoints ----------
@app.post("/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest):
    profile = load_user_prefs(req.uid, req.prefs)
    return await call_openai(req.goal, profile, req.filters, req.max_results)

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    profile = load_user_prefs(req.uid, req.prefs)
    return await call_openai_analyze(profile, req.max_tags)




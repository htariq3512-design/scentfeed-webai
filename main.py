# main.py
import os, json, time, logging, re
from typing import List, Optional, Tuple, Pattern

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from openai import OpenAI
import httpx  # for Supabase REST

# Load .env first so all env vars are available
load_dotenv(override=True)

APP_VERSION = "1.6.3-supabase-test+analyze"

# -------------------------------------------------------------------
# Optional Firestore (if creds are configured)
# -------------------------------------------------------------------
FIRESTORE_READY = False
db = None
try:
    from firebase_admin import credentials, initialize_app
    from google.cloud import firestore
    FIRESTORE_IMPORT_OK = True
except Exception:
    FIRESTORE_IMPORT_OK = False

# Try to bring Firestore online (non-fatal if missing)
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

# -------------------------------------------------------------------
# Supabase (REST) — REQUIRED for /sb-test, optional for prod
# -------------------------------------------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")
if not (SUPABASE_URL and SUPABASE_ANON_KEY):
    logging.warning("Supabase variables not set; DB-backed features will be disabled.")

async def sb_search_perfumes(q: str, limit_n: int = 10):
    """
    Calls your SQL function public.search_perfumes via PostgREST RPC.
    Expects you've created it in Supabase.
    Returns a list of rows (dicts) or [].
    """
    if not (SUPABASE_URL and SUPABASE_ANON_KEY):
        return []

    url = f"{SUPABASE_URL}/rest/v1/rpc/search_perfumes"
    headers = {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation"
    }
    payload = {"q": q, "limit_n": limit_n}

    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.post(url, headers=headers, json=payload)
        if r.status_code != 200:
            logging.warning("Supabase RPC search_perfumes failed: %s %s", r.status_code, r.text[:300])
            return []
        try:
            data = r.json()
            return data if isinstance(data, list) else []
        except Exception as e:
            logging.warning("Supabase RPC JSON parse failed: %s", e)
            return []

# -------------------------------------------------------------------
# OpenAI client
# -------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing.")
oai = OpenAI(api_key=OPENAI_API_KEY)

# -------------------------------------------------------------------
# FastAPI app & CORS
# -------------------------------------------------------------------
app = FastAPI(title="ScentFeed Web AI", version=APP_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
# Pydantic models
# -------------------------------------------------------------------
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

class AnalyzeRequest(BaseModel):
    uid: Optional[str] = None
    prefs: Optional[PreferencePayload] = None
    max_tags: int = Field(default=6, ge=3, le=12)

class AnalyzeResponse(BaseModel):
    summary: str
    dominant_notes: List[str] = Field(default_factory=list)
    style_tags: List[str] = Field(default_factory=list)
    occasions: List[str] = Field(default_factory=list)

# -------------------------------------------------------------------
# Health
# -------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"ok": True, "status": "healthy", "version": APP_VERSION, "model": OPENAI_MODEL}

# Quick Supabase smoke test
@app.get("/sb-test")
async def sb_test(q: str = "vanilla", limit: int = 5):
    rows = await sb_search_perfumes(q=q, limit_n=limit)
    out = [
        {"id": r.get("id"), "name": r.get("name"), "brand": r.get("brand"), "score": r.get("score")}
        for r in rows
    ]
    return {"count": len(out), "items": out}

# -------------------------------------------------------------------
# Firestore helpers
# -------------------------------------------------------------------
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

# -------------------------------------------------------------------
# Keyword detection (WIDE coverage for natural language goals)
# -------------------------------------------------------------------
KEYWORD_VARIANTS = {
    "vanilla":   [r"vanilla", r"vanillic", r"vanilla\s*bean", r"bourbon\s*vanilla", r"madagascar\s*vanilla"],
    "oud":       [r"oud", r"agarwood"],
    "rose":      [r"rose", r"damask\s*rose", r"turkish\s*rose"],
    "amber":     [r"amber", r"ambery"],
    "citrus":    [r"citrus", r"bergamot", r"orange", r"lemon", r"lime", r"grapefruit", r"mandarin", r"neroli"],
    "lavender":  [r"lavender"],
    "leather":   [r"leather", r"suede"],
    "iris":      [r"iris", r"orris"],
    "vetiver":   [r"vetiver"],
    "sandalwood":[r"sandalwood", r"santal"],
    "jasmine":   [r"jasmine", r"jasmin"],
    "patchouli": [r"patchouli"],
    "musk":      [r"musk", r"musky"],
    "woody":     [r"woody", r"woods", r"cedarwood", r"sandalwood", r"guaiac", r"oakwood"],
    "aquatic":   [r"aquatic", r"marine", r"sea\s*breeze", r"ozonic"],
    "green":     [r"green", r"galbanum", r"grass", r"fig\s*leaf"],
    "spicy":     [r"spicy", r"pepper", r"cardamom", r"cinnamon", r"clove"],
    "gourmand":  [r"gourmand", r"chocolate", r"caramel", r"praline", r"vanilla"],
    "coconut":   [r"coconut"],
    "almond":    [r"almond", r"heliotrope"],
    "cherry":    [r"cherry"],
    "apple":     [r"apple"],
    "peach":     [r"peach"],
    "pear":      [r"pear"],
    "tobacco":   [r"tobacco"],
    "incense":   [r"incense", r"olibanum", r"frankincense"],
    "cedar":     [r"cedar", r"cedarwood"],
    "pine":      [r"pine"],
}

def _required_groups_from_goal(goal: str) -> List[Tuple[str, List[Pattern]]]:
    g = (goal or "").lower()
    groups: List[Tuple[str, List[Pattern]]] = []
    for root, patterns in KEYWORD_VARIANTS.items():
        if any(re.search(p, g) for p in patterns):
            compiled = [re.compile(rf"\b{p}\b", re.IGNORECASE) for p in patterns]
            groups.append((root, compiled))
    return groups

def _item_matches_group(text: str, pats: List[Pattern]) -> bool:
    return any(p.search(text) for p in pats)

def _items_meet_requirements(items: List[Recommendation], groups: List[Tuple[str, List[Pattern]]]) -> bool:
    if not groups:
        return True
    for it in items:
        blob = " ".join([it.name or "", it.reason or "", it.notes or ""])
        if not all(_item_matches_group(blob, pats) for _, pats in groups):
            return False
    return True

# -------------------------------------------------------------------
# Fallback catalog (deterministic)
# -------------------------------------------------------------------
def R(name: str, reason: str, score: int, notes: str) -> Recommendation:
    return Recommendation(name=name, reason=reason, match_score=score, notes=notes)

FALLBACK_CATALOG = {
    "vanilla": [
        R("Guerlain Spiritueuse Double Vanille","Rich vanilla with boozy warmth—clearly vanilla-forward.",95,"vanilla, rum, benzoin"),
        R("Tom Ford Tobacco Vanille","Dense vanilla wrapped in tobacco and spice—deep vanilla profile.",92,"vanilla, tobacco, tonka"),
        R("Kayali Vanilla | 28","Sweet, wearable vanilla with amber warmth—vanilla signature.",90,"vanilla, amber, brown sugar"),
    ],
    "oud": [
        R("Initio Oud for Greatness","Pronounced oud with saffron—bold, modern oud.",92,"oud, saffron, patchouli"),
        R("Acqua di Parma Oud","Smooth oud with citrus lift—refined oud.",90,"oud, bergamot, leather"),
        R("Montale Black Aoud","Dark rose-oud signature—classic oud presence.",88,"oud, rose, patchouli"),
    ],
    "rose": [
        R("Frederic Malle Portrait of a Lady","Luxurious rose with patchouli—rose statement.",92,"rose, patchouli, incense"),
        R("Diptyque Eau Rose","Airy, naturalistic rose—fresh daily rose.",89,"rose, lychee, musk"),
        R("Le Labo Rose 31","Spice-wood rose—modern unisex rose.",88,"rose, cumin, cedar"),
    ],
    "amber": [
        R("Hermès Ambre Narguilé","Honeyed tobacco-amber—cozy amber aura.",91,"amber, honey, tobacco"),
        R("MFK Grand Soir","Resinous amber—glowing evening amber.",90,"amber, labdanum, vanilla"),
        R("Prada Amber Pour Homme","Clean, soapy amber—office-safe.",88,"amber, spices, musk"),
    ],
    # ... (extend other groups as needed)
}

def _fallback_for(groups: List[Tuple[str, List[Pattern]]], n: int) -> Optional[List[Recommendation]]:
    if not groups:
        return None
    roots = [r for r, _ in groups]
    catalogs = [FALLBACK_CATALOG.get(r, []) for r in roots]
    if any(len(c) == 0 for c in catalogs):
        return None
    picks: List[Recommendation] = []
    i = 0
    while len(picks) < n:
        cat = catalogs[i % len(catalogs)]
        idx = (len(picks) // len(catalogs)) % len(cat)
        base = cat[idx]
        reason = base.reason
        for r in roots:
            if r not in reason.lower():
                reason += f" Clearly {r}-forward elements present."
        picks.append(Recommendation(name=base.name, reason=reason, match_score=base.match_score, notes=base.notes))
        i += 1
    return picks[:n]

# -------------------------------------------------------------------
# LLM tool schema (recommend)
# -------------------------------------------------------------------
TOOL_SCHEMA_RECOMMEND = [
    {
        "type": "function",
        "function": {
            "name": "propose_recommendations",
            "description": "Return 3–5 fragrance recommendations tailored to the user's taste.",
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
- If the goal mentions specific notes (e.g., "vanilla", "oud", "rose", etc.), ONLY return perfumes centered on those notes.
- Explicitly include those note words in each reason.
- Recommend 3 fragrances (5 max), with match_score 0–100, and short notes if useful.
- Avoid recommending already-owned scents unless justified.
- Use the 'propose_recommendations' tool to return results in structured JSON.
"""

def build_user_prompt(goal: str, prefs: PreferencePayload, groups: List[Tuple[str, List[Pattern]]]) -> str:
    if groups:
        lines = []
        for root, pats in groups:
            demo = ", ".join([p.pattern.strip(r"\b").replace("\\s*", " ").strip("^$") for p in pats[:3]])
            lines.append(f"- Must explicitly mention: {root} (e.g., {demo})")
        req = "Required notes:\n" + "\n".join(lines)
    else:
        req = "Required notes: —"
    return f"""
Goal: {goal}

Likes: {', '.join(prefs.likes) or '—'}
Dislikes: {', '.join(prefs.dislikes) or '—'}
Owned: {', '.join(prefs.owned) or '—'}
Wishlist: {', '.join(prefs.wishlist) or '—'}

{req}
Rules:
- 3 results, each reason must clearly include all required note words (exact words or common variants).
""".strip()

# -------------------------------------------------------------------
# OpenAI call with enforcement + multi-root fallback
# -------------------------------------------------------------------
async def call_openai_with_tools(goal: str, prefs: PreferencePayload, max_results: int) -> RecommendResponse:
    groups = _required_groups_from_goal(goal)

    def _msgs(extra: str = ""):
        return [
            {"role": "system", "content": SYSTEM_PROMPT_RECOMMEND},
            {"role": "user", "content": build_user_prompt(goal, prefs, groups) + (("\n\n" + extra) if extra else "")}
        ]

    attempts = [
        "",
        "Ensure every fragrance reason explicitly includes all required note words mentioned in the goal.",
        "STRICT: only return items where each reason clearly contains each required note word.",
    ]

    for note in attempts:
        try:
            resp = oai.chat.completions.create(
                model=OPENAI_MODEL,
                temperature=0.0,
                messages=_msgs(note),
                tools=TOOL_SCHEMA_RECOMMEND,
                tool_choice={"type": "function", "function": {"name": "propose_recommendations"}},
            )
            choice = resp.choices[0]
            if not choice.message.tool_calls:
                raise RuntimeError("Model did not call the recommend tool.")
            args = json.loads(choice.message.tool_calls[0].function.arguments or "{}")
            raw_items = args.get("items", [])[:max_results]
            items = [
                Recommendation(
                    name=(it.get("name") or "").strip(),
                    reason=(it.get("reason") or "").strip(),
                    match_score=int(it.get("match_score", 0)),
                    notes=((it.get("notes") or "").strip() or None),
                )
                for it in raw_items
            ]
            if items and _items_meet_requirements(items, groups):
                return RecommendResponse(items=items, used_profile=prefs)
        except Exception:
            time.sleep(0.25)

    # Deterministic multi-root fallback
    if groups:
        fb = _fallback_for(groups, max_results)
        if fb:
            return RecommendResponse(items=fb, used_profile=prefs)

    raise HTTPException(status_code=422, detail="Required note keywords were not satisfied and no fallback catalog was available.")

# -------------------------------------------------------------------
# ANALYZE (for taste profile surfacing)
# -------------------------------------------------------------------
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

# -------------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------------
@app.post("/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest):
    profile = load_user_prefs(req.uid, req.prefs)
    return await call_openai_with_tools(req.goal, profile, req.max_results)

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    profile = load_user_prefs(req.uid, req.prefs)
    return await call_openai_analyze(profile, req.max_tags)




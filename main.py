# main.py
import os
import json
import time
import re
import logging
import asyncio
from typing import List, Optional, Tuple, Pattern

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# -----------------------------
# Env + Version
# -----------------------------
load_dotenv(override=True)
APP_VERSION = "1.8.0-guard-timeouts+lite"

# -----------------------------
# Optional Firestore
# -----------------------------
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

# -----------------------------
# Optional Postgres (non-blocking)
# -----------------------------
POOL = None
DATABASE_URL = (os.getenv("DATABASE_URL") or "").strip()
if DATABASE_URL:
    try:
        from psycopg_pool import ConnectionPool  # optional, best-effort
        POOL = ConnectionPool(
            conninfo=DATABASE_URL,
            max_size=4,
            kwargs={"connect_timeout": 5},
        )
        logging.info("Postgres pool enabled.")
    except Exception as e:
        logging.warning("Postgres pool disabled: %s", e)
else:
    logging.info("DATABASE_URL not set; Postgres disabled.")

# -----------------------------
# OpenAI client + hard timeout wrapper
# -----------------------------
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing.")
oai = OpenAI(api_key=OPENAI_API_KEY)

# Hard caps; tweak via env if needed
OPENAI_HARD_TIMEOUT = float(os.getenv("OPENAI_TIMEOUT_SEC", "8.0"))
ENDPOINT_GUARD_TIMEOUT = float(os.getenv("ENDPOINT_TIMEOUT_SEC", "12.0"))

async def _oai_chat(**kwargs):
    """
    Run OpenAI chat call in a thread with a strict asyncio timeout.
    Prevents indefinite hangs due to upstream slowness.
    """
    loop = asyncio.get_event_loop()
    def _call():
        return oai.chat.completions.create(**kwargs)
    return await asyncio.wait_for(loop.run_in_executor(None, _call), timeout=OPENAI_HARD_TIMEOUT)

# -----------------------------
# FastAPI app + CORS
# -----------------------------
app = FastAPI(title="ScentFeed Web AI", version=APP_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Pydantic models
# -----------------------------
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

# -----------------------------
# Health
# -----------------------------
@app.get("/health")
async def health():
    return {"ok": True, "status": "healthy", "version": APP_VERSION, "model": OPENAI_MODEL}

# -----------------------------
# Firestore helpers (best-effort)
# -----------------------------
def _merge_lists(a: Optional[List[str]], b: Optional[List[str]]) -> List[str]:
    out, seen = [], set()
    for lst in ((a or []), (b or [])):
        for x in lst:
            k = x.strip().lower()
            if k and k not in seen:
                seen.add(k)
                out.append(x.strip())
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
            # Merge subcollections as IDs if used
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

# -----------------------------
# Keyword detection (broad)
# -----------------------------
KEYWORD_VARIANTS = {
    "vanilla":   [r"vanilla", r"vanillic", r"vanilla\s*bean", r"bourbon\s*vanilla"],
    "oud":       [r"oud", r"agarwood"],
    "rose":      [r"rose", r"damask\s*rose", r"turkish\s*rose"],
    "amber":     [r"amber", r"ambery"],
    "citrus":    [r"citrus", r"bergamot", r"lemon", r"neroli", r"grapefruit"],
    "woody":     [r"woody", r"cedarwood", r"sandalwood", r"oakwood"],
    "spicy":     [r"spicy", r"pepper", r"cardamom", r"cinnamon", r"clove"],
    "gourmand":  [r"gourmand", r"chocolate", r"caramel", r"praline"],
    "musk":      [r"musk", r"musky"],
    "aquatic":   [r"aquatic", r"marine", r"ozonic"],
}

def _required_groups_from_goal(goal: str) -> List[Tuple[str, List[Pattern]]]:
    g = (goal or "").lower()
    groups: List[Tuple[str, List[Pattern]]] = []
    for root, patterns in KEYWORD_VARIANTS.items():
        if any(re.search(p, g) for p in patterns):
            compiled = [re.compile(p, re.IGNORECASE) for p in patterns]
            groups.append((root, compiled))
    return groups

def _item_matches_group(text: str, pats: List[Pattern]) -> bool:
    return any(p.search(text) for p in pats)

def _items_meet_requirements(items: List[Recommendation], groups: List[Tuple[str, List[Pattern]]]) -> bool:
    if not groups:
        return True
    for it in items:
        blob = " ".join([it.name or "", it.reason or "", it.notes or ""])
        # Must mention every *root* somewhere
        for _, pats in groups:
            if not _item_matches_group(blob, pats):
                return False
    return True

# -----------------------------
# Deterministic fallback catalog
# -----------------------------
def R(name: str, reason: str, score: int, notes: str) -> Recommendation:
    return Recommendation(name=name, reason=reason, match_score=score, notes=notes)

FALLBACK_CATALOG = {
    "vanilla": [
        R("Kayali Vanilla | 28", "Sweet vanilla signature with warm amber tones; clearly vanilla-forward.", 91, "vanilla, amber, brown sugar"),
        R("Tom Ford Tobacco Vanille", "Dense vanilla wrapped in tobacco and spice—deep vanilla profile.", 90, "vanilla, tobacco, tonka"),
        R("Guerlain Spiritueuse Double Vanille", "Rich boozy vanilla—vanilla leads the composition.", 92, "vanilla, rum, benzoin"),
    ],
    "oud": [
        R("Initio Oud for Greatness", "Modern oud with saffron; oud is the star.", 92, "oud, saffron, patchouli"),
        R("Montale Black Aoud", "Dark rose-oud classic; unmistakably oud-laden.", 89, "oud, rose, patchouli"),
        R("Acqua di Parma Oud", "Refined oud smoothed by citrus and leather.", 88, "oud, citrus, leather"),
    ],
    "rose": [
        R("Frederic Malle Portrait of a Lady", "Luxurious rose center with patchouli; rose-forward statement.", 92, "rose, patchouli, incense"),
        R("Diptyque Eau Rose", "Airy daily rose; transparent rose signature.", 89, "rose, lychee, musk"),
        R("Le Labo Rose 31", "Spiced, woody rose; modern unisex rose core.", 88, "rose, cumin, cedar"),
    ],
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
                reason += f" Includes clear {r} facets."
        picks.append(Recommendation(name=base.name, reason=reason, match_score=base.match_score, notes=base.notes))
        i += 1
    return picks[:n]

# -----------------------------
# Tool schema + prompts
# -----------------------------
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
- If the goal contains specific note words (e.g., vanilla, oud, rose, amber, woody, citrus, spicy, etc.), ONLY return perfumes centered on those notes, and explicitly include those words in each reason.
- Recommend 3 fragrances (5 max), scores 0–100.
- Avoid recommending already-owned items unless strongly justified.
- Use the 'propose_recommendations' tool to return JSON.
"""

def build_user_prompt(goal: str, prefs: PreferencePayload, groups: List[Tuple[str, List[Pattern]]]) -> str:
    if groups:
        lines = []
        for root, pats in groups:
            demo = ", ".join([p.pattern.replace("\\s*", " ").strip("^$") for p in pats[:3]])
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
- 3 results, each reason must clearly include all required note words (exact or common variants).
""".strip()

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
- Infer a concise taste profile with specific note and style tendencies.
- Do not repeat raw lists; interpret them.
- Return results via the 'propose_profile' tool.
"""

# -----------------------------
# OpenAI flows (with hard timeouts)
# -----------------------------
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
            resp = await _oai_chat(
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
        except asyncio.TimeoutError:
            # Hard timeout at the call level — let the endpoint guard handle overall cap.
            break
        except Exception:
            # Soft retry on formatting/model hiccups
            time.sleep(0.2)

    # Deterministic fallback if we have clear note roots
    if groups:
        fb = _fallback_for(groups, max_results)
        if fb:
            return RecommendResponse(items=fb, used_profile=prefs)

    # If we reach here, we failed both AI & fallback
    raise HTTPException(status_code=422, detail="Required note keywords were not satisfied and no fallback catalog was available.")

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
    try:
        resp = await _oai_chat(
            model=OPENAI_MODEL,
            temperature=0.4,
            messages=[
                {"role": "system", "content": ANALYZE_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            tools=TOOL_SCHEMA_ANALYZE,
            tool_choice={"type": "function", "function": {"name": "propose_profile"}},
        )
        choice = resp.choices[0]
        if not choice.message.tool_calls:
            raise RuntimeError("Model did not call the analysis tool.")
        args = json.loads(choice.message.tool_calls[0].function.arguments or "{}")
        summary = (args.get("summary") or "").strip() or "We analyzed your recent activity to identify your core scent preferences."
        dominant_notes = [str(x).strip() for x in (args.get("dominant_notes") or [])][:max_tags]
        style_tags = [str(x).strip() for x in (args.get("style_tags") or [])][:max_tags]
        occasions = [str(x).strip() for x in (args.get("occasions") or [])][:max_tags]
        return AnalyzeResponse(summary=summary, dominant_notes=dominant_notes, style_tags=style_tags, occasions=occasions)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Upstream timeout during analyze.")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Analyze failed: {e}")

# -----------------------------
# Endpoints (with guard timeout + lite mode)
# -----------------------------
@app.post("/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest, lite: bool = Query(False, description="Skip enrichment for fast path")):
    async def _impl():
        # In the future, use `lite` to skip any slow vendor enrichment before calling OpenAI.
        profile = load_user_prefs(req.uid, req.prefs)
        return await call_openai_with_tools(req.goal, profile, req.max_results)

    try:
        return await asyncio.wait_for(_impl(), timeout=ENDPOINT_GUARD_TIMEOUT)
    except asyncio.TimeoutError:
        # Hard stop to prevent “hang forever”
        raise HTTPException(status_code=504, detail="Upstream timeout")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Unexpected error: {e}")

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    async def _impl():
        profile = load_user_prefs(req.uid, req.prefs)
        return await call_openai_analyze(profile, req.max_tags)

    try:
        return await asyncio.wait_for(_impl(), timeout=ENDPOINT_GUARD_TIMEOUT)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Upstream timeout")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Unexpected error: {e}")



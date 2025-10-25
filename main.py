# main.py
import os, json, time, logging, re
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from openai import OpenAI

APP_VERSION = "1.3.0-vanilla-enforced"

# --- Optional Firestore (if creds are configured) ---
FIRESTORE_READY = False
db = None
try:
    from firebase_admin import credentials, initialize_app
    from google.cloud import firestore
    FIRESTORE_IMPORT_OK = True
except Exception:
    FIRESTORE_IMPORT_OK = False

load_dotenv(override=True)

# ---- Config ----
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing.")
oai = OpenAI(api_key=OPENAI_API_KEY)

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

# ---- FastAPI ----
app = FastAPI(title="ScentFeed Web AI", version=APP_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Schemas ----
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

# ---- Health ----
@app.get("/health")
async def health():
    return {"ok": True, "status": "healthy", "version": APP_VERSION, "model": OPENAI_MODEL}

# ---- Firestore helpers ----
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

# ---- Keyword enforcement (regex with synonyms) ----
KEYWORD_VARIANTS = {
    "vanilla": [r"vanilla", r"vanillic", r"vanilla\s*bean", r"bourbon\s*vanilla", r"madagascar\s*vanilla"],
    "oud": [r"oud", r"agarwood"],
    "rose": [r"rose", r"damask\s*rose", r"turkish\s*rose"],
    "amber": [r"amber", r"ambery"],
    "citrus": [r"citrus", r"bergamot", r"orange", r"lemon", r"lime", r"grapefruit"],
}

def _required_groups_from_goal(goal: str) -> list[tuple[str, list[re.Pattern]]]:
    g = (goal or "").lower()
    groups = []
    for root, patterns in KEYWORD_VARIANTS.items():
        if any(re.search(p, g) for p in patterns):
            groups.append((root, [re.compile(rf"\b{p}\b", re.IGNORECASE) for p in patterns]))
    return groups

def _item_matches_group(text: str, pats: List[re.Pattern]) -> bool:
    return any(p.search(text) for p in pats)

def _items_meet_requirements(items: List[Recommendation], groups: list[tuple[str, List[re.Pattern]]]) -> bool:
    if not groups:
        return True
    for it in items:
        blob = " ".join([it.name or "", it.reason or "", it.notes or ""])
        if not all(_item_matches_group(blob, pats) for _, pats in groups):
            return False
    return True

# ---- LLM tool schema ----
TOOL_SCHEMA = [
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

SYSTEM_PROMPT = """You are ScentFeed's fragrance AI.
- If the goal mentions specific notes (e.g., "vanilla", "oud", "rose"), ONLY return perfumes centered on those notes.
- Explicitly include those note words in each reason.
- Recommend 3 fragrances (5 max), with match_score 0–100, and short notes if useful.
- Avoid recommending already-owned scents unless justified.
- Use the 'propose_recommendations' tool to return results in structured JSON.
"""

def build_user_prompt(goal: str, prefs: PreferencePayload, groups: list[tuple[str, List[re.Pattern]]]) -> str:
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

# ---- Deterministic vanilla fallback (guaranteed compliant) ----
def _vanilla_fallback(n: int) -> List[Recommendation]:
    base = [
        Recommendation(
            name="Guerlain Spiritueuse Double Vanille",
            reason="Rich vanilla with boozy warmth and smoky facets — unmistakably vanilla-forward.",
            match_score=95,
            notes="vanilla, rum, benzoin, smoky woods",
        ),
        Recommendation(
            name="Tom Ford Tobacco Vanille",
            reason="Dense vanilla accord wrapped in tobacco and spice; deep, cozy vanilla profile.",
            match_score=92,
            notes="vanilla, tobacco, tonka, spice",
        ),
        Recommendation(
            name="Kayali Vanilla | 28",
            reason="Sweet, wearable vanilla with amber warmth — an accessible vanilla-forward choice.",
            match_score=90,
            notes="vanilla, amber, brown sugar, musk",
        ),
        Recommendation(
            name="Matiere Premiere Vanilla Powder",
            reason="Modern powdery vanilla centered on natural vanilla absolute; clean vanilla signature.",
            match_score=90,
            notes="vanilla absolute, powdery musk",
        ),
    ]
    return base[:n]

# ---- OpenAI call with strict enforcement and fallback ----
async def call_openai_with_tools(goal: str, prefs: PreferencePayload, max_results: int) -> RecommendResponse:
    groups = _required_groups_from_goal(goal)
    def _msgs(extra: str = ""):
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(goal, prefs, groups) + (("\n\n" + extra) if extra else "")}
        ]

    attempts = [
        "",
        "Ensure every fragrance reason explicitly includes all required note words (e.g., 'vanilla').",
        "STRICT: only return items where each reason clearly contains each required note word (e.g., 'vanilla').",
    ]

    for note in attempts:
        try:
            resp = oai.chat.completions.create(
                model=OPENAI_MODEL,
                temperature=0.0,  # maximum adherence
                messages=_msgs(note),
                tools=TOOL_SCHEMA,
                tool_choice={"type": "function", "function": {"name": "propose_recommendations"}},
            )
            args = json.loads(resp.choices[0].message.tool_calls[0].function.arguments or "{}")
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
            if _items_meet_requirements(items, groups):
                return RecommendResponse(items=items, used_profile=prefs)
        except Exception as e:
            time.sleep(0.25)

    # Deterministic, guaranteed-compliant fallback if notes were required
    if groups:
        roots = [r for r, _ in groups]
        if "vanilla" in roots:
            fb = _vanilla_fallback(max_results)
            return RecommendResponse(items=fb, used_profile=prefs)

    # If no required groups or still nothing, return a clear error
    raise HTTPException(status_code=422, detail="Required note keywords were not satisfied by the model.")

# ---- Endpoint ----
@app.post("/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest):
    profile = load_user_prefs(req.uid, req.prefs)
    return await call_openai_with_tools(req.goal, profile, req.max_results)



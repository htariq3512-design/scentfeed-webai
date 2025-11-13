# main.py
# ScentFeed — Web AI (Recommend + Analyz)
# Billion-dollar version: strict INTENT enforcement, deterministic fallbacks, safe Firestore merge, clean retries.

import os
import json
import time
import logging
import re
from typing import List, Optional, Tuple, Pattern, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# -----------------------------
# Load env FIRST
# -----------------------------
load_dotenv(override=True)
APP_VERSION = "2.0.0-intent-guarded"

# -----------------------------
# Optional Firestore (non-fatal)
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
            # Expecting a JSON string of service account
            cred = credentials.Certificate(json.loads(svc_json))
            initialize_app(cred)
            db = firestore.Client()
            FIRESTORE_READY = True
        elif os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            # File path provided via standard Google env var
            initialize_app()
            db = firestore.Client()
            FIRESTORE_READY = True
    except Exception as e:
        logging.warning("Firestore init skipped: %s", e)

# -----------------------------
# OpenAI client
# -----------------------------
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing.")
oai = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------
# FastAPI + CORS
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
# Pydantic Models
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
    return {
        "ok": True,
        "status": "healthy",
        "version": APP_VERSION,
        "model": OPENAI_MODEL
    }

# -----------------------------
# Firestore helpers (safe merge)
# -----------------------------
def _merge_lists(a: Optional[List[str]], b: Optional[List[str]]) -> List[str]:
    out, seen = [], set()
    for src in (a or []), (b or []):
        for x in src:
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
            # top-level doc
            doc = db.collection("users").document(uid).get()
            if doc.exists:
                d = doc.to_dict() or {}
                likes = _merge_lists(likes, d.get("likes", []))
                dislikes = _merge_lists(dislikes, d.get("dislikes", []))
                owned = _merge_lists(owned, d.get("owned", []))
                wishlist = _merge_lists(wishlist, d.get("wishlist", []))
            # nested preferences/default doc
            pref_doc = db.collection("users").document(uid).collection("preferences").document("default").get()
            if pref_doc.exists:
                d2 = pref_doc.to_dict() or {}
                likes = _merge_lists(likes, d2.get("likes", []))
                dislikes = _merge_lists(dislikes, d2.get("dislikes", []))
                owned = _merge_lists(owned, d2.get("owned", []))
                wishlist = _merge_lists(wishlist, d2.get("wishlist", []))
            # subcollections (id sets)
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
# Keyword detection (wide)
# -----------------------------
KEYWORD_VARIANTS: Dict[str, List[str]] = {
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

# -----------------------------
# INTENT parsing (strict)
# -----------------------------
def parse_intent_block(goal_text: str) -> dict:
    """
    Parse a lightweight INTENT block embedded in the goal.

    [INTENT]
    STRICT FILTER: prioritize brands from pakistan
    STRICT CONTEXT: ensure suitability for eid
    BRAND BIAS: prefer or include byredo
    PERFUMER: prefer creations by francis kurkdjian
    REQUIRED NOTES: include oud, vanilla
    [/INTENT]
    """
    lower = goal_text.lower()
    start = lower.find("[intent]")
    end = lower.find("[/intent]")
    if start == -1 or end == -1 or end <= start:
        return {"countries": [], "occasions": [], "brands": [], "perfumers": [], "notes": []}

    raw = goal_text[start + len("[intent]"): end]
    lines = [l.strip().lower() for l in raw.splitlines() if l.strip()]
    out = {"countries": [], "occasions": [], "brands": [], "perfumers": [], "notes": []}

    def _push(key: str, payload: str):
        # split by commas and 'and'
        tmp = payload.replace(" and ", ",")
        vals = [v.strip() for v in tmp.split(",") if v.strip()]
        for v in vals:
            if v and v not in out[key]:
                out[key].append(v)

    for ln in lines:
        if ln.startswith("strict filter:"):
            _push("countries", ln.replace("strict filter:", "").replace("prioritize brands from", ""))
        elif ln.startswith("strict context:"):
            _push("occasions", ln.replace("strict context:", "").replace("ensure suitability for", ""))
        elif ln.startswith("brand bias:"):
            _push("brands", ln.replace("brand bias:", "").replace("prefer or include", ""))
        elif ln.startswith("perfumer:"):
            _push("perfumers", ln.replace("perfumer:", "").replace("prefer creations by", ""))
        elif ln.startswith("required notes:"):
            _push("notes", ln.replace("required notes:", ""))

    # trim
    for k in out:
        out[k] = [x.strip(" .") for x in out[k] if x.strip(" .")]
    return out

# -----------------------------
# Deterministic fallback
# -----------------------------
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
        # Ensure each root is referenced
        for r in roots:
            if r not in reason.lower():
                reason += f" Clearly {r}-forward elements present."
        picks.append(R(base.name, reason, base.match_score, base.notes or ""))
        i += 1
    return picks[:n]

# -----------------------------
# Tool schema (recommend)
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

SYSTEM_PROMPT_RECOMMEND = """You are ScentFeed's fragrance AI recommender.

You ALWAYS follow these rules:

1) If the user's goal contains an [INTENT] ... [/INTENT] block, you must treat every clause in it as hard constraints in this order of precedence:
   (a) REQUIRED NOTES
   (b) STRICT FILTER / brand-country-region priorities
   (c) STRICT CONTEXT / occasions
   (d) BRAND BIAS
   (e) PERFUMER
   If a constraint cannot be satisfied, select the closest valid alternative and explicitly say why in the reason.

2) Return 3 items (max 5 when asked explicitly), each with:
   - name (string, canonical perfume name)
   - reason (one concise sentence that explicitly references the required note keywords and any matched intent signals such as country/occasion/brand/perfumer)
   - match_score (0–100)
   - notes (short, comma-separated note highlights; optional)

3) Do NOT recommend items already marked as OWNED unless clearly justified.

4) Use the 'propose_recommendations' tool to return results in structured JSON only.
"""

def build_user_prompt(goal: str, prefs: PreferencePayload, groups: List[Tuple[str, List[Pattern]]]) -> str:
    # Parse intent block if present
    intent = parse_intent_block(goal)
    intent_lines = []
    if any(intent.values()):
        def fmt(k, vs):
            return f"- {k}: {', '.join(vs)}" if vs else None
        maybe = [
            fmt("countries", intent["countries"]),
            fmt("occasions", intent["occasions"]),
            fmt("brands", intent["brands"]),
            fmt("perfumers", intent["perfumers"]),
            fmt("notes", intent["notes"]),
        ]
        intent_lines = [x for x in maybe if x]

    if groups:
        lines = []
        for root, pats in groups:
            demo = ", ".join([p.pattern.strip(r"\b").replace("\\s*", " ").strip("^$") for p in pats[:3]])
            lines.append(f"- Must explicitly mention: {root} (e.g., {demo})")
        note_req = "Required notes (auto-detected):\n" + "\n".join(lines)
    else:
        note_req = "Required notes (auto-detected): —"

    intent_block = ""
    if intent_lines:
        intent_block = "[INTENT]\n" + "\n".join(intent_lines) + "\n[/INTENT]\n"

    return f"""
{intent_block}USER GOAL: {goal}

LIKES: {', '.join(prefs.likes) or '—'}
DISLIKES: {', '.join(prefs.dislikes) or '—'}
OWNED: {', '.join(prefs.owned) or '—'}
WISHLIST: {', '.join(prefs.wishlist) or '—'}

{note_req}

RULES:
- Return exactly 3 results.
- Each reason MUST explicitly reference all required note words and any matched intent (country/occasion/brand/perfumer) when applicable.
- If you cannot satisfy a constraint, choose the closest alternative and explain why in the reason.
""".strip()

def _items_meet_requirements(
    items: List[Recommendation],
    groups: List[Tuple[str, List[Pattern]]],
    intent_terms: dict = None
) -> bool:
    if not items:
        return False

    # Enforce note groups
    if groups:
        for it in items:
            blob = " ".join([it.name or "", it.reason or "", it.notes or ""]).lower()
            for _, pats in groups:
                if not any(p.search(blob) for p in pats):
                    return False

    # Enforce INTENT (best-effort presence in reason/notes)
    if intent_terms:
        countries  = intent_terms.get("countries", [])
        occasions  = intent_terms.get("occasions", [])
        brands     = intent_terms.get("brands", [])
        perfumers  = intent_terms.get("perfumers", [])
        req_notes  = intent_terms.get("notes", [])

        for it in items:
            blob = " ".join([it.reason or "", it.notes or ""]).lower()
            if countries  and not any(c in blob for c in countries):  return False
            if occasions  and not any(o in blob for o in occasions):  return False
            if brands     and not any(b in blob for b in brands):     return False
            if perfumers  and not any(p in blob for p in perfumers):  return False
            if req_notes  and not all(n in blob for n in req_notes):  return False

    return True

# -----------------------------
# OpenAI call with retries
# -----------------------------
async def call_openai_with_tools(goal: str, prefs: PreferencePayload, max_results: int) -> RecommendResponse:
    groups = _required_groups_from_goal(goal)
    intent_terms = parse_intent_block(goal)

    def _msgs(extra: str = ""):
        return [
            {"role": "system", "content": SYSTEM_PROMPT_RECOMMEND},
            {"role": "user", "content": build_user_prompt(goal, prefs, groups) + (("\n\n" + extra) if extra else "")}
        ]

    attempts = [
        "",
        "Ensure every fragrance reason explicitly includes all required note words and matched intent terms (country/occasion/brand/perfumer).",
        "STRICT: Only return items where each reason clearly contains each required note word and matched intent terms.",
    ]

    # Guarded retries
    for note in attempts:
        try:
            resp = oai.chat.completions.create(
                model=OPENAI_MODEL,
                temperature=0.0,
                messages=_msgs(note),
                tools=TOOL_SCHEMA_RECOMMEND,
                tool_choice={"type": "function", "function": {"name": "propose_recommendations"}},
                timeout=20.0,
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
            if items and _items_meet_requirements(items, groups, intent_terms):
                return RecommendResponse(items=items, used_profile=prefs)
        except Exception as e:
            logging.warning("OpenAI attempt failed (%s): %s", note or "base", e)
            time.sleep(0.25)

    # Deterministic fallback (note-based)
    if groups:
        fb = _fallback_for(groups, max_results)
        if fb:
            return RecommendResponse(items=fb, used_profile=prefs)

    raise HTTPException(
        status_code=422,
        detail="Constraints were not satisfied (INTENT/notes) and no fallback was available."
    )

# -----------------------------
# ANALYZE tool
# -----------------------------
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
                tool_choice={"type": "function", "function": {"name": "propose_profile"}},
                timeout=20.0,
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

# -----------------------------
# Endpoints
# -----------------------------
@app.post("/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest):
    if not req.goal or not req.goal.strip():
        raise HTTPException(status_code=400, detail="Missing 'goal'.")
    profile = load_user_prefs(req.uid, req.prefs)
    # Guard: enforce 3 results by default, cap at 5 just in case
    max_results = max(1, min(5, req.max_results))
    return await call_openai_with_tools(req.goal.strip(), profile, max_results)

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    profile = load_user_prefs(req.uid, req.prefs)
    max_tags = max(3, min(12, req.max_tags))
    return await call_openai_analyze(profile, max_tags)

# -----------------------------
# Optional: simple SB test for embeddings source (kept if you already rely on this)
# If not using Supabase here, you can remove this endpoint.
# -----------------------------
@app.get("/sb-test")
async def sb_test(q: str = "vanilla", limit: int = 5):
    # This is a placeholder health-like test; adapt to your own vector DB if needed.
    # Keeping the structure so your existing scripts won't break.
    return {"count": limit, "items": [
        {"id": "demo-1", "name": "By the Fireplace", "brand": "Maison Margiela", "score": 0.12},
        {"id": "demo-2", "name": "Black Opium", "brand": "YSL", "score": 0.11},
        {"id": "demo-3", "name": "Le Male", "brand": "Jean Paul Gaultier", "score": 0.10},
        {"id": "demo-4", "name": "La Vie Est Belle", "brand": "Lancôme", "score": 0.09},
        {"id": "demo-5", "name": "My Way", "brand": "Giorgio Armani", "score": 0.08},
    ][:limit]}




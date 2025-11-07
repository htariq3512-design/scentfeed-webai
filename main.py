# main.py
import os, json, time, logging, re
from typing import List, Optional, Tuple, Pattern

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# --- DB pool for Supabase (REQUIRED for /recommend IDs) ---
from psycopg_pool import ConnectionPool

# --- OpenAI (optional; used by /recommend_ai and /analyze) ---
try:
    from openai import OpenAI
    OPENAI_IMPORT_OK = True
except Exception:
    OPENAI_IMPORT_OK = False

APP_VERSION = "2.0.0-backend-ids+ai-optional"

# ------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL")  # <-- set this in Render (Supabase URI)
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL env var is required for /recommend (IDs).")

# Optional OpenAI (do not hard fail if missing)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
AI_READY = bool(OPENAI_IMPORT_OK and OPENAI_API_KEY)
oai = OpenAI(api_key=OPENAI_API_KEY) if AI_READY else None

# Create a pooled connection to Supabase Postgres
pool = ConnectionPool(conninfo=DATABASE_URL, kwargs={"sslmode": "require"})

# ------------------------------------------------------------------------------
# FastAPI app + CORS
# ------------------------------------------------------------------------------
app = FastAPI(title="ScentFeed Web AI", version=APP_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# Shared models
# ------------------------------------------------------------------------------
class PreferencePayload(BaseModel):
    likes: List[str] = Field(default_factory=list)
    dislikes: List[str] = Field(default_factory=list)
    owned: List[str] = Field(default_factory=list)
    wishlist: List[str] = Field(default_factory=list)

# ----- /recommend (IDs) request/response (for iOS app contract) -----
class RecommendIdsRequest(BaseModel):
    # matches the iOS RecommendClient payload
    query: str = ""
    topN: int = 20
    userId: Optional[str] = None
    filters: Optional[dict] = None
    prefs: Optional[dict] = None
    persona: Optional[dict] = None

class RankedId(BaseModel):
    id: str
    score: Optional[float] = None

class RecommendIdsResponse(BaseModel):
    items: List[RankedId]

# ----- OpenAI-driven models (kept for your AI flows) -----
class RecommendAIRequest(BaseModel):
    uid: Optional[str] = None
    goal: str = Field(default="Suggest 3 fragrances.")
    max_results: int = Field(default=3, ge=1, le=5)
    prefs: Optional[PreferencePayload] = None

class Recommendation(BaseModel):
    name: str
    reason: str
    match_score: int = Field(ge=0, le=100)
    notes: Optional[str] = None

class RecommendAIResponse(BaseModel):
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

# ------------------------------------------------------------------------------
# Health
# ------------------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "ok": True,
        "status": "healthy",
        "version": APP_VERSION,
        "ai_ready": AI_READY,
        "model": OPENAI_MODEL if AI_READY else None,
    }

# ------------------------------------------------------------------------------
# /recommend  → DB-backed: returns IDs + scores from Supabase RPC
# ------------------------------------------------------------------------------
@app.post("/recommend", response_model=RecommendIdsResponse)
def recommend_ids(req: RecommendIdsRequest):
    q = req.query or ""
    limit = max(1, min(100, int(req.topN or 20)))

    # Call your RPC: public.search_perfumes(q text, ..., limit_n int)
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                select id, score
                from public.search_perfumes(%s::text, null, null, null, null, null, %s::int)
                """,
                (q, limit),
            )
            rows = cur.fetchall()

    items = [{"id": r[0], "score": float(r[1]) if r[1] is not None else None} for r in rows]
    return {"items": items}

# ------------------------------------------------------------------------------
# Everything below this point is your existing OpenAI-driven logic,
# kept intact but moved under /recommend_ai and /analyze
# (OpenAI is optional; we return 503 if not configured).
# ------------------------------------------------------------------------------

# ---- Keyword detection (wide) for AI prompt control ----
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

def _items_meet_requirements(items: List["Recommendation"], groups: List[Tuple[str, List[Pattern]]]) -> bool:
    if not groups:
        return True
    for it in items:
        blob = " ".join([getattr(it, "name", "") or "", getattr(it, "reason", "") or "", getattr(it, "notes", "") or ""])
        if not all(_item_matches_group(blob, pats) for _, pats in groups):
            return False
    return True

def R(name: str, reason: str, score: int, notes: str) -> "Recommendation":
    return Recommendation(name=name, reason=reason, match_score=score, notes=notes)

FALLBACK_CATALOG = {
    # ... (keep your fallback catalog exactly as you had it)
    "vanilla": [
        R("Guerlain Spiritueuse Double Vanille","Rich vanilla with boozy warmth—clearly vanilla-forward.",95,"vanilla, rum, benzoin"),
        R("Tom Ford Tobacco Vanille","Dense vanilla wrapped in tobacco and spice—deep vanilla profile.",92,"vanilla, tobacco, tonka"),
        R("Kayali Vanilla | 28","Sweet, wearable vanilla with amber warmth—vanilla signature.",90,"vanilla, amber, brown sugar"),
    ],
    # (snipped for brevity; include the rest from your file unchanged)
}

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

async def call_openai_with_tools(goal: str, prefs: PreferencePayload, max_results: int) -> RecommendAIResponse:
    if not AI_READY:
        raise HTTPException(status_code=503, detail="OpenAI not configured.")
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
                return RecommendAIResponse(items=items, used_profile=prefs)
        except Exception:
            time.sleep(0.25)

    # simple deterministic fallback if keywords present
    if groups:
        roots = [r for r, _ in groups]
        cats = [FALLBACK_CATALOG.get(r, []) for r in roots]
        if all(cats):
            picks: List[Recommendation] = []
            i = 0
            while len(picks) < max_results:
                cat = cats[i % len(cats)]
                base = cat[(len(picks) // len(cats)) % len(cat)]
                reason = base.reason
                for r in roots:
                    if r not in reason.lower():
                        reason += f" Clearly {r}-forward elements present."
                picks.append(Recommendation(name=base.name, reason=reason, match_score=base.match_score, notes=base.notes))
                i += 1
            return RecommendAIResponse(items=picks[:max_results], used_profile=prefs)

    raise HTTPException(status_code=422, detail="AI recommend failed and no fallback available.")

ANALYZE_SYSTEM_PROMPT = """You are ScentFeed's taste analyst.
- Read likes, dislikes, owned, wishlist.
- Infer a concise taste profile (notes and styles).
- Avoid repeating the raw lists; interpret them.
- Be specific and helpful.
- Return results via the propose_profile tool.
"""

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

async def call_openai_analyze(prefs: PreferencePayload, max_tags: int) -> AnalyzeResponse:
    if not AI_READY:
        raise HTTPException(status_code=503, detail="OpenAI not configured.")
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
    raise HTTPException(status_code=503, detail=f"AI analyze failed: {last_err}")

# ----------------------- Public AI endpoints -----------------------
@app.post("/recommend_ai", response_model=RecommendAIResponse)
async def recommend_ai(req: RecommendAIRequest):
    if not AI_READY:
        raise HTTPException(status_code=503, detail="OpenAI not configured.")
    prefs = req.prefs or PreferencePayload()
    return await call_openai_with_tools(req.goal, prefs, req.max_results)

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    if not AI_READY:
        raise HTTPException(status_code=503, detail="OpenAI not configured.")
    prefs = req.prefs or PreferencePayload()
    return await call_openai_analyze(prefs, req.max_tags)




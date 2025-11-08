# main.py
import os, json, time, logging, re
from typing import List, Optional, Tuple, Pattern, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from openai import OpenAI

# Load .env first so all env vars are available
load_dotenv(override=True)

APP_VERSION = "2.0.0-constraints+retry+rerank"

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
# Optional Postgres (only if DATABASE_URL provided)
# -------------------------------------------------------------------
POOL = None
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
if DATABASE_URL:
    try:
        from psycopg_pool import ConnectionPool  # optional dependency
        POOL = ConnectionPool(conninfo=DATABASE_URL, max_size=5, kwargs={"connect_timeout": 5})
        logging.info("Postgres pool enabled.")
    except Exception as e:
        logging.warning("Postgres pool disabled: %s", e)
else:
    logging.info("DATABASE_URL not set; Postgres disabled.")

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
# Models (Pydantic)
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

# Parsed constraint hints
class ConstraintHints(BaseModel):
    country: Optional[str] = None        # "PK", "FR", ...
    brand: Optional[str] = None
    house: Optional[str] = None
    perfumer: Optional[str] = None
    max_price: Optional[float] = None
    gender: Optional[str] = None         # men | women | unisex
    occasion: Optional[str] = None       # date | office | gym | wedding | ...
    note_words: List[str] = Field(default_factory=list)

# -------------------------------------------------------------------
# Health
# -------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"ok": True, "status": "healthy", "version": APP_VERSION, "model": OPENAI_MODEL}

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
# Keyword detection (wide coverage for natural language notes)
# -------------------------------------------------------------------
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
# Deterministic fallback catalog (truncated sample)
# -------------------------------------------------------------------
def R(name: str, reason: str, score: int, notes: str) -> Recommendation:
    return Recommendation(name=name, reason=reason, match_score=score, notes=notes)

FALLBACK_CATALOG = {
    "vanilla": [
        R("Guerlain Spiritueuse Double Vanille","Rich vanilla with boozy warmth—vanilla-forward.",95,"vanilla, rum, benzoin"),
        R("Tom Ford Tobacco Vanille","Dense vanilla wrapped in tobacco and spice—deep vanilla profile.",92,"vanilla, tobacco, tonka"),
        R("Kayali Vanilla | 28","Sweet, wearable vanilla with amber warmth—vanilla signature.",90,"vanilla, amber, brown sugar"),
    ],
    "oud": [
        R("Initio Oud for Greatness","Bold modern oud with saffron accents.",92,"oud, saffron, patchouli"),
        R("Acqua di Parma Oud","Refined oud lifted by citrus.",90,"oud, bergamot, leather"),
        R("Montale Black Aoud","Dark rose-oud signature.",88,"oud, rose, patchouli"),
    ],
    "rose": [
        R("Frederic Malle Portrait of a Lady","Luxurious rose with patchouli.",92,"rose, patchouli, incense"),
        R("Diptyque Eau Rose","Airy, naturalistic rose.",89,"rose, lychee, musk"),
        R("Le Labo Rose 31","Spice-wood rose; modern unisex.",88,"rose, cumin, cedar"),
    ],
    # ...add more groups as you like
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
# Goal → Constraint parsing (Step 1 logic, built-in)
# -------------------------------------------------------------------
_COUNTRY_ALIASES = {
    # add aliases → ISO-ish tokens the LLM can echo back
    "pakistan": "pk", "india": "in", "france": "fr", "united states": "us", "usa": "us",
    "saudi": "sa", "saudi arabia":"sa", "uae":"ae", "united arab emirates":"ae",
    "uk": "gb", "united kingdom": "gb", "england": "gb",
    "italy": "it", "spain": "es", "turkey":"tr", "turkiye":"tr",
}

_GENDER = {"men","male","masculine","women","female","feminine","unisex"}
_OCCASIONS = {"date","office","work","gym","wedding","party","club","daily","signature","evening","summer","winter","fall","spring"}

_PRICE_PAT = re.compile(r"(?:under|max|<=|less than)\s*\$?\s*(\d+(\.\d+)?)", re.IGNORECASE)

def _extract_country(text: str) -> Optional[str]:
    t = text.lower()
    for key, iso in _COUNTRY_ALIASES.items():
        if key in t:
            return iso.upper()
    # fallback for patterns like "made in pk"
    m = re.search(r"\b(in|from)\s+([A-Za-z][A-Za-z ]+)\b", t)
    if m:
        raw = m.group(2).strip()
        return _COUNTRY_ALIASES.get(raw, raw[:2]).upper()
    return None

def _extract_brand_house_perfumer(text: str) -> Dict[str, Optional[str]]:
    t = text.strip()
    tl = t.lower()

    brand = None
    house = None
    perfumer = None

    # Hints like: "brand: ajmal", "house: armaf", "perfumer: alberto morillas"
    for key in ["brand", "house", "perfumer"]:
        m = re.search(rf"{key}\s*:\s*([A-Za-z0-9 '\-&]+)", tl)
        if m:
            val = m.group(1).strip()
            if key == "brand": brand = val
            if key == "house": house = val
            if key == "perfumer": perfumer = val

    # Natural phrasing: "by Alberto Morillas", "from Ajmal", "house Ajmal"
    if perfumer is None:
        m = re.search(r"\bby\s+([A-Za-z][A-Za-z '\-]+)\b", t, re.IGNORECASE)
        if m:
            perfumer = m.group(1).strip().lower()

    if house is None:
        m = re.search(r"\bhouse\s+([A-Za-z0-9 '\-&]+)\b", t, re.IGNORECASE)
        if m:
            house = m.group(1).strip().lower()

    if (brand is None) or (brand == house):
        m = re.search(r"\bfrom\s+([A-Za-z0-9 '\-&]+)\b", t, re.IGNORECASE)
        if m:
            brand = (brand or m.group(1).strip().lower())

    return {
        "brand": brand,
        "house": house,
        "perfumer": perfumer,
    }

def _extract_price_gender_occasion(text: str) -> Dict[str, Optional[Any]]:
    t = text.lower()
    price = None
    m = _PRICE_PAT.search(t)
    if m:
        try:
            price = float(m.group(1))
        except Exception:
            price = None

    gender = None
    for g in _GENDER:
        if re.search(rf"\b{re.escape(g)}\b", t):
            gender = g
            break

    occasion = None
    for oc in _OCCASIONS:
        if re.search(rf"\b{re.escape(oc)}\b", t):
            occasion = oc
            break

    return {"max_price": price, "gender": gender, "occasion": occasion}

def _extract_note_words(text: str) -> List[str]:
    words = []
    tl = text.lower()
    for root, pats in KEYWORD_VARIANTS.items():
        if any(re.search(p, tl) for p in pats):
            words.append(root)
    return words

async def analyze_goal_constraints(goal: str) -> Dict[str, Any]:
    """Synchronous heuristics wrapped as async for consistency."""
    if not goal:
        return {}
    country = _extract_country(goal)
    brand_house_perfumer = _extract_brand_house_perfumer(goal)
    pg = _extract_price_gender_occasion(goal)
    notes = _extract_note_words(goal)

    out: Dict[str, Any] = {
        "country": country,
        "brand": brand_house_perfumer["brand"],
        "house": brand_house_perfumer["house"],
        "perfumer": brand_house_perfumer["perfumer"],
        "max_price": pg["max_price"],
        "gender": pg["gender"],
        "occasion": pg["occasion"],
        "note_words": notes,
    }
    return out

# -------------------------------------------------------------------
# Constraint helpers (enforcement & ranking)
# -------------------------------------------------------------------
def _constraints_summary(c: ConstraintHints) -> str:
    parts = []
    if c.country:   parts.append(f"Country: {c.country}")
    if c.brand:     parts.append(f"Brand: {c.brand}")
    if c.house:     parts.append(f"House: {c.house}")
    if c.perfumer:  parts.append(f"Perfumer: {c.perfumer}")
    if c.max_price is not None: parts.append(f"Max Price: {c.max_price}")
    if c.gender:    parts.append(f"Gender: {c.gender}")
    if c.occasion:  parts.append(f"Occasion: {c.occasion}")
    if c.note_words:parts.append(f"Required Notes: {', '.join(c.note_words)}")
    return " | ".join(parts) if parts else "—"

def _violates_hard_constraints(item: Recommendation, c: ConstraintHints) -> bool:
    """Hard constraints = country/brand/house/perfumer (if provided).
       We require explicit mention in reason/notes to avoid hallucinated matches."""
    blob = " ".join([item.name or "", item.reason or "", item.notes or ""]).lower()

    def contains(x: Optional[str]) -> bool:
        return (x or "").strip().lower() in blob if x else True

    if c.country and not contains(c.country):
        return True
    if c.brand and not contains(c.brand):
        return True
    if c.house and not contains(c.house):
        return True
    if c.perfumer and not contains(c.perfumer):
        return True
    return False

def _rerank_by_constraints(items: List[Recommendation], c: ConstraintHints) -> List[Recommendation]:
    def score(it: Recommendation) -> int:
        s = 0
        text = " ".join([it.name or "", it.reason or "", it.notes or ""]).lower()

        def has(word: Optional[str]) -> bool:
            return bool(word and word.strip().lower() in text)

        # strong signals
        if has(c.country): s += 30
        if has(c.brand): s += 30
        if has(c.house): s += 30
        if has(c.perfumer): s += 30
        # note words
        for w in c.note_words:
            if w.lower() in text: s += 5
        # light nudges
        if has(c.occasion): s += 3
        if has(c.gender): s += 2
        return s

    return sorted(items, key=score, reverse=True)

# -------------------------------------------------------------------
# LLM tool schemas
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

Rules you MUST follow strictly:
- If the goal or constraints include specific country/brand/house/perfumer, ONLY return fragrances that match those keys (mention them explicitly in each reason).
- If required note words are present, each reason must explicitly include those words (or common variants).
- Do NOT recommend already-owned scents unless strongly justified.
- Return 3 items (max 5), each with: name, reason (1–2 sentences), match_score (0–100), and short notes if useful.
- Use the 'propose_recommendations' tool to return strictly structured JSON.
"""

def build_user_prompt(goal: str, prefs: PreferencePayload, groups: List[Tuple[str, List[Pattern]]], constraints: ConstraintHints) -> str:
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

Parsed Constraints: {_constraints_summary(constraints)}

Likes: {', '.join(prefs.likes) or '—'}
Dislikes: {', '.join(prefs.dislikes) or '—'}
Owned: {', '.join(prefs.owned) or '—'}
Wishlist: {', '.join(prefs.wishlist) or '—'}

{req}

STRICTNESS:
- Enforce all provided hard keys first: Country/Brand/House/Perfumer.
- Enforce required note words in each reason.
- If not enough candidates exist, prefer closest matches and say why, but DO NOT ignore hard keys silently.
""".strip()

# -------------------------------------------------------------------
# OpenAI call with enforcement + retry + rerank + fallback
# -------------------------------------------------------------------
async def call_openai_with_tools(goal: str, prefs: PreferencePayload, max_results: int, constraints: ConstraintHints) -> RecommendResponse:
    groups = _required_groups_from_goal(goal)

    def _msgs(extra: str = ""):
        return [
            {"role": "system", "content": SYSTEM_PROMPT_RECOMMEND},
            {"role": "user", "content": build_user_prompt(goal, prefs, groups, constraints) + (("\n\n" + extra) if extra else "")}
        ]

    attempts = [
        "",
        "STRICT: Re-check and ONLY return items that satisfy Country/Brand/House/Perfumer if provided, and include each required note word explicitly.",
    ]

    last_items: List[Recommendation] = []
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
            if not items:
                continue

            # Hard filter (country/brand/house/perfumer must be mentioned if provided)
            filtered = [it for it in items if not _violates_hard_constraints(it, constraints)]

            # If we requested any hard keys, prefer strictly valid; otherwise accept if notes satisfied.
            any_hard = any([constraints.country, constraints.brand, constraints.house, constraints.perfumer])

            if any_hard:
                # If nothing strictly valid, keep items but try another attempt
                if filtered:
                    ranked = _rerank_by_constraints(filtered, constraints)
                    return RecommendResponse(items=ranked[:max_results], used_profile=prefs)
            else:
                # No hard keys: validate required notes if present
                if _items_meet_requirements(items, groups):
                    ranked = _rerank_by_constraints(items, constraints)
                    return RecommendResponse(items=ranked[:max_results], used_profile=prefs)

            last_items = items
        except Exception:
            time.sleep(0.25)

    # Final attempt: return best we have, reranked
    if last_items:
        ranked = _rerank_by_constraints(last_items, constraints)
        return RecommendResponse(items=ranked[:max_results], used_profile=prefs)

    # Deterministic fallback if available
    fb = _fallback_for(groups, max_results)
    if fb:
        ranked = _rerank_by_constraints(fb, constraints)
        return RecommendResponse(items=ranked[:max_results], used_profile=prefs)

    raise HTTPException(status_code=422, detail="No valid recommendations after enforcing constraints.")

# -------------------------------------------------------------------
# Analyze (taste surfacing)
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
    try:
        parsed = await analyze_goal_constraints(req.goal)   # Step 1 parser (built-in)
        constraints = ConstraintHints(**parsed)
    except Exception:
        constraints = ConstraintHints()
    return await call_openai_with_tools(req.goal, profile, req.max_results, constraints)

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    profile = load_user_prefs(req.uid, req.prefs)
    return await call_openai_analyze(profile, req.max_tags)




# main.py
#
# ScentFeed Web AI backend (TikTok/Meta-style ranker)
#
# - Loads perfumes from Supabase `perfumes` table
# - Filters by price_band (0-100 / 100-200 / 200+)
# - Uses OpenAI to RANK a candidate list by index (no name-mismatch issues)
# - Returns top N items with reasons + match_score
# - If anything fails or catalog is empty: returns items = [] so the iOS app
#   can fall back to its own offline CSV engine.
#

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Tuple
import os
import time
import json
import threading

import requests
from openai import OpenAI

# ---------- OpenAI client ----------

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ---------- FastAPI app ----------

BACKEND_VERSION = "4.1.1-ranker-no-fallback"
app = FastAPI(title="ScentFeed Web AI", version=BACKEND_VERSION)

# ---------- Pydantic models (wire contract with the iOS app) ----------


class Prefs(BaseModel):
    """
    User preference payload from the app.

    IMPORTANT: We allow these to be Optional so that if the client sends
    null or omits them, the request still validates. We normalize to []
    when we actually use them.
    """
    likes: Optional[List[str]] = None
    dislikes: Optional[List[str]] = None
    owned: Optional[List[str]] = None
    wishlist: Optional[List[str]] = None


class RecommendFilters(BaseModel):
    """
    Filters coming from the app.
    We mainly care about price_band, but we forward gender/region/raw as hints.
    """
    gender: Optional[str] = None          # "male" | "female" | "unisex"
    region: Optional[str] = None          # "pakistan", "gulf", "europe", ...
    price_band: Optional[str] = None      # "0-100", "100-200", "200+"
    raw: Optional[Dict[str, Any]] = None  # any extra future stuff


class RecommendRequest(BaseModel):
    uid: Optional[str] = None
    goal: str
    max_results: int = 3
    prefs: Prefs
    filters: Optional[RecommendFilters] = None


class RecommendItem(BaseModel):
    name: str
    reason: str
    match_score: int = Field(ge=0, le=100)
    notes: Optional[str] = None


class UsedProfile(BaseModel):
    likes: List[str] = Field(default_factory=list)
    dislikes: List[str] = Field(default_factory=list)
    owned: List[str] = Field(default_factory=list)
    wishlist: List[str] = Field(default_factory=list)


class RecommendResponse(BaseModel):
    items: List[RecommendItem]
    used_profile: UsedProfile
    request_id: Optional[str] = None


class HealthResponse(BaseModel):
    ok: bool
    status: str
    version: str
    model: str


# ---------- Supabase catalog loading (cached) ----------

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = (
    os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    or os.environ.get("SUPABASE_ANON_KEY")
)

_catalog_lock = threading.Lock()
_catalog_cache: Optional[Tuple[float, List[Dict[str, Any]]]] = None
_CATALOG_TTL_SECONDS = 300  # 5 minutes


def load_catalog_from_supabase() -> List[Dict[str, Any]]:
    """
    Fetch all rows from public.perfumes via Supabase REST.
    We treat the rows as generic dicts with at least:
      - name (text)
      - brand (text)
      - description (text, nullable)
      - price_band (text, nullable)   e.g. "0-100", "100-200", "200+"
      - price (numeric/text, nullable)
      - notes_top / notes_heart / notes_base (text[] or string, nullable)
      - projection / longevity / celebrity_endorsers (optional)
    """
    global _catalog_cache

    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        print("⚠️ Supabase not configured (SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY missing).")
        return []

    now = time.time()
    with _catalog_lock:
        if _catalog_cache is not None:
            ts, items = _catalog_cache
            if now - ts < _CATALOG_TTL_SECONDS:
                return items

    url = SUPABASE_URL.rstrip("/") + "/rest/v1/perfumes"
    headers = {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Accept": "application/json",
    }
    params = {"select": "*"}
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        raw = resp.json()
        if not isinstance(raw, list):
            print("⚠️ Unexpected Supabase response shape for perfumes:", type(raw))
            return []

        items: List[Dict[str, Any]] = []
        for row in raw:
            try:
                name = (row.get("name") or "").strip()
                if not name:
                    continue
                items.append(row)
            except Exception as e:
                print("⚠️ Failed to parse perfume row from Supabase:", e, "row=", row)

        with _catalog_lock:
            _catalog_cache = (now, items)
        print(f"✅ Loaded {len(items)} perfumes from Supabase catalog.")
        return items
    except Exception as e:
        print("❌ Error loading perfumes from Supabase:", repr(e))
        return []


# ---------- Utility helpers ----------

def trimmed_or_none(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    v = value.strip()
    return v or None


def resolve_price_band(filters: Optional[RecommendFilters]) -> Optional[str]:
    """
    Normalize price bands between frontend naming and DB values.
    We expect DB price_band values like: "0-100", "100-200", "200+".
    """
    if not filters or not filters.price_band:
        return None

    pb = filters.price_band.strip().lower()

    if pb in ("0-100", "0 – 100", "0 — 100", "0_to_100", "0_100", "0 –100"):
        return "0-100"
    if pb in ("100-200", "100 – 200", "100 — 200", "100_to_200", "100_200"):
        return "100-200"
    if pb in ("200+", "200_plus", "200 – up", "200+ "):
        return "200+"

    # fallback: if user gives "under100" etc, map roughly
    if "0" in pb and "100" in pb:
        return "0-100"
    if "100" in pb and "200" in pb:
        return "100-200"
    if "200" in pb or "+" in pb or "plus" in pb:
        return "200+"

    return None


def build_system_prompt() -> str:
    """
    System prompt for RANKING:
    - Model sees a candidate list with INDEXES.
    - It must output JSON with candidate_index, reason, match_score, notes.
    """
    return (
        "You are ScentFeed, a professional perfume recommendation engine.\n"
        "\n"
        "You are given:\n"
        " - A natural language GOAL.\n"
        " - A PROFILE (likes / dislikes / owned / wishlist).\n"
        " - Optional FILTERS (region, gender, price_band).\n"
        " - A CANDIDATE LIST of perfumes from the catalog, each with an INDEX.\n"
        "\n"
        "Your job:\n"
        " - RANK the candidates for how well they fit the GOAL and PROFILE.\n"
        " - You MUST ONLY choose perfumes from the candidate list.\n"
        " - Refer to perfumes by their integer candidate_index.\n"
        " - Respect price_band as much as possible (candidates are already filtered).\n"
        " - If PROFILE shows strong likes (e.g. vanilla, oud, floral), bias toward matching notes.\n"
        " - If PROFILE shows dislikes, avoid them.\n"
        " - Avoid recommending things the user already OWNS unless there are not enough new options.\n"
        " - Vary your top picks when the GOAL clearly changes (different vibe / setting / budget).\n"
        "\n"
        "Occasion hints:\n"
        " - \"office\" → clean, non-offensive, moderate projection.\n"
        " - \"gym\" → very fresh, light, not cloying.\n"
        " - \"clubbing\" → louder, sexy, attention-grabbing.\n"
        " - \"date night\" → intimate, attractive, usually sweeter.\n"
        " - \"special event\" / \"wedding\" → elegant, long-lasting, signature-worthy.\n"
        "\n"
        "VERY IMPORTANT RESPONSE FORMAT:\n"
        " - You MUST answer with ONLY valid, minified JSON.\n"
        " - No markdown, no explanations outside JSON.\n"
        " - Schema:\n"
        "   {\n"
        "     \"items\": [\n"
        "       {\n"
        "         \"candidate_index\": 0,\n"
        "         \"reason\": \"short explanation (1-2 sentences)\",\n"
        "         \"match_score\": 0-100,\n"
        "         \"notes\": \"optional, short comma-separated key notes\" or null\n"
        "       },\n"
        "       ...\n"
        "     ],\n"
        "     \"used_profile\": {\n"
        "       \"likes\": [...],\n"
        "       \"dislikes\": [...],\n"
        "       \"owned\": [...],\n"
        "       \"wishlist\": [...]\n"
        "     },\n"
        "     \"request_id\": \"string-id-for-debugging\"\n"
        "   }\n"
        "\n"
        "The candidate_index MUST be a valid integer index from the CANDIDATE LIST.\n"
        "Never include trailing commas. Never add comments. Only valid JSON.\n"
    )


def summarize_candidate(idx: int, row: Dict[str, Any]) -> str:
    """
    Turn a Supabase row into a compact summary line for the LLM.
    """
    name = (row.get("name") or "").strip()
    brand = (row.get("brand") or "Unknown").strip()
    description = (row.get("description") or "").strip()
    price_band = (row.get("price_band") or "").strip() or "unknown"
    price = row.get("price")
    projection = (row.get("projection") or "").strip() or "unknown"
    longevity = (row.get("longevity") or "").strip() or "unknown"

    def normalize_notes(field: str) -> List[str]:
        v = row.get(field)
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        parts = str(v).split(",")
        return [p.strip() for p in parts if p.strip()]

    notes_top = normalize_notes("notes_top")
    notes_heart = normalize_notes("notes_heart")
    notes_base = normalize_notes("notes_base")
    all_notes: List[str] = []
    for n in notes_top + notes_heart + notes_base:
        if n not in all_notes:
            all_notes.append(n)
    notes_str = ", ".join(all_notes) if all_notes else ""

    price_part = f"price_band: {price_band}"
    if price is not None:
        price_part += f", approx_price: {price}"

    parts = [
        f"[{idx}] {name} by {brand}",
        price_part,
        f"projection: {projection}",
        f"longevity: {longevity}",
    ]
    if notes_str:
        parts.append(f"notes: {notes_str}")
    if description:
        parts.append(f"desc: {description}")

    return " | ".join(parts)


def build_user_prompt(req: RecommendRequest, candidates: List[Dict[str, Any]]) -> str:
    """
    User prompt that includes GOAL, PROFILE, FILTERS and the CANDIDATE LIST.
    """

    prefs = req.prefs
    likes = prefs.likes or []
    dislikes = prefs.dislikes or []
    owned = prefs.owned or []
    wishlist = prefs.wishlist or []

    parts: List[str] = []

    parts.append(f"GOAL: {req.goal.strip()}")

    parts.append(
        "PROFILE:\n"
        f"  likes: {likes}\n"
        f"  dislikes: {dislikes}\n"
        f"  owned: {owned}\n"
        f"  wishlist: {wishlist}"
    )

    if req.filters:
        f = req.filters
        parts.append(
            "FILTERS:\n"
            f"  gender: {f.gender}\n"
            f"  region: {f.region}\n"
            f"  price_band: {f.price_band}\n"
            f"  raw: {f.raw}"
        )
    else:
        parts.append("FILTERS:\n  gender: null\n  region: null\n  price_band: null\n  raw: null")

    parts.append(f"MAX_RESULTS: {req.max_results}")

    lines = [summarize_candidate(i, row) for i, row in enumerate(candidates)]
    catalog_block = "\n".join(lines)
    parts.append(
        "CANDIDATE LIST (you MUST only choose perfumes from this list using candidate_index):\n"
        + catalog_block
    )

    return "\n\n".join(parts)


def filter_candidates_for_request(
    catalog: List[Dict[str, Any]],
    req: RecommendRequest,
    max_candidates: int = 60,
) -> List[Dict[str, Any]]:
    """
    Filter by price_band and optionally sample / truncate to a manageable size.
    """
    if not catalog:
        return []

    resolved_band = resolve_price_band(req.filters)
    filtered = catalog

    if resolved_band:
        def match_band(row: Dict[str, Any]) -> bool:
            rb = (row.get("price_band") or "").strip()
            return rb == resolved_band

        band_matches = [r for r in catalog if match_band(r)]
        if band_matches:
            filtered = band_matches
        else:
            print(
                f"⚠️ No perfumes matched price_band='{resolved_band}' in Supabase; "
                f"falling back to full catalog of {len(catalog)} perfumes."
            )

    if len(filtered) > max_candidates:
        filtered = filtered[:max_candidates]

    print(
        f"🧪 Candidate selection: {len(filtered)} perfumes after filters "
        f"(price_band={resolved_band or 'any'}, total_catalog={len(catalog)})"
    )

    return filtered


# ---------- Routes ----------


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    catalog = load_catalog_from_supabase()
    status = "healthy-with-catalog" if catalog else "healthy-no-catalog"
    return HealthResponse(
        ok=True,
        status=status,
        version=BACKEND_VERSION,
        model="gpt-4o-mini",
    )


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest) -> RecommendResponse:
    """
    Main recommendation endpoint called by the iOS app.
    - Loads Supabase catalog
    - Filters candidates by price_band
    - Asks OpenAI to RANK by candidate_index
    - Maps back to real perfumes and returns top N
    - If anything fails: items = [], so iOS can fall back to offline CSV.
    """
    goal = trimmed_or_none(req.goal)
    if not goal:
        raise HTTPException(status_code=400, detail="Goal must not be empty.")

    likes = req.prefs.likes or []
    dislikes = req.prefs.dislikes or []
    owned = req.prefs.owned or []
    wishlist = req.prefs.wishlist or []

    # 1) Load catalog
    catalog_all = load_catalog_from_supabase()

    if not catalog_all:
        print("⚠️ Catalog empty or Supabase not configured — returning items = [] so app can use offline CSV.")
        return RecommendResponse(
            items=[],
            used_profile=UsedProfile(
                likes=likes,
                dislikes=dislikes,
                owned=owned,
                wishlist=wishlist,
            ),
            request_id="no-catalog",
        )

    # 2) Build candidate list for this request
    candidates = filter_candidates_for_request(catalog_all, req)

    if not candidates:
        print("⚠️ No candidates after filtering, returning items = [] so app can use offline CSV.")
        return RecommendResponse(
            items=[],
            used_profile=UsedProfile(
                likes=likes,
                dislikes=dislikes,
                owned=owned,
                wishlist=wishlist,
            ),
            request_id="no-candidates-after-filter",
        )

    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(req, candidates)

    try:
        started = time.time()

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.5,
            max_tokens=900,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        latency_ms = int((time.time() - started) * 1000)
        choice = completion.choices[0]
        raw_json = choice.message.content

        print(
            f"🌐 /recommend status: 200, latency={latency_ms}ms, "
            f"candidates={len(candidates)}"
        )

        try:
            data = json.loads(raw_json)
        except Exception:
            print(f"❗️/recommend decode failed. Raw body: {raw_json!r}")
            return RecommendResponse(
                items=[],
                used_profile=UsedProfile(
                    likes=likes,
                    dislikes=dislikes,
                    owned=owned,
                    wishlist=wishlist,
                ),
                request_id=f"decode-error-{completion.id}",
            )

        used_profile_data = data.get("used_profile") or {}
        used_profile = UsedProfile(
            likes=used_profile_data.get("likes") or likes,
            dislikes=used_profile_data.get("dislikes") or dislikes,
            owned=used_profile_data.get("owned") or owned,
            wishlist=used_profile_data.get("wishlist") or wishlist,
        )

        items_data = data.get("items") or []
        cleaned_items: List[RecommendItem] = []

        for raw_item in items_data:
            idx = raw_item.get("candidate_index")
            try:
                idx_int = int(idx)
            except Exception:
                continue
            if idx_int < 0 or idx_int >= len(candidates):
                continue

            row = candidates[idx_int]
            name = (row.get("name") or "").strip()
            if not name:
                continue

            reason = raw_item.get("reason") or "No reason provided."
            match_score = raw_item.get("match_score") or 75
            notes = raw_item.get("notes")

            try:
                score_int = int(match_score)
            except Exception:
                score_int = 75

            cleaned_items.append(
                RecommendItem(
                    name=name,
                    reason=reason,
                    match_score=max(0, min(100, score_int)),
                    notes=notes,
                )
            )

        max_results = max(1, req.max_results)
        cleaned_items = cleaned_items[:max_results]

        # If model gave nothing usable, let iOS use offline CSV.
        if not cleaned_items:
            print("⚠️ Model returned empty/invalid ranked items — returning items = [] so app can use offline CSV.")
            return RecommendResponse(
                items=[],
                used_profile=used_profile,
                request_id=data.get("request_id") or completion.id,
            )

        request_id = data.get("request_id") or completion.id

        return RecommendResponse(
            items=cleaned_items,
            used_profile=used_profile,
            request_id=request_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        print("❌ /recommend unexpected error:", repr(e))
        return RecommendResponse(
            items=[],
            used_profile=UsedProfile(
                likes=likes,
                dislikes=dislikes,
                owned=owned,
                wishlist=wishlist,
            ),
            request_id="internal-error",
        )



# main.py
#
# ScentFeed Web AI backend (TikTok/Meta-style ranker)
#
# - Loads perfumes from Supabase `perfumes` table
# - Strictly filters by scent + price band when possible
# - Can flex price by ~20% if needed to reach 3 candidates
# - Uses OpenAI to RANK a candidate list by index
# - ALWAYS returns catalog-based perfumes (no hardcoded Dior/Chanel fallback)

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

BACKEND_VERSION = "4.2.0-ranked-scoring"

app = FastAPI(title="ScentFeed Web AI", version=BACKEND_VERSION)

# ---------- Pydantic models (wire contract with the iOS app) ----------


class Prefs(BaseModel):
    likes: List[str] = Field(default_factory=list)
    dislikes: List[str] = Field(default_factory=list)
    owned: List[str] = Field(default_factory=list)
    wishlist: List[str] = Field(default_factory=list)


class RecommendRequest(BaseModel):
    """
    Flexible filters: we accept any shape here as a dict to avoid 422s
    when the iOS side sends extra keys.
    """
    uid: Optional[str] = None
    goal: str
    max_results: int = 3
    prefs: Prefs
    filters: Optional[Dict[str, Any]] = None

    class Config:
        extra = "allow"


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
    Expected fields (loose):
      - name (text)
      - brand (text)
      - description (text, nullable)
      - price_band (text, nullable)   e.g. "0-100", "100-200", "200+"
      - price (numeric/text, nullable)
      - notes_top / notes_heart / notes_base (text[] or string, nullable)
      - projection / longevity / celebrity_endorsers (optional)
      - scent_tags / setting_tags / gender (string or string[])
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


def parse_structured_goal(goal: str) -> Dict[str, Optional[str]]:
    """
    Parse goal strings like:
      "scent: fresh | setting: every day | budget: 0-100"
    into:
      {"scent": "fresh", "setting": "every day", "price_band": "0-100"}
    """
    result: Dict[str, Optional[str]] = {
        "scent": None,
        "setting": None,
        "price_band": None,
    }
    parts = [p.strip() for p in goal.split("|")]
    for part in parts:
        lower = part.lower()
        if lower.startswith("scent:"):
            result["scent"] = part.split(":", 1)[1].strip().lower()
        elif lower.startswith("setting:"):
            result["setting"] = part.split(":", 1)[1].strip().lower()
        elif lower.startswith("budget:") or lower.startswith("price:"):
            result["price_band"] = part.split(":", 1)[1].strip().lower()
    return result


def parse_tags(value: Any) -> List[str]:
    """
    Normalize scent_tags / setting_tags / gender-like fields into a list[str] (lowercased).
    Can handle:
      - null
      - ["Fresh", "Citrus"]
      - "Fresh, Citrus"
    """
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip().lower() for x in value if str(x).strip()]
    # treat as comma-separated string
    return [t.strip().lower() for t in str(value).split(",") if t.strip()]


def resolve_price_band(filters: Optional[Dict[str, Any]], parsed_goal_band: Optional[str]) -> Optional[str]:
    """
    Normalize price bands between frontend naming and DB values.
    We expect DB price_band values like: "0-100", "100-200", "200+".
    Prefer explicit filters, then parsed goal.
    """
    band = None
    if filters:
        # Support both snake_case and camelCase keys
        band = (
            filters.get("price_band")
            or filters.get("priceBand")
            or filters.get("budget")
        )
    if not band and parsed_goal_band:
        band = parsed_goal_band

    if not band:
        return None

    pb = str(band).strip().lower()

    if pb in ("0-100", "0 – 100", "0 — 100", "0_to_100", "0_100", "0 –100"):
        return "0-100"
    if pb in ("100-200", "100 – 200", "100 — 200", "100_to_200", "100_200"):
        return "100-200"
    if pb in ("200+", "200_plus", "200 – up", "200+ "):
        return "200+"

    # fallback: rough mapping
    if "0" in pb and "100" in pb:
        return "0-100"
    if "100" in pb and "200" in pb:
        return "100-200"
    if "200" in pb or "+" in pb or "plus" in pb:
        return "200+"

    return None


def flexible_price_range(resolved_band: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    20% flex around band boundaries.
    - "0-100"   → 0 .. 120, target_mid ~ 50
    - "100-200" → 80 .. 240, target_mid ~ 150
    - "200+"    → 160 .. None, target_mid ~ 250
    Returns (lo, hi, target_mid)
    """
    if resolved_band == "0-100":
        return 0.0, 120.0, 50.0
    if resolved_band == "100-200":
        return 80.0, 240.0, 150.0
    if resolved_band == "200+":
        return 160.0, None, 250.0
    return None, None, None


def to_float_or_none(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def build_system_prompt() -> str:
    """
    System prompt for RANKING:
    - Model sees a candidate list with INDEXES.
    - It must output JSON with candidate_index, reason, match_score, notes.
    - We embed our scoring preferences here.
    """
    return (
        "You are ScentFeed, a professional perfume recommendation engine used in a mobile app.\n"
        "\n"
        "You are given:\n"
        " - A natural language GOAL which may encode SCENT, SETTING and BUDGET.\n"
        " - A structured SUMMARY of the parsed SCENT / SETTING / PRICE_BAND.\n"
        " - A PROFILE (likes / dislikes / owned / wishlist).\n"
        " - Optional FILTERS (gender, region, raw JSON).\n"
        " - A CANDIDATE LIST of perfumes from the catalog, each with an integer INDEX.\n"
        "\n"
        "Your job:\n"
        " - RANK the candidates for how well they fit the GOAL and PROFILE.\n"
        " - You MUST ONLY choose perfumes from the candidate list (by candidate_index).\n"
        " - Always try to return up to 3 items.\n"
        "\n"
        "SCORING PRIORITIES (think in a rough numeric way):\n"
        " 1) SCENT is most important (like +40 points if the scent family matches what the user asked for).\n"
        " 2) PRICE is also very important:\n"
        "    - Prefer perfumes in the requested price_band exactly.\n"
        "    - If there are not enough in that band, you may include perfumes within roughly 20% of the price range.\n"
        "    - When you step outside the strict band, mention that briefly in the reason.\n"
        " 3) SETTING is helpful but secondary:\n"
        "    - If setting tags match the chosen setting (e.g. 'Work', 'Every Day', 'Date Night', 'Night Out', 'Summer', 'Gym', etc.), treat that like +10–15 points.\n"
        "    - If setting is adjacent (e.g. 'Night Out' vs 'Date Night', 'Work' vs 'Every Day'), give a smaller bonus.\n"
        " 4) PROFILE PERSONALIZATION:\n"
        "    - If the perfume name appears in likes, boost strongly (like +25 points).\n"
        "    - If the brand appears in likes or wishlist, small bonus (like +8 points).\n"
        "    - If the perfume is in dislikes, strongly penalize (like -40 points).\n"
        "    - If the brand is in dislikes, small penalty.\n"
        "    - If the perfume is already owned, lightly down-rank (we still show it if options are limited).\n"
        "    - If the perfume is in wishlist, boost.\n"
        "\n"
        "Occasion hints for SETTING:\n"
        " - 'work' or 'office' → clean, non-offensive, moderate projection.\n"
        " - 'gym' → very fresh, light, not cloying.\n"
        " - 'night out' or 'clubbing' → louder, sexy, attention-grabbing.\n"
        " - 'date night' → intimate, attractive, often sweeter or more sensual.\n"
        " - 'summer' → airy, citrus, fresh, not heavy.\n"
        " - 'winter' or 'cozy' → warmer, sweeter, heavier, more amber/vanilla/spice.\n"
        "\n"
        "DIVERSITY:\n"
        " - When multiple candidates are very close in score, prefer variety in brands and note profiles instead of making all 3 nearly identical.\n"
        "\n"
        "VERY IMPORTANT RESPONSE FORMAT:\n"
        " - You MUST answer with ONLY valid, minified JSON.\n"
        " - No markdown, no explanations outside JSON.\n"
        " - Schema:\n"
        "   {\n"
        "     \"items\": [\n"
        "       {\n"
        "         \"candidate_index\": 0,\n"
        "         \"reason\": \"short explanation (1–2 sentences)\",\n"
        "         \"match_score\": 0–100,\n"
        "         \"notes\": \"optional, short comma-separated key notes\" or null\n"
        "       },\n"
        "       ... (up to 3 items)\n"
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
    scent_tags = ", ".join(parse_tags(row.get("scent_tags")))
    setting_tags = ", ".join(parse_tags(row.get("setting_tags")))
    gender = ", ".join(parse_tags(row.get("gender")))

    def normalize_notes(field: str) -> List[str]:
        v = row.get(field)
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        # treat as string
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
    if gender:
        parts.append(f"gender: {gender}")
    if scent_tags:
        parts.append(f"scent_tags: {scent_tags}")
    if setting_tags:
        parts.append(f"setting_tags: {setting_tags}")
    if notes_str:
        parts.append(f"notes: {notes_str}")
    if description:
        parts.append(f"desc: {description}")

    return " | ".join(parts)


def build_user_prompt(req: RecommendRequest, candidates: List[Dict[str, Any]]) -> str:
    """
    User prompt that includes GOAL, parsed SCENT/SETTING/PRICE, PROFILE, FILTERS and CANDIDATES.
    """
    parsed = parse_structured_goal(req.goal)

    parts: List[str] = []

    parts.append(f"GOAL: {req.goal.strip()}")

    parts.append(
        "STRUCTURED_GOAL:\n"
        f"  scent: {parsed.get('scent')}\n"
        f"  setting: {parsed.get('setting')}\n"
        f"  price_band: {parsed.get('price_band')}"
    )

    prefs = req.prefs
    parts.append(
        "PROFILE:\n"
        f"  likes: {prefs.likes}\n"
        f"  dislikes: {prefs.dislikes}\n"
        f"  owned: {prefs.owned}\n"
        f"  wishlist: {prefs.wishlist}"
    )

    filters = req.filters or {}
    parts.append("FILTERS_JSON:\n" + json.dumps(filters, ensure_ascii=False))

    parts.append(f"MAX_RESULTS: {req.max_results}")

    # Candidate list
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
    Filter by:
      - scent tag (strict)
      - price_band (strict, then ~20% flex if < needed)
      - gender (if provided)
    Then truncate to max_candidates.
    """
    if not catalog:
        return []

    parsed = parse_structured_goal(req.goal)
    target_scent = (parsed.get("scent") or "").lower().strip() or None
    parsed_price_band = parsed.get("price_band")
    filters = req.filters or {}

    # gender filter (optional)
    gender_filter = None
    if "gender" in filters:
        gender_filter = str(filters.get("gender") or "").lower().strip()
    elif "selectedGender" in filters:
        gender_filter = str(filters.get("selectedGender") or "").lower().strip()

    resolved_band = resolve_price_band(filters, parsed_price_band)

    # First pass: scent + gender filter only
    def scent_gender_ok(row: Dict[str, Any]) -> bool:
        if target_scent:
            row_scent_tags = parse_tags(row.get("scent_tags"))
            if target_scent not in row_scent_tags:
                return False

        if gender_filter:
            row_gender_tags = parse_tags(row.get("gender"))
            # keep if exact gender or unisex
            if gender_filter not in row_gender_tags and "unisex" not in row_gender_tags:
                return False

        return True

    base_pool = [r for r in catalog if scent_gender_ok(r)]

    if not base_pool:
        # No exact scent+gender matches -> as a last resort use full catalog
        print(
            f"⚠️ No perfumes matched scent={target_scent!r} and gender={gender_filter!r}; "
            f"falling back to full catalog of {len(catalog)} perfumes."
        )
        base_pool = catalog[:]

    # Second pass: strict price_band filter within base_pool
    strict_candidates: List[Dict[str, Any]] = []
    if resolved_band:
        for r in base_pool:
            rb = (r.get("price_band") or "").strip()
            if rb == resolved_band:
                strict_candidates.append(r)
    else:
        strict_candidates = base_pool[:]

    min_needed = max(3, req.max_results)

    if len(strict_candidates) >= min_needed:
        filtered = strict_candidates
        price_flex_used = False
    else:
        # Not enough strict band matches → apply 20% flex on numeric price
        lo, hi, target_mid = flexible_price_range(resolved_band) if resolved_band else (None, None, None)
        flex_candidates: List[Dict[str, Any]] = []

        if lo is not None:
            for r in base_pool:
                price_val = to_float_or_none(r.get("price"))
                if price_val is None:
                    continue
                if hi is not None:
                    if lo <= price_val <= hi:
                        flex_candidates.append(r)
                else:
                    if price_val >= lo:
                        flex_candidates.append(r)

        # Combine strict + flex (unique by id/name)
        seen_ids = set()
        combined: List[Dict[str, Any]] = []

        def add_unique(rows: List[Dict[str, Any]]) -> None:
            for row in rows:
                key = row.get("id") or row.get("name")
                if not key:
                    continue
                if key in seen_ids:
                    continue
                seen_ids.add(key)
                combined.append(row)

        add_unique(strict_candidates)
        add_unique(flex_candidates)

        # If still not enough, just fill from base_pool by proximity to target_mid
        if len(combined) < min_needed and target_mid is not None:
            remaining: List[Dict[str, Any]] = []
            for r in base_pool:
                key = r.get("id") or r.get("name")
                if not key or key in seen_ids:
                    continue
                price_val = to_float_or_none(r.get("price"))
                if price_val is None:
                    continue
                remaining.append(r)

            remaining.sort(
                key=lambda r: abs(
                    (to_float_or_none(r.get("price")) or target_mid) - target_mid
                )
            )
            for r in remaining:
                key = r.get("id") or r.get("name")
                if key in seen_ids:
                    continue
                seen_ids.add(key)
                combined.append(r)
                if len(combined) >= min_needed:
                    break

        filtered = combined
        price_flex_used = True

    # Truncate to max_candidates
    if len(filtered) > max_candidates:
        filtered = filtered[:max_candidates]

    print(
        f"🧪 Candidate selection: {len(filtered)} perfumes after filters "
        f"(scent={target_scent or 'any'}, price_band={resolved_band or 'any'}, "
        f"gender={gender_filter or 'any'}, price_flex_used={price_flex_used}, "
        f"total_catalog={len(catalog)})"
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
    - Filters candidates by scent + price_band (with 20% flex if needed)
    - Asks OpenAI to RANK by candidate_index
    - Maps back to real perfumes and returns top N (up to 3)
    """
    goal = trimmed_or_none(req.goal)
    if not goal:
        raise HTTPException(status_code=400, detail="Goal must not be empty.")

    max_results = max(1, min(req.max_results, 3))  # app wants 3, but clamp for safety

    # 1) Load catalog
    catalog_all = load_catalog_from_supabase()

    if not catalog_all:
        # No catalog → let the app use offline CSV fallback instead of random hardcoded items
        print("⚠️ Catalog empty or Supabase not configured — returning 503 to trigger offline fallback.")
        raise HTTPException(status_code=503, detail="Catalog unavailable.")

    # 2) Build candidate list for this request
    candidates = filter_candidates_for_request(catalog_all, req, max_candidates=60)

    # If somehow still no candidates (tiny DB), return 503 so iOS uses offline CSV
    if not candidates:
        print("⚠️ No candidates after filtering — returning 503 to trigger offline CSV fallback.")
        raise HTTPException(status_code=503, detail="No candidates after filtering.")

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
            f"candidates={len(candidates)}, goal={goal!r}"
        )

        # Parse JSON from the model
        try:
            data = json.loads(raw_json)
        except Exception:
            print(f"❗️/recommend decode failed. Raw body: {raw_json!r}")
            # Decode error → let app use offline CSV instead of hardcoded perfumes
            raise HTTPException(status_code=502, detail="LLM decode error.")

        # used_profile from model (or default)
        used_profile = data.get("used_profile") or {}
        used_profile_obj = UsedProfile(
            likes=used_profile.get("likes") or req.prefs.likes,
            dislikes=used_profile.get("dislikes") or req.prefs.dislikes,
            owned=used_profile.get("owned") or req.prefs.owned,
            wishlist=used_profile.get("wishlist") or req.prefs.wishlist,
        )

        # items from model: candidate_index + reason + score + notes
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

        # Keep at most max_results
        cleaned_items = cleaned_items[:max_results]

        # If the model gave us no usable items, tell the app to use offline CSV
        if not cleaned_items:
            print("⚠️ Model returned empty/invalid ranked items — 502 to trigger offline CSV fallback.")
            raise HTTPException(status_code=502, detail="LLM returned no items.")

        request_id = data.get("request_id") or completion.id

        return RecommendResponse(
            items=cleaned_items,
            used_profile=used_profile_obj,
            request_id=request_id,
        )

    except HTTPException:
        # Already a proper error for the app to interpret
        raise
    except Exception as e:
        print("❌ /recommend unexpected error:", repr(e))
        # Generic error → iOS will use offline CSV fallback
        raise HTTPException(status_code=500, detail="Internal server error.")



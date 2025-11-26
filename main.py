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

app = FastAPI(title="ScentFeed Web AI", version="4.0.0-catalog-aware")

# ---------- Pydantic models (wire contract with the iOS app) ----------


class Prefs(BaseModel):
    likes: List[str] = Field(default_factory=list)
    dislikes: List[str] = Field(default_factory=list)
    owned: List[str] = Field(default_factory=list)
    wishlist: List[str] = Field(default_factory=list)


class RecommendFilters(BaseModel):
    """
    Filters coming from the app.
    We mainly care about price_band, but we also accept gender/region/raw.
    """
    gender: Optional[str] = None
    region: Optional[str] = None
    price_band: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None


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


BACKEND_VERSION = "4.0.0-catalog-aware"


# ---------- Catalog model (Supabase perfumes) ----------


class PerfumeCatalogItem(BaseModel):
    id: Optional[str] = None
    name: str
    brand: Optional[str] = None
    description: Optional[str] = None
    image_url: Optional[str] = None
    notes_top: List[str] = Field(default_factory=list)
    notes_heart: List[str] = Field(default_factory=list)
    notes_base: List[str] = Field(default_factory=list)
    projection: Optional[str] = None
    longevity: Optional[str] = None
    price: Optional[float] = None            # raw numeric price if available
    price_band: Optional[str] = None         # "0-100", "100-200", "200+"
    celebrity_endorsers: List[str] = Field(default_factory=list)

    @property
    def all_notes(self) -> List[str]:
        return list(dict.fromkeys(self.notes_top + self.notes_heart + self.notes_base))

    def to_catalog_line(self) -> str:
        """
        One compact line used inside the model prompt.

        Example:
        "Dior Sauvage by Dior | price_band: 100-200 | price: 135 | notes: bergamot, ambroxan | projection: strong | longevity: long"
        """
        brand = self.brand or "Unknown"
        pb = (self.price_band or "unknown").strip()
        price_str = f"{self.price}" if self.price is not None else "unknown"
        notes = ", ".join(self.all_notes) if self.all_notes else ""
        proj = self.projection or "unknown"
        long_ = self.longevity or "unknown"

        parts = [
            f"{self.name} by {brand}",
            f"price_band: {pb}",
            f"price: {price_str}",
        ]
        if notes:
            parts.append(f"notes: {notes}")
        parts.append(f"projection: {proj}")
        parts.append(f"longevity: {long_}")
        return " | ".join(parts)


# ---------- Supabase catalog loading (cached) ----------

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_ANON_KEY")

_catalog_lock = threading.Lock()
_catalog_cache: Optional[Tuple[float, List[PerfumeCatalogItem]]] = None
_CATALOG_TTL_SECONDS = 300  # 5 minutes


def load_catalog_from_supabase() -> List[PerfumeCatalogItem]:
    """
    Fetch all rows from public.perfumes via Supabase REST.
    Uses a very simple in-memory cache to avoid hammering the DB.
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

        items: List[PerfumeCatalogItem] = []
        for row in raw:
            try:
                price_value = row.get("price")
                # Normalize numeric price if possible
                if isinstance(price_value, str):
                    try:
                        price_value = float(price_value)
                    except Exception:
                        price_value = None

                item = PerfumeCatalogItem(
                    id=str(row.get("id")) if row.get("id") is not None else None,
                    name=str(row.get("name") or "").strip() or "Untitled",
                    brand=row.get("brand"),
                    description=row.get("description"),
                    image_url=row.get("image_url"),
                    notes_top=row.get("notes_top") or [],
                    notes_heart=row.get("notes_heart") or [],
                    notes_base=row.get("notes_base") or [],
                    projection=row.get("projection"),
                    longevity=row.get("longevity"),
                    price=price_value,
                    price_band=row.get("price_band"),
                    celebrity_endorsers=row.get("celebrity_endorsers") or [],
                )
                items.append(item)
            except Exception as e:
                print("⚠️ Failed to parse perfume row from Supabase:", e, "row=", row)

        with _catalog_lock:
            _catalog_cache = (now, items)

        print(f"✅ Loaded {len(items)} perfumes from Supabase catalog.")
        return items

    except Exception as e:
        print("❌ Error loading perfumes from Supabase:", repr(e))
        return []


# ---------- Price band helpers ----------

def normalize_price_band(pb: Optional[str]) -> Optional[str]:
    """
    Normalize various spellings of price band to canonical values:
    - "0-100", "100-200", "200+"
    """
    if not pb:
        return None

    p = pb.strip().lower().replace(" ", "")
    # Replace common dashes with a simple hyphen
    p = p.replace("–", "-").replace("—", "-")

    if p.startswith("0-100") or p == "0-100":
        return "0-100"
    if p.startswith("100-200") or p == "100-200":
        return "100-200"
    if p.startswith("200+") or p.startswith("200plus") or p.startswith("200-") or p == "200+":
        return "200+"

    # legacy labels:
    if p in ("budget", "$", "low", "under100"):
        return "0-100"
    if p in ("mid", "$$", "medium", "100to200"):
        return "100-200"
    if p in ("luxury", "$$$", "high", "over200", "200plus"):
        return "200+"

    return None


def infer_price_band_from_goal(goal: str) -> Optional[str]:
    """
    Lightweight heuristic: look at the GOAL string for clues about budget.

    We expect patterns like:
      - "budget: 0-100"
      - "budget: 100-200"
      - "budget: 200+"
    and also support minor variations with different dashes.
    """
    g = goal.lower().replace("–", "-").replace("—", "-")
    if "0-100" in g or "0to100" in g or "0- 100" in g:
        return "0-100"
    if "100-200" in g or "100to200" in g or "100- 200" in g:
        return "100-200"
    if "200+" in g or "200plus" in g or "200+" in g.replace(" ", ""):
        return "200+"
    return None


def resolve_price_band(req: RecommendRequest) -> Optional[str]:
    """
    Final price band:
      1) filters.price_band if provided
      2) else inferred from goal text
    """
    if req.filters and req.filters.price_band:
        pb = normalize_price_band(req.filters.price_band)
        if pb:
            return pb
    inferred = infer_price_band_from_goal(req.goal)
    return normalize_price_band(inferred) if inferred else None


def filter_catalog_by_price_band(
    catalog: List[PerfumeCatalogItem],
    price_band: Optional[str],
) -> List[PerfumeCatalogItem]:
    if not catalog:
        return []

    if price_band is None:
        # No filter → return entire catalog
        return catalog

    filtered = [
        p for p in catalog
        if normalize_price_band(p.price_band) == price_band
    ]

    if not filtered:
        print(f"⚠️ No perfumes matched price_band={price_band}; falling back to full catalog.")
        return catalog

    return filtered


# ---------- Utility helpers ----------

def trimmed_or_none(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    v = value.strip()
    return v or None


def build_system_prompt() -> str:
    """
    System prompt that teaches the model how to behave:
    - ONLY pick from catalog we provide.
    - Occasion-aware, region-aware, budget-aware.
    - STRICT JSON.
    """
    return (
        "You are ScentFeed, a professional perfume recommendation engine.\n"
        "\n"
        "You are given:\n"
        " - A natural language GOAL.\n"
        " - A PROFILE (likes / dislikes / owned / wishlist).\n"
        " - Optional FILTERS (region, gender, price_band).\n"
        " - A CATALOG of perfumes actually available in the app.\n"
        "\n"
        "Rules:\n"
        " - You MUST ONLY recommend perfumes that appear in the CATALOG list.\n"
        " - Do NOT invent perfumes that are not in the catalog.\n"
        " - Respect price_band as much as possible; if price_band is set, stick to that band.\n"
        " - If PROFILE shows strong likes (e.g. vanilla, oud, floral), bias toward catalog items whose notes match that.\n"
        " - If PROFILE shows dislikes, avoid those notes/styles.\n"
        " - Avoid recommending things the user already OWNS unless there are not enough new options.\n"
        " - If the GOAL clearly changes (different vibe, setting, budget), explore different perfumes, not always the same three.\n"
        "\n"
        "Occasion examples:\n"
        " - \"oud for Eid\" → prefer rich, elegant ouds that feel festive and respectful.\n"
        " - \"vanilla for Christmas\" → cozy, warm, gourmand vanillas that fit winter and holidays.\n"
        " - \"office safe\" → clean, non-offensive, moderate projection.\n"
        " - \"gym\" → very fresh, light, not cloying.\n"
        " - \"wedding\" → special, elegant, long-lasting.\n"
        "\n"
        "VERY IMPORTANT RESPONSE FORMAT:\n"
        " - You MUST answer with ONLY valid, minified JSON.\n"
        " - No markdown, no explanations outside JSON.\n"
        " - Schema:\n"
        "   {\n"
        "     \"items\": [\n"
        "       {\n"
        "         \"name\": \"<MUST match name from CATALOG>\",\n"
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
        "Never include trailing commas. Never add comments. Only valid JSON.\n"
    )


def build_user_prompt(req: RecommendRequest, catalog: List[PerfumeCatalogItem], price_band: Optional[str]) -> str:
    """
    Convert the incoming request into a user message that also includes
    the CATALOG (the only allowed options).
    """
    parts: List[str] = []

    parts.append(f"GOAL: {req.goal.strip()}")

    # Profile
    prefs = req.prefs
    parts.append(
        "PROFILE:\n"
        f"  likes: {prefs.likes}\n"
        f"  dislikes: {prefs.dislikes}\n"
        f"  owned: {prefs.owned}\n"
        f"  wishlist: {prefs.wishlist}"
    )

    # Filters (if any)
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

    parts.append(f"INFERRED_PRICE_BAND: {price_band or 'null'}")
    parts.append(f"MAX_RESULTS: {req.max_results}")

    # Catalog
    lines = [p.to_catalog_line() for p in catalog]
    catalog_block = "\n".join(lines)
    parts.append(
        "CATALOG (you MUST only choose perfumes from this list):\n"
        + catalog_block
    )

    return "\n\n".join(parts)


def clamp_items_to_catalog(
    items: List[Dict[str, Any]],
    catalog: List[PerfumeCatalogItem],
    max_results: int,
) -> List[RecommendItem]:
    """
    - Ensure every item actually exists in the catalog.
    - Drop anything that doesn't match by name.
    """
    if not catalog:
        return []

    by_name = {p.name.lower(): p for p in catalog}

    cleaned: List[RecommendItem] = []
    for item in items:
        raw_name = (item.get("name") or "").strip()
        if not raw_name:
            continue
        key = raw_name.lower()
        perfume = by_name.get(key)
        if not perfume:
            # Model tried to hallucinate or spelling mismatch; drop it
            continue

        reason = item.get("reason") or "No reason provided."
        match_score = item.get("match_score") or 75
        notes = item.get("notes")

        try:
            match_score_int = int(match_score)
        except Exception:
            match_score_int = 75

        cleaned.append(
            RecommendItem(
                name=perfume.name,
                reason=reason,
                match_score=max(0, min(100, match_score_int)),
                notes=notes,
            )
        )

    if not cleaned:
        return []

    max_results = max(1, max_results)
    return cleaned[:max_results]


def fallback_from_catalog(goal: str, catalog: List[PerfumeCatalogItem], max_results: int) -> List[RecommendItem]:
    """
    If the model fails or hallucinates, we still want to show *something*,
    but strictly from the catalog (and already filtered by price_band).
    """
    if not catalog:
        return []

    max_results = max(1, max_results)
    # Simple deterministic choice: first N by name
    sorted_catalog = sorted(catalog, key=lambda p: (p.brand or "", p.name))
    chosen = sorted_catalog[:max_results]

    base_reason = f"Fits your request: {goal.strip()} (safe catalog fallback)."

    return [
        RecommendItem(
            name=p.name,
            reason=base_reason,
            match_score=70,
            notes=", ".join(p.all_notes) if p.all_notes else None,
        )
        for p in chosen
    ]


def build_global_fallback_items(goal: str) -> List[RecommendItem]:
    """
    Last-resort fallback if Supabase is completely unavailable or empty.
    These are not catalog-bound, but guarantee the app always shows something.
    """
    g = goal.strip()
    base_reason = f"Fits your request: {g}."

    return [
        RecommendItem(
            name="Dior Sauvage Eau de Toilette",
            reason=base_reason + " Versatile fresh 'blue' scent that works for almost any casual or office situation.",
            match_score=88,
            notes="bergamot, pepper, ambroxan",
        ),
        RecommendItem(
            name="Parfums de Marly Layton",
            reason=base_reason + " Sweet, spicy and mass-appealing; great for dates and evenings with strong performance.",
            match_score=86,
            notes="apple, vanilla, cardamom",
        ),
        RecommendItem(
            name="Chanel Bleu de Chanel Eau de Parfum",
            reason=base_reason + " Modern, classy blue scent that feels put-together and safe in many settings.",
            match_score=84,
            notes="citrus, incense, woods",
        ),
    ]


# ---------- Routes ----------


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    # Also check if catalog is reachable (best-effort)
    catalog_ok = bool(load_catalog_from_supabase())
    status = "healthy-with-catalog" if catalog_ok else "healthy-no-catalog"
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
    Catalog-aware version:
      - Reads perfumes from Supabase (including price_band & price).
      - Infers price_band from filters/goal.
      - Filters catalog by that band.
      - Uses OpenAI to pick from catalog.
      - Falls back to catalog-only or global defaults if needed.
    """
    goal = trimmed_or_none(req.goal)
    if not goal:
        raise HTTPException(status_code=400, detail="Goal must not be empty.")

    # Load catalog from Supabase
    catalog_all = load_catalog_from_supabase()
    if not catalog_all:
        print("⚠️ Catalog empty or Supabase not configured — using global hard fallback recommendations.")
        fallback_items = build_global_fallback_items(goal)
        return RecommendResponse(
            items=fallback_items,
            used_profile=UsedProfile(
                likes=req.prefs.likes,
                dislikes=req.prefs.dislikes,
                owned=req.prefs.owned,
                wishlist=req.prefs.wishlist,
            ),
            request_id="fallback-no-catalog",
        )

    # Resolve price band and pre-filter catalog
    price_band = resolve_price_band(req)
    catalog = filter_catalog_by_price_band(catalog_all, price_band)

    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(req, catalog, price_band)

    try:
        started = time.time()

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.6,
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

        print(f"🌐 /recommend status: 200, latency={latency_ms}ms, price_band={price_band}")

        # Parse JSON from the model
        try:
            data = json.loads(raw_json)
        except Exception as e:
            print(f"❗️/recommend decode failed. Raw body: {raw_json}")
            # Try catalog-based fallback first
            items = fallback_from_catalog(goal, catalog, req.max_results)
            if not items:
                items = build_global_fallback_items(goal)
            return RecommendResponse(
                items=items,
                used_profile=UsedProfile(
                    likes=req.prefs.likes,
                    dislikes=req.prefs.dislikes,
                    owned=req.prefs.owned,
                    wishlist=req.prefs.wishlist,
                ),
                request_id=f"fallback-decode-error-{completion.id}",
            )

        # used_profile from model (or default)
        used_profile = data.get("used_profile") or {}
        used_profile_obj = UsedProfile(
            likes=used_profile.get("likes") or req.prefs.likes,
            dislikes=used_profile.get("dislikes") or req.prefs.dislikes,
            owned=used_profile.get("owned") or req.prefs.owned,
            wishlist=used_profile.get("wishlist") or req.prefs.wishlist,
        )

        # items from model
        items_data = data.get("items") or []
        cleaned_items = clamp_items_to_catalog(items_data, catalog, req.max_results)

        if not cleaned_items:
            print("⚠️ No valid items after clamping — using catalog-based fallback.")
            cleaned_items = fallback_from_catalog(goal, catalog, req.max_results)
            if not cleaned_items:
                print("⚠️ Catalog-based fallback also empty — using global hard fallback.")
                cleaned_items = build_global_fallback_items(goal)

        request_id = data.get("request_id") or completion.id

        return RecommendResponse(
            items=cleaned_items,
            used_profile=used_profile_obj,
            request_id=request_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        print("❌ /recommend unexpected error:", repr(e))
        # On unexpected error, try catalog-based fallback, then global.
        items = fallback_from_catalog(goal, catalog_all, req.max_results)
        if not items:
            items = build_global_fallback_items(goal)
        return RecommendResponse(
            items=items,
            used_profile=UsedProfile(
                likes=req.prefs.likes,
                dislikes=req.prefs.dislikes,
                owned=req.prefs.owned,
                wishlist=req.prefs.wishlist,
            ),
            request_id="fallback-internal-error",
        )



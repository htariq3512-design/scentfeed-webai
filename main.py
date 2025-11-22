# main.py
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

app = FastAPI(title="ScentFeed Web AI", version="2.0.0")

# ---------- Pydantic models (wire contract with the iOS app) ----------


class Prefs(BaseModel):
    likes: List[str] = Field(default_factory=list)
    dislikes: List[str] = Field(default_factory=list)
    owned: List[str] = Field(default_factory=list)
    wishlist: List[str] = Field(default_factory=list)


class RecommendFilters(BaseModel):
    """
    Filters coming from the app. We mainly care about price_band,
    but we forward gender/region/tags to the model as hints.
    """
    gender: Optional[str] = None          # "male" | "female" | "unisex"
    region: Optional[str] = None          # "pakistan", "gulf", "europe", ...
    price_band: Optional[str] = None      # "0-100", "100-200", "200+", or "budget"/"mid"/"luxury"
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


BACKEND_VERSION = "2.0.0-scentfeed-catalog-aware"

# ---------- Catalog model (Supabase perfumes) ----------


class PerfumeCatalogItem(BaseModel):
    id: str
    name: str
    brand: Optional[str] = None
    description: Optional[str] = None
    price_tier: Optional[str] = None   # "0-100", "100-200", "200+"
    projection: Optional[str] = None
    longevity: Optional[str] = None
    notes_top: List[str] = Field(default_factory=list)
    notes_heart: List[str] = Field(default_factory=list)
    notes_base: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    celebrity_endorsers: List[str] = Field(default_factory=list)

    @property
    def all_notes(self) -> List[str]:
        return list(dict.fromkeys(self.notes_top + self.notes_heart + self.notes_base))

    def to_catalog_line(self) -> str:
        """
        One-line compact description for the model.
        Example:
        "Black Opium by Yves Saint Laurent | price: 100-200 | notes: coffee, vanilla | proj: medium | long: long | tags: gourmand, sweet"
        """
        brand = self.brand or "Unknown"
        price = self.price_tier or "unknown"
        notes = ", ".join(self.all_notes) if self.all_notes else ""
        tags = ", ".join(self.tags) if self.tags else ""
        proj = self.projection or "unknown"
        long = self.longevity or "unknown"

        parts = [
            f"{self.name} by {brand}",
            f"price: {price}",
        ]
        if notes:
            parts.append(f"notes: {notes}")
        if tags:
            parts.append(f"tags: {tags}")
        parts.append(f"projection: {proj}")
        parts.append(f"longevity: {long}")
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
                # Supabase returns arrays or null — pydantic will handle defaults
                item = PerfumeCatalogItem(
                    id=str(row.get("id")),
                    name=str(row.get("name") or "").strip() or "Untitled",
                    brand=row.get("brand"),
                    description=row.get("description"),
                    price_tier=row.get("price_tier"),
                    projection=row.get("projection"),
                    longevity=row.get("longevity"),
                    notes_top=row.get("notes_top") or [],
                    notes_heart=row.get("notes_heart") or [],
                    notes_base=row.get("notes_base") or [],
                    tags=row.get("tags") or [],
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


def resolve_price_band(filters: Optional[RecommendFilters]) -> Optional[str]:
    """
    Normalize price bands between frontend naming and DB values.
    Frontend might send:
      - \"0-100\", \"100-200\", \"200+\"
      - OR \"budget\", \"mid\", \"luxury\"
    DB uses \"0-100\", \"100-200\", \"200+\" in price_tier.
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

    # legacy labels:
    if pb in ("budget", "$", "low"):
        return "0-100"
    if pb in ("mid", "$$", "medium"):
        return "100-200"
    if pb in ("luxury", "$$$", "high"):
        return "200+"

    return None


def filter_catalog_for_request(
    catalog: List[PerfumeCatalogItem],
    req: RecommendRequest
) -> List[PerfumeCatalogItem]:
    """
    Apply coarse filters (price band now, later we can add tags/gender/region heuristics).
    """
    if not catalog:
        return []

    price_band = resolve_price_band(req.filters)
    filtered = catalog

    if price_band:
        filtered = [p for p in filtered if (p.price_tier or "").strip() == price_band]

    # If filtering becomes too strict and nothing is left, fall back to full catalog.
    if not filtered:
        print("⚠️ No perfumes matched filters; falling back to full catalog.")
        filtered = catalog

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
    - occasion-aware, region-aware.
    - STRICT JSON.
    """
    return (
        "You are ScentFeed, a professional perfume recommendation engine.\n"
        "\n"
        "You are given:\n"
        " - A natural language GOAL.\n"
        " - A PROFILE (likes / dislikes / owned / wishlist).\n"
        " - Optional FILTERS (region, gender, price_band).\n"
        " - A CATALOG of perfumes that are ACTUALLY available in the app.\n"
        "\n"
        "Rules:\n"
        " - You MUST ONLY recommend perfumes that appear in the CATALOG list.\n"
        " - Do NOT invent perfumes that are not in the catalog.\n"
        " - Respect price_band as much as possible; if price_band is set, prefer perfumes within that band.\n"
        " - If PROFILE shows strong likes (e.g. vanilla, oud, floral), bias toward catalog items whose notes/tags match that.\n"
        " - If PROFILE shows dislikes, avoid those notes/styles.\n"
        " - Avoid recommending things the user already OWNS unless there are not enough new options.\n"
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


def build_user_prompt(req: RecommendRequest, catalog: List[PerfumeCatalogItem]) -> str:
    """
    Convert the incoming request into a single structured user message
    that also includes the CATALOG (the only allowed options).
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
        filters = req.filters
        parts.append(
            "FILTERS:\n"
            f"  gender: {filters.gender}\n"
            f"  region: {filters.region}\n"
            f"  price_band: {filters.price_band}\n"
            f"  raw: {filters.raw}"
        )

    parts.append(f"MAX_RESULTS: {req.max_results}")

    # Catalog
    lines = [p.to_catalog_line() for p in catalog]
    catalog_block = "\n".join(lines)
    parts.append(
        "CATALOG (you MUST only choose perfumes from this list):\n"
        + catalog_block
    )

    return "\n\n".join(parts)


def clamp_items_to_catalog_and_price(
    items: List[Dict[str, Any]],
    catalog: List[PerfumeCatalogItem],
    req: RecommendRequest
) -> List[RecommendItem]:
    """
    - Ensure every item actually exists in the catalog.
    - Re-check price band on the Python side.
    - If something doesn't match, drop it.
    """
    # Build lookup by lowercase name
    by_name = {p.name.lower(): p for p in catalog}
    price_band = resolve_price_band(req.filters)

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

        # Enforce price band if specified
        if price_band is not None:
            if (perfume.price_tier or "").strip() != price_band:
                # Out of band, skip
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

    # If we dropped everything (model ignored catalog or price too strict),
    # relax: ignore price band and just take the best matches we can map,
    # or take a few top catalog entries as a last resort.
    if not cleaned:
        print("⚠️ clamp_items_to_catalog_and_price: no valid items after clamping, relaxing constraints.")

        # First, try mapping by name ignoring price band
        by_name_all = {p.name.lower(): p for p in catalog}
        for item in items:
            raw_name = (item.get("name") or "").strip()
            if not raw_name:
                continue
            key = raw_name.lower()
            perfume = by_name_all.get(key)
            if not perfume:
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

    # Finally, trim to max_results
    max_results = max(1, req.max_results)
    return cleaned[:max_results]


# ---------- Routes ----------


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        ok=True,
        status="healthy",
        version=BACKEND_VERSION,
        model="gpt-4o-mini",
    )


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest) -> RecommendResponse:
    """
    Main recommendation endpoint called by the iOS app.
    Now catalog-aware: strictly picks from Supabase perfumes.
    """
    goal = trimmed_or_none(req.goal)
    if not goal:
        raise HTTPException(status_code=400, detail="Goal must not be empty.")

    # Load catalog from Supabase
    catalog_all = load_catalog_from_supabase()
    if not catalog_all:
        raise HTTPException(
            status_code=500,
            detail="Perfume catalog is empty or Supabase is not configured."
        )

    catalog = filter_catalog_for_request(catalog_all, req)

    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(req, catalog)

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

        print(f"🌐 /recommend status: 200, latency={latency_ms}ms")

        try:
            data = json.loads(raw_json)
        except Exception as e:
            print(f"❗️/recommend decode failed. Raw body: {raw_json}")
            raise HTTPException(status_code=502, detail=f"Failed to decode model JSON: {e}")

        used_profile = data.get("used_profile") or {}
        used_profile_obj = UsedProfile(
            likes=used_profile.get("likes") or req.prefs.likes,
            dislikes=used_profile.get("dislikes") or req.prefs.dislikes,
            owned=used_profile.get("owned") or req.prefs.owned,
            wishlist=used_profile.get("wishlist") or req.prefs.wishlist,
        )

        items_data = data.get("items") or []
        cleaned_items = clamp_items_to_catalog_and_price(items_data, catalog, req)

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
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")



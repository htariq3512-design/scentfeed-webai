#
# ScentFeed Web AI backend – simple, online-only, price-aware (NO Supabase)
#
# - Uses OpenAI only (no DB / catalog)
# - Takes: GOAL + PROFILE + FILTERS (including price_band)
# - Teaches the model about rough price brackets and brand tiers
# - Strongly discourages always returning the same 2–3 perfumes
# - Always returns at least some items via a generic fallback
#
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os
import time
import json

from openai import OpenAI

# ---------- OpenAI client ----------

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ---------- FastAPI app ----------

app = FastAPI(title="ScentFeed Web AI", version="3.1.0-simple-price-aware")

# ---------- Pydantic models (wire contract with the iOS app) ----------


class Prefs(BaseModel):
    likes: List[str] = Field(default_factory=list)
    dislikes: List[str] = Field(default_factory=list)
    owned: List[str] = Field(default_factory=list)
    wishlist: List[str] = Field(default_factory=list)


class RecommendFilters(BaseModel):
    """
    Filters coming from the app. For now we only forward them to the model
    as hints – we do NOT enforce any DB price band.
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


BACKEND_VERSION = "3.1.0-simple-price-aware"


# ---------- Utility helpers ----------

def trimmed_or_none(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    v = value.strip()
    return v or None


def normalize_price_band(pb: Optional[str]) -> Optional[str]:
    """
    Normalize price band hints from frontend to one of:
    - "0-100", "100-200", "200+"
    or None.
    """
    if not pb:
        return None
    s = pb.strip().lower()
    if s in ("0-100", "0 – 100", "0 — 100", "0_to_100", "0_100", "0 –100"):
        return "0-100"
    if s in ("100-200", "100 – 200", "100 — 200", "100_to_200", "100_200"):
        return "100-200"
    if s in ("200+", "200_plus", "200 – up", "200+ "):
        return "200+"
    if s in ("budget", "$", "low"):
        return "0-100"
    if s in ("mid", "$$", "medium"):
        return "100-200"
    if s in ("luxury", "$$$", "high", "premium"):
        return "200+"
    return None


def build_system_prompt() -> str:
    """
    System prompt that teaches the model how to behave:
    - Good perfume recommender
    - Occasion-aware, budget-aware
    - Knows rough price tiers for big brands
    - STRICT JSON
    """
    return (
        "You are ScentFeed, a professional perfume recommendation engine.\n"
        "\n"
        "You are given:\n"
        " - A natural language GOAL.\n"
        " - A PROFILE (likes / dislikes / owned / wishlist).\n"
        " - Optional FILTERS (region, gender, price_band).\n"
        "\n"
        "Your job:\n"
        " - Suggest realistic perfumes that match the GOAL and PROFILE.\n"
        " - Use your knowledge of real perfume prices and brand tiers.\n"
        " - Respect price_band as much as possible:\n"
        "     * '0-100' → mostly designer, affordable lines, Zara, Bath & Body Works,\n"
        "       The Body Shop, Abercrombie, Montblanc, Nautica, etc. Avoid ultra-luxury niche.\n"
        "     * '100-200' → many designer EDPs and some entry niche. Mix of popular brands.\n"
        "     * '200+' → luxury & niche houses (Creed, Maison Francis Kurkdjian,\n"
        "       Parfums de Marly, Tom Ford Private Blend, Xerjoff, Roja, Armani Privé, Louis Vuitton, etc.).\n"
        " - For '0-100', you should NOT recommend ultra-luxury niche houses like Creed,\n"
        "   Parfums de Marly, MFK, Xerjoff, Roja, Tom Ford Private Blend, Louis Vuitton,\n"
        "   Armani Privé, etc. These belong more naturally in '200+' budgets.\n"
        " - If PROFILE shows strong likes (e.g. vanilla, oud, floral), bias toward notes and styles that match.\n"
        " - If PROFILE shows dislikes, avoid those.\n"
        " - Avoid recommending things the user already OWNS unless there are not enough new options.\n"
        "\n"
        "Diversity rule:\n"
        " - When the GOAL clearly changes (vibe, setting, budget), do NOT always return\n"
        "   the same 2–3 perfumes. Prefer to vary the suggestions so different\n"
        "   scenarios feel different for the user.\n"
        "\n"
        "Occasion examples:\n"
        " - 'oud for Eid' → prefer rich, elegant ouds that feel festive and respectful.\n"
        " - 'vanilla for Christmas' → cozy, warm, gourmand vanillas that fit winter and holidays.\n"
        " - 'office safe' → clean, non-offensive, moderate projection.\n"
        " - 'gym' → very fresh, light, not cloying.\n"
        " - 'wedding' → special, elegant, long-lasting.\n"
        "\n"
        "VERY IMPORTANT RESPONSE FORMAT:\n"
        " - You MUST answer with ONLY valid, minified JSON.\n"
        " - No markdown, no explanations outside JSON.\n"
        " - Schema:\n"
        "   {\n"
        "     \"items\": [\n"
        "       {\n"
        "         \"name\": \"string\",\n"
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


def build_user_prompt(req: RecommendRequest) -> str:
    """
    Convert the incoming request into a structured user message.
    No DB catalog – just GOAL + PROFILE + FILTERS + price hints.
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
        norm_price = normalize_price_band(req.filters.price_band)
        parts.append(
            "FILTERS:\n"
            f"  gender: {req.filters.gender}\n"
            f"  region: {req.filters.region}\n"
            f"  price_band_normalized: {norm_price}\n"
            f"  raw: {req.filters.raw}"
        )
    else:
        parts.append("FILTERS:\n  gender: null\n  region: null\n  price_band_normalized: null\n  raw: null")

    parts.append(f"MAX_RESULTS: {req.max_results}")

    return "\n\n".join(parts)


# ---------- Hard fallback (never return empty) ----------

def build_fallback_items(goal: str) -> List[RecommendItem]:
    """
    Last-resort recommendations if:
    - Model returns no items, or
    - JSON decode fails, or
    - Internal error.
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
            name="Yves Saint Laurent Y Eau de Parfum",
            reason=base_reason + " Modern, fresh and slightly sweet, great for daily wear and evenings.",
            match_score=86,
            notes="apple, ginger, amber woods",
        ),
        RecommendItem(
            name="Chanel Bleu de Chanel Eau de Parfum",
            reason=base_reason + " Classy blue scent that feels put-together and safe in many settings.",
            match_score=84,
            notes="citrus, incense, woods",
        ),
    ]


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
    Simple online-only version:
    - No Supabase / DB
    - Just OpenAI + price-band-aware instructions
    - Always falls back to some reasonable designer choices
    """
    goal = trimmed_or_none(req.goal)
    if not goal:
        raise HTTPException(status_code=400, detail="Goal must not be empty.")

    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(req)

    try:
        started = time.time()

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.7,  # a bit higher to encourage variety
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

        # Parse JSON from the model
        try:
            data = json.loads(raw_json)
        except Exception as e:
            print(f"❗️/recommend decode failed. Raw body: {raw_json}")
            fallback_items = build_fallback_items(goal)
            return RecommendResponse(
                items=fallback_items,
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
        items: List[RecommendItem] = []

        for raw_item in items_data:
            name = (raw_item.get("name") or "").strip()
            if not name:
                continue
            reason = raw_item.get("reason") or "No reason provided."
            match_score = raw_item.get("match_score") or 75
            notes = raw_item.get("notes")

            try:
                score_int = int(match_score)
            except Exception:
                score_int = 75

            items.append(
                RecommendItem(
                    name=name,
                    reason=reason,
                    match_score=max(0, min(100, score_int)),
                    notes=notes,
                )
            )

        # If the model gave us no usable items, use fallback.
        if not items:
            print("⚠️ Model returned empty/invalid items — using hard fallback recommendations.")
            items = build_fallback_items(goal)

        request_id = data.get("request_id") or completion.id

        return RecommendResponse(
            items=items,
            used_profile=used_profile_obj,
            request_id=request_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        print("❌ /recommend unexpected error:", repr(e))
        # On unexpected error, also try hard fallback instead of pure 500.
        fallback_items = build_fallback_items(goal)
        return RecommendResponse(
            items=fallback_items,
            used_profile=UsedProfile(
                likes=req.prefs.likes,
                dislikes=req.prefs.dislikes,
                owned=req.prefs.owned,
                wishlist=req.prefs.wishlist,
            ),
            request_id="fallback-internal-error",
        )



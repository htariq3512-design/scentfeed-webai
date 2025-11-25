#
# Simple ScentFeed Web AI backend WITH LOGGING:
# - NO Supabase catalog enforcement for now (pure model suggestions)
# - Uses OpenAI to generate 1–N recommendations
# - Always returns items (falls back to a default trio if the model fails)
# - LOGS every recommendation into Supabase recommend_events table
#
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os
import time
import json

from openai import OpenAI
import requests

# ---------- OpenAI client ----------

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ---------- FastAPI app ----------

app = FastAPI(title="ScentFeed Web AI", version="3.1.0-simple-online-logged")

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


BACKEND_VERSION = "3.1.0-simple-online-logged"

# ---------- Supabase logging config ----------

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_ANON_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    print("⚠️ Supabase logging disabled – missing SUPABASE_URL or SERVICE/ANON KEY.")
else:
    print(f"✅ Supabase logging enabled -> {SUPABASE_URL}/rest/v1/recommend_events")

SUPABASE_LOG_URL = SUPABASE_URL + "/rest/v1/recommend_events"


def log_recommend_event(
    uid: Optional[str],
    goal: str,
    price_band: Optional[str],
    prefs: Prefs,
    items: List[RecommendItem],
    source: str,
    request_id: Optional[str],
) -> None:
    """
    Send recommendation logs to Supabase recommend_events table.

    Expected columns in recommend_events:
      - uid (text)
      - goal (text)
      - price_band (text)
      - likes (jsonb)
      - dislikes (jsonb)
      - owned (jsonb)
      - wishlist (jsonb)
      - items (jsonb)
      - source (text)  -- e.g. "online-openai" or "fallback"
      - request_id (text)
      - created_at (timestamptz, default now())
    """
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        # Logging disabled
        return

    payload = {
        "uid": uid or "unknown",
        "goal": goal,
        "price_band": price_band,
        "likes": prefs.likes,
        "dislikes": prefs.dislikes,
        "owned": prefs.owned,
        "wishlist": prefs.wishlist,
        "items": [item.dict() for item in items],
        "source": source,
        "request_id": request_id,
    }

    try:
        resp = requests.post(
            SUPABASE_LOG_URL,
            json=payload,
            headers={
                "apikey": SUPABASE_SERVICE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                "Content-Type": "application/json",
                "Prefer": "return=minimal",
            },
            timeout=5,
        )
        if resp.status_code >= 300:
            print(f"❗️ Supabase log failed: {resp.status_code} {resp.text}")
        else:
            print("📡 Supabase log stored.")
    except Exception as e:
        print("❌ Supabase log error:", repr(e))


# ---------- Utility helpers ----------

def trimmed_or_none(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    v = value.strip()
    return v or None


def build_system_prompt() -> str:
    """
    System prompt that teaches the model how to behave:
    - Good perfume recommender
    - Occasion-aware, budget-aware
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
        " - Respect price_band as much as possible.\n"
        " - If PROFILE shows strong likes (e.g. vanilla, oud, floral), bias toward those notes.\n"
        " - If PROFILE shows dislikes, avoid those.\n"
        " - Avoid recommending things the user already OWNS unless there are not enough new options.\n"
        " - If the GOAL clearly changes (different vibe, setting, budget), you should explore different perfumes.\n"
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
    No catalog now – just GOAL + PROFILE + FILTERS.
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
    - No Supabase / DB lookups
    - Just OpenAI + safety fallback
    - Logs EVERY call to Supabase recommend_events
    """
    goal = trimmed_or_none(req.goal)
    if not goal:
        raise HTTPException(status_code=400, detail="Goal must not be empty.")

    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(req)

    # Snapshot price_band string once (for logging)
    price_band = req.filters.price_band if req.filters else None

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

        # Parse JSON from the model
        try:
            data = json.loads(raw_json)
        except Exception as e:
            print(f"❗️/recommend decode failed. Raw body: {raw_json}")
            fallback_items = build_fallback_items(goal)

            # LOG fallback
            log_recommend_event(
                uid=req.uid,
                goal=goal,
                price_band=price_band,
                prefs=req.prefs,
                items=fallback_items,
                source="fallback-decode-error",
                request_id=f"fallback-decode-error-{completion.id}",
            )

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
        source = "online-openai"
        request_id = data.get("request_id") or completion.id

        if not items:
            print("⚠️ Model returned empty/invalid items — using hard fallback recommendations.")
            items = build_fallback_items(goal)
            source = "fallback-empty-model"
            request_id = f"{request_id}-fallback-empty"

        # LOG FINAL ITEMS
        log_recommend_event(
            uid=req.uid,
            goal=goal,
            price_band=price_band,
            prefs=req.prefs,
            items=items,
            source=source,
            request_id=request_id,
        )

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

        # LOG fallback
        log_recommend_event(
            uid=req.uid,
            goal=goal,
            price_band=price_band,
            prefs=req.prefs,
            items=fallback_items,
            source="fallback-internal-error",
            request_id="fallback-internal-error",
        )

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



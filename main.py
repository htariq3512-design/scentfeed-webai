# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os
import time

from openai import OpenAI

# ---------- OpenAI client ----------

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ---------- FastAPI app ----------

app = FastAPI(title="ScentFeed Web AI", version="1.8.0")

# ---------- Pydantic models (wire contract with the iOS app) ----------


class Prefs(BaseModel):
    likes: List[str] = Field(default_factory=list)
    dislikes: List[str] = Field(default_factory=list)
    owned: List[str] = Field(default_factory=list)
    wishlist: List[str] = Field(default_factory=list)


class RecommendFilters(BaseModel):
    """
    Placeholder for future filters. Right now we accept it
    and pass it into the model as metadata, but we don't
    hard-enforce on the backend. This keeps the contract
    forward-compatible.
    """
    # Example optional fields – keep them very loose for now
    gender: Optional[str] = None          # "male" | "female" | "unisex"
    region: Optional[str] = None          # "pakistan", "gulf", "europe"...
    price_band: Optional[str] = None      # "budget", "mid", "luxury"
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


BACKEND_VERSION = "1.8.0-scentfeed-occasion-aware"


# ---------- Utility helpers ----------

def trimmed_or_none(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    v = value.strip()
    return v or None


def build_system_prompt() -> str:
    """
    System prompt that teaches the model how to behave:
    - occasion-aware (Eid, Christmas, office, first date, wedding, etc)
    - region-aware when the user mentions a country or city (e.g. Pakistan)
    - always returns STRICT JSON, no extra text.
    """
    return (
        "You are ScentFeed, a professional perfume recommendation engine.\n"
        "\n"
        "Your job:\n"
        " - Read the user's natural language GOAL (their vibe / use case).\n"
        " - Look at their PROFILE (likes / dislikes / owned / wishlist).\n"
        " - Optionally look at FILTERS (region, gender, etc.).\n"
        " - Return 3–5 perfumes that best match.\n"
        "\n"
        "Occasion handling examples:\n"
        " - \"oud for Eid\" → prefer rich, elegant ouds that feel festive and respectful.\n"
        " - \"vanilla for Christmas\" → cozy, warm, gourmand vanillas that fit winter and holidays.\n"
        " - \"office safe\" → clean, non-offensive, moderate projection.\n"
        " - \"gym\" → very fresh, light, not cloying.\n"
        " - \"wedding\" → special, elegant, long-lasting.\n"
        "\n"
        "Region handling examples:\n"
        " - If the goal mentions a region or country (e.g. Pakistan, India, Gulf, Middle East), "
        "   you should *bias* toward:\n"
        "   - popular houses or styles in that region,\n"
        "   - scents that are realistically available there or culturally aligned.\n"
        " - If you don't know strong local brands, you *may* still use designer / niche "
        "   houses, but try to choose ones that are reasonably findable online from that region.\n"
        "\n"
        "Profile handling:\n"
        " - If the user likes vanilla/amber/oud, prefer fragrances where those notes are central.\n"
        " - If they dislike something (e.g. strong citrus or heavy incense), avoid those notes.\n"
        " - Don't repeat things they already own unless the goal explicitly suggests flanker ideas.\n"
        "\n"
        "VERY IMPORTANT RESPONSE FORMAT:\n"
        " - You MUST answer with ONLY valid, minified JSON.\n"
        " - No markdown, no prose outside JSON.\n"
        " - It MUST match this schema exactly:\n"
        "   {\n"
        "     \"items\": [\n"
        "       {\n"
        "         \"name\": \"...\",\n"
        "         \"reason\": \"...\",\n"
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
        "If the goal is unclear or impossible, still return the JSON structure, but with an empty\n"
        "\"items\" array and a short explanation in one dummy item if needed.\n"
        "Never include trailing commas. Never include comments. Only valid JSON."
    )


def build_user_prompt(req: RecommendRequest) -> str:
    """
    Convert the incoming request into a single structured user message
    the model can reason over.
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

    return "\n\n".join(parts)


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
            temperature=0.7,
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

        # Debug log (shortened)
        print(f"🌐 /recommend status: 200, latency={latency_ms}ms")
        # If you want, you can log raw_json here in DEBUG only.

        # Parse JSON into our Pydantic model
        try:
            # raw_json is a JSON string; Pydantic can parse from dict
            import json

            data = json.loads(raw_json)
        except Exception as e:
            print(f"❗️/recommend decode failed. Raw body: {raw_json}")
            raise HTTPException(status_code=502, detail=f"Failed to decode model JSON: {e}")

        # Fill in used_profile defaults if the model omits anything
        used_profile = data.get("used_profile") or {}
        used_profile_obj = UsedProfile(
            likes=used_profile.get("likes") or req.prefs.likes,
            dislikes=used_profile.get("dislikes") or req.prefs.dislikes,
            owned=used_profile.get("owned") or req.prefs.owned,
            wishlist=used_profile.get("wishlist") or req.prefs.wishlist,
        )

        # Build response
        items_data = data.get("items") or []
        items: List[RecommendItem] = []
        for item in items_data:
            # Defensive parsing
            name = item.get("name") or "Unknown fragrance"
            reason = item.get("reason") or "No reason provided."
            match_score = item.get("match_score") or 75
            notes = item.get("notes")
            items.append(
                RecommendItem(
                    name=name,
                    reason=reason,
                    match_score=int(match_score),
                    notes=notes,
                )
            )

        request_id = data.get("request_id") or completion.id

        return RecommendResponse(
            items=items,
            used_profile=used_profile_obj,
            request_id=request_id,
        )

    except HTTPException:
        # Re-raise any explicit HTTP errors
        raise
    except Exception as e:
        print("❌ /recommend unexpected error:", repr(e))
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")




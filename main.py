# main.py
#
# ScentFeed Web AI backend (Supabase catalog + TikTok/Meta-style retrieval + LLM rerank)
#
# ONLINE LOGIC:
# - Always uses Supabase catalog (no hardcoded perfume fallbacks)
# - Strict scent match (if provided)
# - Strict price band first; if < N, relax ~20%; if still < N, closest-price fill
# - Pre-rank candidates using your CSV scores (hype/mass/vers) + profile taste
# - Ask OpenAI to rank by candidate_index (no name mismatch)
#
# If Supabase or OpenAI fails → return error (iOS will fallback to offline CSV).
#
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pydantic.config import ConfigDict
from typing import List, Optional, Dict, Any, Tuple
import os
import time
import json
import threading
import random

import requests
from openai import OpenAI


# ---------- OpenAI client ----------
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ---------- FastAPI app ----------
app = FastAPI(title="ScentFeed Web AI", version="5.0.0-online-rerank")


# ---------- Models ----------
class Prefs(BaseModel):
    likes: List[str] = Field(default_factory=list)
    dislikes: List[str] = Field(default_factory=list)
    owned: List[str] = Field(default_factory=list)
    wishlist: List[str] = Field(default_factory=list)


class RecommendFilters(BaseModel):
    gender: Optional[str] = None
    region: Optional[str] = None
    price_band: Optional[str] = None  # "0-100" | "100-200" | "200+"
    raw: Optional[Dict[str, Any]] = None


class RecommendRequest(BaseModel):
    """
    Backwards compatible request:
    Supports BOTH:
      - old shape: { goal, max_results, prefs, filters }
      - new shape: { goal, max_results, selected_scent, selected_setting, price_band, gender, likes, dislikes, owned, wishlist }
    """
    model_config = ConfigDict(extra="allow")  # accept unknown keys to avoid 422

    uid: Optional[str] = None
    goal: str
    max_results: int = 3

    # Old shape
    prefs: Optional[Prefs] = None
    filters: Optional[RecommendFilters] = None

    # New/flat shape (Swift-friendly)
    selected_scent: Optional[str] = None
    selected_setting: Optional[str] = None
    price_band: Optional[str] = None
    gender: Optional[str] = None

    likes: List[str] = Field(default_factory=list)
    dislikes: List[str] = Field(default_factory=list)
    owned: List[str] = Field(default_factory=list)
    wishlist: List[str] = Field(default_factory=list)


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


BACKEND_VERSION = "5.0.0-online-rerank"


# ---------- Supabase loading (cached) ----------
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = (
    os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    or os.environ.get("SUPABASE_ANON_KEY")
)

_catalog_lock = threading.Lock()
_catalog_cache: Optional[Tuple[float, List[Dict[str, Any]]]] = None
_CATALOG_TTL_SECONDS = 300  # 5 minutes


def load_catalog_from_supabase() -> List[Dict[str, Any]]:
    global _catalog_cache

    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise RuntimeError("Supabase not configured (SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY missing).")

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

    resp = requests.get(url, headers=headers, params=params, timeout=12)
    resp.raise_for_status()
    raw = resp.json()
    if not isinstance(raw, list):
        raise RuntimeError(f"Unexpected Supabase response shape: {type(raw)}")

    items: List[Dict[str, Any]] = []
    for row in raw:
        name = (row.get("name") or "").strip()
        if not name:
            continue
        items.append(row)

    with _catalog_lock:
        _catalog_cache = (now, items)

    return items


# ---------- Normalization helpers ----------
def norm(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    t = str(s).strip().lower()
    return t or None


def split_tags(value: Any) -> List[str]:
    """
    Accept:
      - list[str]
      - comma-separated string
      - None
    """
    if value is None:
        return []
    if isinstance(value, list):
        out = []
        for x in value:
            tx = norm(x)
            if tx:
                out.append(tx)
        return out
    s = str(value)
    parts = [p.strip().lower() for p in s.split(",")]
    return [p for p in parts if p]


def safe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def clamp_int(x: Any, lo: int, hi: int, default: int) -> int:
    try:
        v = int(x)
        return max(lo, min(hi, v))
    except Exception:
        return default


# ---------- Request unification ----------
def unify_request(req: RecommendRequest) -> Dict[str, Any]:
    # Profile lists
    likes = req.likes[:] if req.likes else []
    dislikes = req.dislikes[:] if req.dislikes else []
    owned = req.owned[:] if req.owned else []
    wishlist = req.wishlist[:] if req.wishlist else []

    # If old prefs exists, merge them in (avoid breaking older clients)
    if req.prefs:
        likes = (req.prefs.likes or []) + likes
        dislikes = (req.prefs.dislikes or []) + dislikes
        owned = (req.prefs.owned or []) + owned
        wishlist = (req.prefs.wishlist or []) + wishlist

    # Filters / selected fields
    selected_scent = req.selected_scent
    selected_setting = req.selected_setting
    price_band = req.price_band
    gender = req.gender

    if req.filters:
        if not gender:
            gender = req.filters.gender
        if not price_band:
            price_band = req.filters.price_band

    return {
        "goal": req.goal.strip(),
        "max_results": max(1, int(req.max_results)),
        "selected_scent": norm(selected_scent),
        "selected_setting": norm(selected_setting),
        "price_band": resolve_price_band(price_band),
        "gender": norm(gender),
        "profile": {
            "likes": dedupe_list(likes),
            "dislikes": dedupe_list(dislikes),
            "owned": dedupe_list(owned),
            "wishlist": dedupe_list(wishlist),
        },
    }


def dedupe_list(xs: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        t = (x or "").strip()
        if not t:
            continue
        k = t.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(t)
    return out


def resolve_price_band(pb: Optional[str]) -> Optional[str]:
    if not pb:
        return None
    p = pb.strip().lower()

    if p in ("0-100", "0 – 100", "0—100", "0_to_100", "0_100", "under100", "under 100"):
        return "0-100"
    if p in ("100-200", "100 – 200", "100—200", "100_to_200", "100_200"):
        return "100-200"
    if p in ("200+", "200_plus", "200 plus", "over200", "over 200", "200 and up"):
        return "200+"

    if "0" in p and "100" in p:
        return "0-100"
    if "100" in p and "200" in p:
        return "100-200"
    if "200" in p or "+" in p or "plus" in p:
        return "200+"

    return None


# ---------- Price band logic ----------
def price_strict_range(band: str) -> Tuple[float, Optional[float]]:
    if band == "0-100":
        return (0.0, 100.0)
    if band == "100-200":
        return (100.0, 200.0)
    if band == "200+":
        return (200.0, None)
    return (0.0, None)


def price_relaxed_range(band: str) -> Tuple[float, Optional[float]]:
    # ~20% variance
    if band == "0-100":
        return (0.0, 120.0)
    if band == "100-200":
        return (80.0, 240.0)
    if band == "200+":
        return (160.0, None)
    return (0.0, None)


def in_price_range(price: Optional[float], r: Tuple[float, Optional[float]]) -> bool:
    if price is None:
        return False
    lo, hi = r
    if hi is None:
        return price >= lo
    return lo <= price <= hi


def price_target(band: str) -> float:
    if band == "0-100":
        return 75.0
    if band == "100-200":
        return 150.0
    if band == "200+":
        return 250.0
    return 150.0


# ---------- Setting intent weights ----------
def setting_intent_weights(setting: Optional[str]) -> Tuple[float, float, float]:
    """
    Returns (hype_w, mass_w, vers_w) based on setting intent.
    Mirrors your offline logic.
    """
    s = setting or ""
    if "night" in s and "out" in s:
        return (0.45, 0.25, 0.10)
    if "date" in s:
        return (0.25, 0.45, 0.15)
    if "work" in s:
        return (0.10, 0.25, 0.50)
    if "every" in s:
        return (0.15, 0.30, 0.45)
    if "travel" in s:
        return (0.15, 0.30, 0.50)
    if "gym" in s:
        return (0.05, 0.25, 0.50)
    if "wedding" in s:
        return (0.20, 0.40, 0.35)
    if "summer" in s or "winter" in s or "fall" in s or "spring" in s:
        return (0.15, 0.30, 0.35)
    if "cozy" in s:
        return (0.20, 0.35, 0.30)
    return (0.20, 0.35, 0.35)


# ---------- Candidate shaping ----------
def row_to_features(row: Dict[str, Any]) -> Dict[str, Any]:
    # Tags + notes
    scent_tags = split_tags(row.get("scent_tags"))
    setting_tags = split_tags(row.get("setting_tags"))

    notes_top = split_tags(row.get("notes_top"))
    notes_heart = split_tags(row.get("notes_heart"))
    notes_base = split_tags(row.get("notes_base"))
    all_notes = list(dict.fromkeys(notes_top + notes_heart + notes_base))

    # Scores
    hype = safe_float(row.get("hype_score")) or 50.0
    mass = safe_float(row.get("mass_appeal_score")) or 50.0
    vers = safe_float(row.get("versatility_score")) or 50.0

    price = safe_float(row.get("price"))

    return {
        "name": (row.get("name") or "").strip(),
        "brand": (row.get("brand") or "Unknown").strip(),
        "price": price,
        "gender": norm(row.get("gender")),
        "scent_tags": scent_tags,
        "setting_tags": setting_tags,
        "notes": all_notes,
        "hype_score": hype,
        "mass_appeal_score": mass,
        "versatility_score": vers,
        "projection": (row.get("projection") or "").strip() or None,
        "longevity": (row.get("longevity") or "").strip() or None,
        "description": (row.get("description") or "").strip() or None,
    }


def build_profile_signature(catalog_feats: List[Dict[str, Any]], profile: Dict[str, List[str]]) -> Dict[str, Any]:
    # Build map by normalized name
    by_name = {norm(f["name"]) or f["name"].lower(): f for f in catalog_feats}

    likes = {norm(x) for x in profile.get("likes", []) if norm(x)}
    dislikes = {norm(x) for x in profile.get("dislikes", []) if norm(x)}

    liked_tags = set()
    disliked_tags = set()
    liked_notes = set()
    disliked_notes = set()

    for n in likes:
        f = by_name.get(n)
        if not f:
            continue
        liked_tags.update(f["scent_tags"])
        liked_tags.update(f["setting_tags"])
        liked_notes.update(f["notes"])

    for n in dislikes:
        f = by_name.get(n)
        if not f:
            continue
        disliked_tags.update(f["scent_tags"])
        disliked_tags.update(f["setting_tags"])
        disliked_notes.update(f["notes"])

    return {
        "likes": list(likes),
        "dislikes": list(dislikes),
        "liked_tags": list(liked_tags),
        "disliked_tags": list(disliked_tags),
        "liked_notes": list(liked_notes),
        "disliked_notes": list(disliked_notes),
    }


def overlap_ratio(a: List[str], bset: set) -> float:
    if not a or not bset:
        return 0.0
    inter = sum(1 for x in a if x in bset)
    denom = max(1, min(len(a), len(bset)))
    return inter / float(denom)


def price_tier_bonus(tier: str) -> float:
    return {"strict": 0.60, "relaxed": 0.30, "closest": 0.10, "noprice": -0.15}.get(tier, 0.0)


def pre_score_candidate(
    feat: Dict[str, Any],
    selected_setting: Optional[str],
    band: Optional[str],
    tier: str,
    profile_sig: Dict[str, Any],
) -> float:
    s = 1.0
    s += price_tier_bonus(tier)

    # Setting soft boost
    if selected_setting and selected_setting != "any":
        if selected_setting in feat["setting_tags"]:
            s += 0.25
        else:
            s -= 0.05

    # Quality priors from your 3 scores (0..100)
    hw, mw, vw = setting_intent_weights(selected_setting)
    hype = max(0.0, min(1.0, (feat["hype_score"] / 100.0)))
    mass = max(0.0, min(1.0, (feat["mass_appeal_score"] / 100.0)))
    vers = max(0.0, min(1.0, (feat["versatility_score"] / 100.0)))
    s += hype * hw + mass * mw + vers * vw

    # Taste DNA overlap
    liked_tags = set(profile_sig.get("liked_tags") or [])
    disliked_tags = set(profile_sig.get("disliked_tags") or [])
    liked_notes = set(profile_sig.get("liked_notes") or [])
    disliked_notes = set(profile_sig.get("disliked_notes") or [])

    tag_union = list(dict.fromkeys(feat["scent_tags"] + feat["setting_tags"]))
    s += overlap_ratio(tag_union, liked_tags) * 0.35
    s -= overlap_ratio(tag_union, disliked_tags) * 0.60
    s += overlap_ratio(feat["notes"], liked_notes) * 0.35
    s -= overlap_ratio(feat["notes"], disliked_notes) * 0.60

    # Owned / wishlist / dislikes by name
    nm = norm(feat["name"]) or ""
    if nm and nm in set(profile_sig.get("disliked_names", []) or []):
        s -= 1.0

    return s


def retrieve_candidates(
    feats: List[Dict[str, Any]],
    selected_scent: Optional[str],
    selected_setting: Optional[str],
    band: Optional[str],
    gender: Optional[str],
    max_results: int,
    profile_sig: Dict[str, Any],
    max_candidates: int = 80,
) -> List[Dict[str, Any]]:
    pool = feats

    # Hard scent match if selected (100%)
    if selected_scent and selected_scent not in ("any",):
        pool = [f for f in pool if selected_scent in f["scent_tags"]]
        if not pool:
            return []

    # Gender soft filter (allow unisex)
    if gender and gender not in ("any",):
        filtered = []
        for f in pool:
            g = f.get("gender") or "unisex"
            if gender == "unisex" or g == "unisex" or g == gender:
                filtered.append(f)
        if filtered:
            pool = filtered

    # Price staged retrieval
    staged: List[Tuple[Dict[str, Any], str]] = []

    if band:
        strict_r = price_strict_range(band)
        relaxed_r = price_relaxed_range(band)

        strict = []
        relaxed = []
        leftovers = []
        no_price = []

        for f in pool:
            price = f.get("price")
            if price is None:
                no_price.append(f)
                continue
            if in_price_range(price, strict_r):
                strict.append(f)
            elif in_price_range(price, relaxed_r):
                relaxed.append(f)
            else:
                leftovers.append(f)

        # strict first
        for f in strict:
            staged.append((f, "strict"))

        # relaxed next (only if needed)
        if len(staged) < max_candidates:
            for f in relaxed:
                staged.append((f, "relaxed"))

        # closest fill by distance (only if needed)
        if len(staged) < max_candidates:
            tgt = price_target(band)
            leftovers.sort(key=lambda x: abs((x.get("price") or tgt) - tgt))
            for f in leftovers:
                staged.append((f, "closest"))
                if len(staged) >= max_candidates:
                    break

        # finally add no_price
        if len(staged) < max_candidates:
            for f in no_price:
                staged.append((f, "noprice"))
                if len(staged) >= max_candidates:
                    break

    else:
        staged = [(f, "strict" if f.get("price") is not None else "noprice") for f in pool]

    # Pre-score to choose best candidates for LLM
    scored = []
    for f, tier in staged:
        sc = pre_score_candidate(
            feat=f,
            selected_setting=selected_setting,
            band=band,
            tier=tier,
            profile_sig=profile_sig,
        )
        scored.append((sc, f, tier))

    # Sort + small shuffle within top bands to avoid “same 3 forever”
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[: max_candidates]

    # Mild exploration: shuffle last 20% of the candidate set
    if len(top) > 20:
        head = top[: int(len(top) * 0.8)]
        tail = top[int(len(top) * 0.8):]
        random.shuffle(tail)
        top = head + tail

    candidates = []
    for _, f, tier in top:
        f2 = dict(f)
        f2["_price_tier"] = tier
        candidates.append(f2)

    return candidates


# ---------- LLM prompt ----------
def build_system_prompt() -> str:
    return (
        "You are ScentFeed, a professional perfume recommender with TikTok/Meta ranking discipline.\n"
        "You MUST choose only from the provided CANDIDATE LIST by integer candidate_index.\n"
        "You are given:\n"
        " - GOAL (human text)\n"
        " - Selected scent / setting / price_band / gender\n"
        " - PROFILE lists (likes/dislikes/owned/wishlist)\n"
        " - Taste DNA (derived from liked/disliked perfumes: liked_notes/tags, disliked_notes/tags)\n"
        " - Candidates contain: tags, notes, price, and 3 scores: hype_score / mass_appeal_score / versatility_score (0-100)\n"
        "\n"
        "Ranking rules:\n"
        " - Scent match is HARD (already filtered). Do not violate it.\n"
        " - Prefer strict price matches; relaxed/closest only if needed.\n"
        " - Use setting intent:\n"
        "   • Night Out: hype matters most\n"
        "   • Date Night: mass appeal matters most\n"
        "   • Work/Everyday/Travel: versatility matters most\n"
        " - Avoid DISLIKED notes/tags and DISLIKED perfumes.\n"
        " - Avoid recommending OWNED unless insufficient options.\n"
        " - If wishlist overlaps, it’s a positive signal.\n"
        " - Provide 1-2 sentence reason.\n"
        "\n"
        "Output JSON ONLY (minified), schema:\n"
        "{\n"
        "  \"items\": [\n"
        "    {\"candidate_index\": 0, \"reason\": \"...\", \"match_score\": 0-100, \"notes\": \"comma notes\" or null},\n"
        "    ...\n"
        "  ],\n"
        "  \"used_profile\": {\"likes\":[],\"dislikes\":[],\"owned\":[],\"wishlist\":[]},\n"
        "  \"request_id\": \"string\"\n"
        "}\n"
    )


def summarize_candidate(idx: int, f: Dict[str, Any]) -> str:
    name = f["name"]
    brand = f.get("brand") or "Unknown"
    price = f.get("price")
    tier = f.get("_price_tier") or "strict"
    scent_tags = ", ".join(f.get("scent_tags") or [])
    setting_tags = ", ".join(f.get("setting_tags") or [])
    notes = ", ".join(f.get("notes") or [])
    hype = f.get("hype_score")
    mass = f.get("mass_appeal_score")
    vers = f.get("versatility_score")
    return (
        f"[{idx}] {name} by {brand} | price: {price} | price_tier: {tier} | "
        f"hype_score: {hype}, mass_appeal_score: {mass}, versatility_score: {vers} | "
        f"scent_tags: {scent_tags} | setting_tags: {setting_tags} | notes: {notes}"
    )


def build_user_prompt(u: Dict[str, Any], profile_sig: Dict[str, Any], candidates: List[Dict[str, Any]]) -> str:
    lines = [summarize_candidate(i, c) for i, c in enumerate(candidates)]
    return "\n\n".join([
        f"GOAL: {u['goal']}",
        f"SELECTED:\n  scent: {u['selected_scent']}\n  setting: {u['selected_setting']}\n  price_band: {u['price_band']}\n  gender: {u['gender']}",
        "PROFILE:\n"
        f"  likes: {u['profile']['likes']}\n"
        f"  dislikes: {u['profile']['dislikes']}\n"
        f"  owned: {u['profile']['owned']}\n"
        f"  wishlist: {u['profile']['wishlist']}",
        "TASTE_DNA:\n"
        f"  liked_tags: {profile_sig.get('liked_tags') or []}\n"
        f"  disliked_tags: {profile_sig.get('disliked_tags') or []}\n"
        f"  liked_notes: {profile_sig.get('liked_notes') or []}\n"
        f"  disliked_notes: {profile_sig.get('disliked_notes') or []}",
        f"MAX_RESULTS: {u['max_results']}",
        "CANDIDATE LIST (choose ONLY by candidate_index):\n" + "\n".join(lines),
    ])


# ---------- Routes ----------
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        catalog = load_catalog_from_supabase()
        status = f"healthy-with-catalog:{len(catalog)}"
    except Exception as e:
        status = f"healthy-no-catalog:{repr(e)}"
    return HealthResponse(ok=True, status=status, version=BACKEND_VERSION, model="gpt-4o-mini")


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest) -> RecommendResponse:
    # 0) Validate
    if not req.goal or not req.goal.strip():
        raise HTTPException(status_code=400, detail="Goal must not be empty.")

    # 1) Unify request shape
    u = unify_request(req)
    max_results = u["max_results"]

    # 2) Load Supabase catalog (ONLINE ONLY)
    try:
        raw_rows = load_catalog_from_supabase()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Supabase unavailable: {repr(e)}")

    if not raw_rows:
        raise HTTPException(status_code=503, detail="Supabase catalog empty.")

    # 3) Normalize features
    feats = [row_to_features(r) for r in raw_rows]
    feats = [f for f in feats if f.get("name")]

    # 4) Build taste DNA
    profile_sig = build_profile_signature(feats, u["profile"])

    # 5) Retrieve candidates (filtered + pre-ranked)
    candidates = retrieve_candidates(
        feats=feats,
        selected_scent=u["selected_scent"],
        selected_setting=u["selected_setting"],
        band=u["price_band"],
        gender=u["gender"],
        max_results=max_results,
        profile_sig={
            **profile_sig,
            "disliked_names": [norm(x) for x in u["profile"]["dislikes"] if norm(x)],
        },
        max_candidates=80,
    )

    if not candidates:
        # No hardcoded perfumes. Signal failure; iOS should fall back to offline.
        raise HTTPException(status_code=404, detail="No candidates matched constraints (scent/price/gender).")

    # 6) LLM rerank
    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(u, profile_sig, candidates)

    try:
        started = time.time()
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.35,
            max_tokens=700,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        latency_ms = int((time.time() - started) * 1000)
        raw_json = completion.choices[0].message.content
        print(f"🌐 /recommend status: 200, latency={latency_ms}ms, candidates={len(candidates)}")

        try:
            data = json.loads(raw_json)
        except Exception:
            raise HTTPException(status_code=503, detail=f"Model JSON decode failed. body={raw_json!r}")

        items_data = data.get("items") or []
        cleaned: List[RecommendItem] = []

        for it in items_data:
            idx = it.get("candidate_index")
            try:
                idx = int(idx)
            except Exception:
                continue
            if idx < 0 or idx >= len(candidates):
                continue

            c = candidates[idx]
            name = c.get("name")
            if not name:
                continue

            reason = it.get("reason") or "Recommended based on your selections."
            score = clamp_int(it.get("match_score"), 0, 100, 75)
            notes = it.get("notes")

            cleaned.append(
                RecommendItem(
                    name=name,
                    reason=reason,
                    match_score=score,
                    notes=notes if isinstance(notes, str) else None,
                )
            )
            if len(cleaned) >= max_results:
                break

        if not cleaned:
            raise HTTPException(status_code=503, detail="Model returned no usable ranked items.")

        return RecommendResponse(
            items=cleaned,
            used_profile=UsedProfile(
                likes=u["profile"]["likes"],
                dislikes=u["profile"]["dislikes"],
                owned=u["profile"]["owned"],
                wishlist=u["profile"]["wishlist"],
            ),
            request_id=(data.get("request_id") or completion.id),
        )

    except HTTPException:
        raise
    except Exception as e:
        # No fake recommendations; let iOS fall back.
        raise HTTPException(status_code=503, detail=f"Online recommend failed: {repr(e)}")



import os, json
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError
from openai import OpenAI
 
# ---------- Config ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in Render environment variables.")

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="ScentFeed WebAI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Lock down to your app’s domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Models ----------
class AskRequest(BaseModel):
    query: str
    max_sources: int = Field(3, ge=1, le=10)

class AskResponse(BaseModel):
    ok: bool = True
    answer: Optional[str] = None
    error: Optional[str] = None

class RecommendRequest(BaseModel):
    likes: List[str] = Field(default_factory=list)
    dislikes: List[str] = Field(default_factory=list)
    owned: List[str] = Field(default_factory=list)
    wishlist: List[str] = Field(default_factory=list)
    avoid_notes: List[str] = Field(default_factory=list)
    user_id: Optional[str] = None
    max_results: int = Field(3, ge=1, le=5)

class PerfumeRec(BaseModel):
    name: str
    description: str
    why: str
    score: float = Field(ge=0, le=1)

class RecommendResponse(BaseModel):
    ok: bool = True
    recommendations: List[PerfumeRec]
    error: Optional[str] = None

# ---------- Routes ----------
@app.get("/health")
def health():
    return {"ok": True, "status": "healthy"}

@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest):
    try:
        msg = [
            {"role": "system", "content": "You are a helpful, concise research assistant."},
            {"role": "user", "content": payload.query},
        ]
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=msg,
            temperature=0.3,
        )
        answer = res.choices[0].message.content
        return AskResponse(ok=True, answer=answer)
    except Exception as e:
        return AskResponse(ok=False, error=str(e))

@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    """
    Produces 1–3 tailored perfume recommendations.
    Personalization signals: likes, dislikes, owned, wishlist, avoid_notes.
    Output is strict JSON the iOS app can decode.
    """
    try:
        system = (
            "You are ScentFeed’s fragrance sommelier. "
            "Recommend modern perfumes using up-to-date general knowledge, "
            "but strictly tailor picks to the user's likes/dislikes/owned/wishlist/avoid_notes. "
            "Never include anything from the owned list as a recommendation. "
            "Focus on broad availability and give tight, vivid descriptions. "
            "Return STRICT JSON ONLY."
        )

        # Guidance for the model about the JSON we expect
        json_contract = {
            "recommendations": [
                {
                    "name": "string (perfume or line)",
                    "description": "string (notes, vibe, season, longevity in one compact paragraph)",
                    "why": "string (1-2 sentences explaining why it matches their tastes)",
                    "score": 0.0  # 0..1 confidence based on the fit to likes/avoid/dislikes
                }
            ]
        }

        user_context = {
            "likes": req.likes,
            "dislikes": req.dislikes,
            "owned": req.owned,
            "wishlist": req.wishlist,
            "avoid_notes": req.avoid_notes,
            "max_results": req.max_results,
        }

        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    "Personalization signals:\n"
                    + json.dumps(user_context, ensure_ascii=False, indent=2)
                    + "\n\n"
                    "Constraints:\n"
                    f"- Return at most {req.max_results} items.\n"
                    "- Avoid owned items.\n"
                    "- Respect avoid_notes and dislikes.\n"
                    "- Prefer crowd-available scents unless niche is explicitly liked.\n"
                    "- Output JSON EXACTLY in this schema (no prose outside JSON):\n"
                    + json.dumps(json_contract, ensure_ascii=False, indent=2)
                ),
            },
        ]

        # Ask for JSON object only
        chat = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.4,
            response_format={"type": "json_object"},
        )

        raw = chat.choices[0].message.content or "{}"
        data = json.loads(raw)

        # Validate & trim to max_results
        recs_raw = data.get("recommendations", [])
        # Normalize fields and clamp count
        normalized = []
        for r in recs_raw[: req.max_results]:
            try:
                normalized.append(
                    PerfumeRec(
                        name=str(r.get("name", "")).strip(),
                        description=str(r.get("description", "")).strip(),
                        why=str(r.get("why", "")).strip(),
                        score=float(r.get("score", 0)),
                    )
                )
            except (ValueError, ValidationError):
                # Skip invalid entries
                continue

        if not normalized:
            raise ValueError("Model returned empty or invalid recommendations.")

        return RecommendResponse(ok=True, recommendations=normalized)

    except Exception as e:
        return RecommendResponse(ok=False, recommendations=[], error=str(e))




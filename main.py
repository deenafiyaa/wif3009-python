"""
main.py — FastAPI backend for RedFlagAI Dating Profile Auditor.

Serves the frontend HTML, handles API routes, and keeps the
AI provider key server-side so the browser never sees it.

Supports:
  - OpenRouter  (OPENROUTER_API_KEY in .env)  ← default
  - Google Gemini (GEMINI_API_KEY in .env)    ← fallback
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from audit import ProfileInput, run_audit
from samples import SAMPLE_PROFILES

# ── LOAD .env ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL   = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct")
GEMINI_API_KEY     = os.getenv("GEMINI_API_KEY", "")   # fallback

# Determine which provider is active
def _active_provider() -> str:
    if OPENROUTER_API_KEY and OPENROUTER_API_KEY != "your-openrouter-api-key-here":
        return "openrouter"
    if GEMINI_API_KEY and GEMINI_API_KEY != "your-gemini-api-key-here":
        return "gemini"
    return "none"

PROVIDER = _active_provider()

if PROVIDER == "none":
    print("\n⚠️  WARNING: No AI API key found in backend/.env")
    print("   Add OPENROUTER_API_KEY (https://openrouter.ai/keys)")
    print("   or GEMINI_API_KEY (https://aistudio.google.com/app/apikey)\n")
else:
    model_display = OPENROUTER_MODEL if PROVIDER == "openrouter" else "gemini"
    print(f"✓ Provider: {PROVIDER.upper()} · Model: {model_display}")

# ── APP SETUP ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="RedFlagAI",
    description="Dating Profile Auditor — 5-stage ensemble AI pipeline",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR)), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR))


# ── ROUTES ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main SPA page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/health")
async def health():
    """
    Health check.
    Returns which AI provider is active and whether it is configured.
    The frontend uses this to show the green/red status dot.
    """
    provider = _active_provider()
    model    = OPENROUTER_MODEL if provider == "openrouter" else ("gemini-1.5-flash" if provider == "gemini" else "none")
    return {
        "status":             "ok",
        "api_key_configured": provider != "none",
        "provider":           provider,
        "model":              model,
    }


@app.get("/api/samples")
async def get_samples():
    """Return sample profiles for the frontend dropdown selector."""
    return {"samples": SAMPLE_PROFILES}


@app.post("/api/audit")
async def audit_profile(profile: ProfileInput):
    """
    Main endpoint — runs the full 5-stage pipeline.

    The AI key is injected here from environment variables.
    The browser only calls this URL; it never touches the AI provider directly.
    """
    if _active_provider() == "none":
        raise HTTPException(
            status_code=503,
            detail="No AI API key configured. Add OPENROUTER_API_KEY to backend/.env and restart.",
        )

    if not profile.bio.strip():
        raise HTTPException(status_code=422, detail="Bio cannot be empty.")

    if len(profile.bio) > 5000:
        raise HTTPException(status_code=422, detail="Bio must be under 5000 characters.")

    try:
        result = await run_audit(profile)
        return result

    except Exception as e:
        msg = str(e).lower()
        if any(w in msg for w in ["api_key", "invalid", "permission", "authentication", "401"]):
            raise HTTPException(status_code=401, detail="Invalid API key. Check backend/.env.")
        if any(w in msg for w in ["quota", "rate", "429", "resource_exhausted", "too many"]):
            raise HTTPException(status_code=429, detail="API rate limit hit. Wait a moment and retry.")
        raise HTTPException(status_code=500, detail=f"Audit pipeline error: {str(e)}")


# ── DEV ENTRY POINT ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)

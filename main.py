"""
main.py — FastAPI backend for RedFlagAI Dating Profile Auditor.
The Anthropic API key is kept server-side in .env and never exposed to the client.
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

# ── LOAD ENV ───────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
if not ANTHROPIC_API_KEY or ANTHROPIC_API_KEY.startswith("sk-ant-api03-YOUR"):
    print("\n⚠️  WARNING: ANTHROPIC_API_KEY is not set in backend/.env")
    print("   Copy backend/.env.example to backend/.env and add your key.\n")

# ── APP SETUP ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="RedFlagAI",
    description="Dating Profile Auditor — 5-stage ensemble AI pipeline",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files & templates live one level up (frontend/)
FRONTEND_DIR = BASE_DIR.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(FRONTEND_DIR / "templates"))


# ── ROUTES ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main SPA."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/health")
async def health():
    """Health check — also reports whether API key is configured."""
    key_ok = bool(ANTHROPIC_API_KEY) and not ANTHROPIC_API_KEY.startswith("sk-ant-api03-YOUR")
    return {
        "status": "ok",
        "api_key_configured": key_ok,
        "model": "claude-sonnet-4-20250514",
    }


@app.get("/api/samples")
async def get_samples():
    """Return sample profiles for the frontend selector."""
    return {"samples": SAMPLE_PROFILES}


@app.post("/api/audit")
async def audit_profile(profile: ProfileInput):
    """
    Run the full 5-stage audit pipeline.
    The Anthropic API key is injected server-side — never sent to the client.
    """
    if not ANTHROPIC_API_KEY or ANTHROPIC_API_KEY.startswith("sk-ant-api03-YOUR"):
        raise HTTPException(
            status_code=503,
            detail="Anthropic API key not configured on the server. "
                   "Add your key to backend/.env and restart the server.",
        )

    if not profile.bio.strip():
        raise HTTPException(status_code=422, detail="Bio cannot be empty.")

    if len(profile.bio) > 5000:
        raise HTTPException(status_code=422, detail="Bio must be under 5000 characters.")

    try:
        result = await run_audit(profile)
        return result
    except Exception as e:
        error_msg = str(e)
        if "authentication" in error_msg.lower() or "api_key" in error_msg.lower():
            raise HTTPException(status_code=401, detail="Invalid Anthropic API key.")
        if "rate_limit" in error_msg.lower():
            raise HTTPException(status_code=429, detail="Anthropic rate limit hit. Try again shortly.")
        raise HTTPException(status_code=500, detail=f"Audit pipeline error: {error_msg}")


# ── DEV ENTRY POINT ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)

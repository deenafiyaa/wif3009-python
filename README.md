# RedFlagAI — Dating Profile Auditor

A full-stack AI-powered dating profile auditing tool with a Python FastAPI backend and a dark forensic-themed frontend.

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your OpenRouter API key
Copy `.env.example` into `.env`:
```
OPENROUTER_API_KEY=your-openrouter-api-key-here
```
Get a key at: https://openrouter.ai/openrouter/free

### 3. Start the server
```bash
uvicorn main:app --reload --port 8000
```

### 4. Open the app
Visit: http://localhost:8000

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Serves the frontend |
| POST | `/api/audit` | Runs the full 5-stage audit |
| GET | `/api/samples` | Returns sample profiles |
| GET | `/api/health` | Health check |

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENROUTER_API_KEY` | Yes | Your OpenRouter API key |
| `PORT` | No | Server port (default: 8000) |
| `ALLOWED_ORIGINS` | No | CORS origins (default: *) |

## Tech Stack

- **Backend**: Python 3.11+, FastAPI, Anthropic SDK, python-dotenv
- **Frontend**: Vanilla HTML/CSS/JS (no build step required)
- **AI**: Claude claude-sonnet-4 via Anthropic API (server-side, key never exposed)
- **Fonts**: Syne + IBM Plex Mono + Instrument Sans (Google Fonts)

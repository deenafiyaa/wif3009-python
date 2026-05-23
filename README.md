# RedFlagAI — Dating Profile Auditor

A full-stack AI-powered dating profile auditing tool with a Python FastAPI backend and a dark forensic-themed frontend.

## Project Structure

```
redflag/
├── backend/
│   ├── main.py          # FastAPI app — all routes
│   ├── audit.py         # 5-stage pipeline logic
│   ├── samples.py       # Sample profile data
│   ├── .env             # YOUR API KEY GOES HERE (never commit this)
│   └── requirements.txt
├── frontend/
│   ├── templates/
│   │   └── index.html   # Main HTML page (Jinja2)
│   └── static/
│       ├── style.css    # All styles
│       └── app.js       # All frontend JS
└── README.md
```

## Setup & Run

### 1. Install dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. Add your Anthropic API key
Edit `backend/.env`:
```
ANTHROPIC_API_KEY=sk-ant-api03-YOUR-KEY-HERE
```
Get a key at: https://console.anthropic.com

### 3. Start the server
```bash
cd backend
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
| `ANTHROPIC_API_KEY` | Yes | Your Anthropic API key |
| `PORT` | No | Server port (default: 8000) |
| `ALLOWED_ORIGINS` | No | CORS origins (default: *) |

## Tech Stack

- **Backend**: Python 3.11+, FastAPI, Anthropic SDK, python-dotenv
- **Frontend**: Vanilla HTML/CSS/JS (no build step required)
- **AI**: Claude claude-sonnet-4 via Anthropic API (server-side, key never exposed)
- **Fonts**: Syne + IBM Plex Mono + Instrument Sans (Google Fonts)

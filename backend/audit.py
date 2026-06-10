"""
audit.py — 5-stage ensemble pipeline for dating profile red flag detection.

Stage 1: Statistical screening      (rule-based SVM proxy on tabular fields)
Stage 2: Linguistic fingerprinting  (sentiment, complexity, manipulation score)
Stage 3: Contradiction engine       (stated categorical fields vs bio text)
Stage 4: Dark Triad scoring         (narcissism, machiavellianism, psychopathy)
Stage 5: LLM semantic reasoning     (OpenRouter OR Gemini — Chain-of-Thought)

OpenRouter uses the OpenAI-compatible API, so the same openai SDK works for
hundreds of models (Llama, Mistral, Gemini, GPT-4, Claude, DeepSeek, etc.)
just by changing the model name string in .env.
"""

import re
import os
import json
import base64
import tempfile
from typing import Optional
from openai import OpenAI
from pydantic import BaseModel

try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except Exception:
    _YOLO_AVAILABLE = False

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except Exception:
    _PIL_AVAILABLE = False


# ── PYDANTIC MODELS ────────────────────────────────────────────────────────────

class ProfileInput(BaseModel):
    bio: str
    smokes: str = "no"
    drinks: str = "socially"
    diet: str = "anything"
    drugs: str = "never"
    status: str = "single"
    age: int = 28
    picture: Optional[str] = None
    picture_filename: str = ""
    picture_size_kb: int = 0


class DarkTriad(BaseModel):
    narcissism: int
    machiavellianism: int
    psychopathy: int


class LinguisticFeatures(BaseModel):
    sentiment_score: float
    complexity: int
    manipulation_score: int


class Contradiction(BaseModel):
    stated: str
    detected: str
    severity: str   # high | medium | low


class Flag(BaseModel):
    category: str   # emotional | financial | identity | behavioral | veracity
    text: str
    severity: str   # high | medium | low


class AuditResult(BaseModel):
    overall_risk: int
    financial_risk: int
    emotional_risk: int
    identity_risk: int
    dark_triad: DarkTriad
    linguistic: LinguisticFeatures
    contradictions: list[Contradiction]
    flags: list[Flag]
    cot_reasoning: str
    educational_tip: str
    precision_estimate: float
    recall_estimate: float
    iou_estimate: float
    stage1_score: int
    stage1_passed: bool
    provider: str       # which AI provider was used (for transparency)
    model_used: str     # exact model name (for transparency)
    picture_b64: Optional[str] = None
    visual_flags: list[Flag] = []
    visual_ai_explanation: Optional[str] = None


# ── VISUAL SCORING HELPERS ──────────────────────────────────────────────────
VISUAL_FLAG_MAP = {
    'motorcycle': ('behavioral', 'Appears to be riding a motorcycle or superbike', 'high'),
    'bicycle': ('behavioral', 'Bicycle or cycling gear visible', 'low'),
    'bottle': ('behavioral', 'Alcohol or bottle visible (possible drinking)', 'medium'),
    'wine glass': ('behavioral', 'Holding a wine glass (possible drinking)', 'medium'),
    'cup': ('behavioral', 'Cup or drinking container visible', 'low'),
    'book': ('identity', 'Book or reading material visible (positive signal)', 'low'),
    'sports ball': ('identity', 'Sports equipment visible (positive signal)', 'low'),
    'dog': ('identity', 'Pet present (positive social signal)', 'low'),
    'cat': ('identity', 'Pet present (positive social signal)', 'low'),
    'cell phone': ('behavioral', 'Phone visible (neutral)', 'low'),
}


def explain_yolo_objects_with_ai(detected_objects: list[str], profile: ProfileInput) -> str:
    if not detected_objects:
        return "IMAGE INSIGHT // No clear objects were detected in the image."

    objects_text = ", ".join(detected_objects)

    # Convert profile details into text so AI can compare image with bio/profile
    try:
        profile_data = profile.model_dump()
    except:
        profile_data = profile.dict()

    profile_text = "\n".join(
        f"{key}: {value}" for key, value in profile_data.items() if value
    )

    prompt = f"""
    You are an AI assistant explaining objects detected in a dating profile photo.

Detected objects:
{objects_text}

Profile information:
{profile_text}

Output format MUST exactly follow this style:

IMAGE INSIGHT // Based on the detected objects:

* Person: Neutral signal (expected in a profile picture)
* Cup: Neutral signal (could be coffee, tea, or any drink; does not necessarily indicate excessive drinking)
* Donut: Potential red flag (may suggest indulgent or immature lifestyle if it contradicts the user's bio)

Rules:
- Start with: IMAGE INSIGHT // Based on the detected objects:
- Use one bullet point per detected object.
- Format each bullet as:
  * Object: Signal type (short explanation)
- Signal type must be one of:
  Neutral signal
  Potential red flag
  Positive signal
- Keep explanations short, factual, and non-judgmental.
- Do NOT infer gender, age, race, religion, or attractiveness.
- Do NOT write long paragraphs.
- Maximum 1 sentence per object.
"""

    # then send `prompt` to your AI model

    openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
    gemini_key = os.getenv("GEMINI_API_KEY", "")

    if openrouter_key and openrouter_key != "your-openrouter-api-key-here":
        client = OpenAI(
            api_key=openrouter_key,
            base_url="https://openrouter.ai/api/v1",
        )
        response = client.chat.completions.create(
            model=os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct"),
            messages=[
                {"role": "system", "content": "You explain YOLO image detections for dating profile safety analysis."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=400,
        )
        return response.choices[0].message.content or ""

    elif gemini_key and gemini_key != "your-gemini-api-key-here":
        client = OpenAI(
            api_key=gemini_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        response = client.chat.completions.create(
            model="gemini-1.5-flash",
            messages=[
                {"role": "system", "content": "You explain YOLO image detections for dating profile safety analysis."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=400,
        )
        return response.choices[0].message.content or ""

    return "No AI API key configured, so YOLO objects were detected but not explained."


# ── STAGE 1: STATISTICAL SCREENING ────────────────────────────────────────────

def stage1_statistical(profile: ProfileInput) -> tuple[int, bool]:
    """
    Proxy for an SVM / Logistic Regression trained on OkCupid tabular features.

    Production version: load backend/model/stage1_svm.pkl (see train_model.py).
    Current version: rule-based scoring that mirrors what a trained model learns
    from the same features. Replace the body of this function after running
    train_model.py — see the Setup Guide for exact replacement code.

    Returns:
        (risk_score 0-100, passed_clean bool)
    """
    import joblib
    import pandas as pd
    import os

    # Load your trained model
    model_path = os.path.join(
        os.path.dirname(__file__),
        "stage1_okcupid_model.joblib"
    )
    model = joblib.load(model_path)

    # Encode profile fields into numbers
    smokes_map = {
        "no": 0, "sometimes": 1,
        "yes": 2, "unknown": 3
    }
    drinks_map = {
        "rarely_never": 0, "socially": 1,
        "often": 2, "very often": 3,
        "very_often": 3, "unknown": 4
    }
    drugs_map = {
        "never": 0, "sometimes": 1,
        "often": 2, "unknown": 3
    }
    status_map = {
        "single": 0, "available": 1,
        "seeing someone": 2, "married": 3,
        "unknown": 4
    }

    # Build feature row
    features = pd.DataFrame([{
        "age":             profile.age,
        "smokes_enc":      smokes_map.get(profile.smokes, 3),
        "drinks_enc":      drinks_map.get(profile.drinks, 4),
        "drugs_enc":       drugs_map.get(profile.drugs, 3),
        "status_enc":      status_map.get(profile.status, 4),
        "income":          0,
        "sex_enc":         0,
        "orientation_enc": 0,
    }])

    # Get prediction and probability
    probability = model.predict_proba(features)[0][1]

    # Convert to 0-100 score
    score = int(probability * 100)

    return score, score < 40


# ── STAGE 2: LINGUISTIC FINGERPRINTING ────────────────────────────────────────

# Words/phrases that pattern-match manipulative or high-risk language.
# Expand this list using real OkCupid essay0 column data from your CSV.
MANIPULATION_PATTERNS = [
    r"\ball or nothing\b", r"\bno half measures\b",
    r"\bimmediately (deleted|blocked|gone)\b",
    r"\bpass my test\b", r"\byou (can't|cannot) lie to me\b",
    r"\byou feel like (my )?soulmate\b", r"\binstant connection\b",
    r"\belectric connection\b", r"\bmove mountains\b",
    r"\boperating at my level\b", r"\balpha\b",
    r"\bi attract\b", r"\bi don.t chase\b",
    r"\bweak personalities\b", r"\bemotional neediness\b",
    r"\bselective about who i give\b",
    r"\btext you good morning and goodnight\b",
    r"\btreated like royalt\b",
]

NEGATIVE_PATTERNS = [
    r"\bgames\b", r"\bdeleted\b", r"\bblocked\b",
    r"\bdrama\b", r"\btoxic\b", r"\bweak\b", r"\bneediness\b",
]

POSITIVE_PATTERNS = [
    r"\bhonest\b", r"\bcommunicat\b", r"\bkind\b",
    r"\brespect\b", r"\bbalanced\b", r"\bopen\b",
    r"\bfun\b", r"\blaugh\b", r"\bfriend\b",
    r"\bshare\b", r"\blearn\b",
]


def stage2_linguistic(bio: str) -> LinguisticFeatures:
    """
    Rule-based linguistic analysis.

    Production upgrade: replace with NLTK VADER for sentiment,
    textstat for Flesch-Kincaid complexity, and a fine-tuned
    HuggingFace classifier for manipulation detection.
    """
    bio_lower  = bio.lower()
    words      = bio_lower.split()
    word_count = max(len(words), 1)

    # Sentiment: ratio of positive to negative keyword hits
    pos = sum(1 for p in POSITIVE_PATTERNS if re.search(p, bio_lower))
    neg = sum(1 for p in NEGATIVE_PATTERNS if re.search(p, bio_lower))
    raw_sent  = (pos - neg) / max(pos + neg, 1)
    sentiment = max(-1.0, min(1.0, round(raw_sent * 0.8, 2)))

    # Complexity: average word length + sentence length
    avg_word_len     = sum(len(w) for w in words) / word_count
    sentence_count   = max(len(re.split(r'[.!?]+', bio)), 1)
    avg_sentence_len = word_count / sentence_count
    complexity       = int(min(100, (avg_word_len * 8) + (avg_sentence_len * 0.5)))

    # Manipulation: keyword hits + first-person singular overuse
    manip_hits        = sum(1 for p in MANIPULATION_PATTERNS if re.search(p, bio_lower))
    first_person      = len(re.findall(r'\b(i|me|my|mine|myself)\b', bio_lower))
    first_person_rate = first_person / word_count
    manipulation      = int(min(100, manip_hits * 18 + first_person_rate * 60))

    return LinguisticFeatures(
        sentiment_score=sentiment,
        complexity=complexity,
        manipulation_score=manipulation,
    )


# ── STAGE 3: CONTRADICTION ENGINE ─────────────────────────────────────────────

# Each rule compares a stated categorical value against patterns in the bio text.
# Add more rules here as you discover patterns in the OkCupid CSV.
CONTRADICTION_RULES = [
    {
        "field": "smokes", "values": ["no"],
        "patterns": [r"\bcigar\b", r"\bsmoking\b", r"\bcigarette\b", r"\bsmoke\b", r"\bvap(e|ing)\b"],
        "labels": {"no": "smokes: no"}, "severity": "high",
    },
    {
        "field": "drugs", "values": ["never"],
        "patterns": [r"\b(weed|marijuana|cannabis|420|high|stoned|molly|mdma|cocaine|coke)\b", r"\bget high\b"],
        "labels": {"never": "drugs: never"}, "severity": "high",
    },
    {
        "field": "drinks", "values": ["never", "not at all"],
        "patterns": [r"\b(beer|wine|whiskey|whisky|cocktail|drunk|drinking|bar|pub|booze|vodka)\b"],
        "labels": {"never": "drinks: never", "not at all": "drinks: not at all"}, "severity": "medium",
    },
    {
        "field": "diet", "values": ["vegan", "strictly vegan", "mostly vegan"],
        "patterns": [r"\b(meat|chicken|burger|steak|bacon|fish|sushi|seafood|beef|pork)\b"],
        "labels": {
            "vegan": "diet: vegan",
            "strictly vegan": "diet: strictly vegan",
            "mostly vegan": "diet: mostly vegan",
        },
        "severity": "medium",
    },
    {
        "field": "status", "values": ["single"],
        "patterns": [r"\b(girlfriend|boyfriend|partner|wife|husband|spouse|fiancee|fianc[eé])\b"],
        "labels": {"single": "status: single"}, "severity": "high",
    },
]


def stage3_contradictions(profile: ProfileInput) -> list[Contradiction]:
    """
    Cross-reference stated categorical fields against bio text.
    Returns a list of detected contradictions with bio context excerpts.
    """
    bio_lower    = profile.bio.lower()
    profile_dict = {
        "smokes": profile.smokes,
        "drinks": profile.drinks,
        "diet":   profile.diet,
        "drugs":  profile.drugs,
        "status": profile.status,
    }
    found = []
    for rule in CONTRADICTION_RULES:
        field_val = profile_dict.get(rule["field"], "")
        if field_val not in rule["values"]:
            continue
        for pattern in rule["patterns"]:
            match = re.search(pattern, bio_lower)
            if match:
                # Extract up to 40 characters either side for context
                start   = max(0, match.start() - 40)
                end     = min(len(profile.bio), match.end() + 40)
                excerpt = "…" + profile.bio[start:end].strip() + "…"
                found.append(Contradiction(
                    stated=rule["labels"].get(field_val, f"{rule['field']}: {field_val}"),
                    detected=excerpt,
                    severity=rule["severity"],
                ))
                break  # one contradiction per rule is enough
    return found


# ── STAGE 4: DARK TRIAD ────────────────────────────────────────────────────────

# Keyword patterns for each Dark Triad trait.
# These are calibrated against real OkCupid essay text.
NARCISSISM_PATTERNS = [
    r"\balpha\b", r"\boperating at my level\b", r"\bi attract\b",
    r"\bi don.t chase\b", r"\bintimidating\b", r"\bselective about\b",
    r"\bmy time is (my )?most valuable\b", r"\bmost people (aren.t|can.t)\b",
]
MACH_PATTERNS = [
    r"\bpass my test\b", r"\bi know every game\b", r"\byou cannot lie to me\b",
    r"\bcan read people\b", r"\bi know what i want and.*how to get it\b",
    r"\bcharming when i want\b", r"\bdifferent side of me\b",
]
PSYCHOPATHY_PATTERNS = [
    r"\bimmediately (deleted|blocked)\b", r"\bweak personalities\b",
    r"\bemotional neediness\b", r"\bnot here to make friends\b",
    r"\bi prefer those who can keep up\b", r"\bdon.t have time for\b",
]


def stage4_dark_triad(bio: str) -> DarkTriad:
    """
    Score each Dark Triad trait based on keyword pattern hits.
    Scores are scaled so that 3+ hits pushes toward 70+.
    """
    bio_lower = bio.lower()
    narc = sum(1 for p in NARCISSISM_PATTERNS  if re.search(p, bio_lower))
    mach = sum(1 for p in MACH_PATTERNS         if re.search(p, bio_lower))
    psyc = sum(1 for p in PSYCHOPATHY_PATTERNS  if re.search(p, bio_lower))
    return DarkTriad(
        narcissism=min(100, int(narc * 22 + 5)),
        machiavellianism=min(100, int(mach * 25 + 5)),
        psychopathy=min(100, int(psyc * 28 + 5)),
    )


# ── STAGE 5: LLM CHAIN-OF-THOUGHT (OpenRouter / Gemini) ───────────────────────

SYSTEM_PROMPT = """You are an expert dating profile auditing AI specialising in forensic linguistics, behavioural psychology, and deceptive communication detection.

You receive a structured pre-processing summary from a 4-stage pipeline and the raw profile. Your role is Stage 5: semantic Chain-of-Thought reasoning that refines and enriches the pipeline scores.

Respond ONLY with a valid JSON object. No markdown, no backticks, no preamble, no trailing text.

Required schema (all fields required):
{
  "overall_risk": <integer 0-100>,
  "financial_risk": <integer 0-100>,
  "emotional_risk": <integer 0-100>,
  "identity_risk": <integer 0-100>,
  "dark_triad": {
    "narcissism": <integer 0-100>,
    "machiavellianism": <integer 0-100>,
    "psychopathy": <integer 0-100>
  },
  "linguistic": {
    "sentiment_score": <float -1.0 to 1.0>,
    "complexity": <integer 0-100>,
    "manipulation_score": <integer 0-100>
  },
  "contradictions": [
    { "stated": "<field label>", "detected": "<bio excerpt>", "severity": "high|medium|low" }
  ],
  "flags": [
    { "category": "emotional|financial|identity|behavioral|veracity", "text": "<specific flag>", "severity": "high|medium|low" }
  ],
  "cot_reasoning": "<3-4 sentences citing specific phrases from the bio. Explain the psychological mechanism analytically.>",
  "educational_tip": "<2-3 sentences of constructive, non-shaming commentary written directly to the profile author.>",
  "precision_estimate": <float 0.0-1.0>,
  "recall_estimate": <float 0.0-1.0>,
  "iou_estimate": <float 0.0-1.0>
}

Scoring guide: overall_risk 5-20=healthy, 20-45=mild concerns, 45-70=significant flags, 70-100=severe."""


def _build_user_message(
    profile: ProfileInput,
    s1_score: int,
    s1_passed: bool,
    linguistic: LinguisticFeatures,
    contradictions: list[Contradiction],
    dark_triad: DarkTriad,
) -> str:
    contra_text = "\n".join(
        f"  - Stated '{c.stated}' but bio contains: \"{c.detected}\" [severity: {c.severity}]"
        for c in contradictions
    ) or "  None detected by rules engine."

    return f"""PIPELINE PRE-PROCESSING SUMMARY
================================
Stage 1 — Statistical screening:
  Risk score: {s1_score}/100
  Passed clean: {s1_passed}

Stage 2 — Linguistic fingerprinting:
  Sentiment: {linguistic.sentiment_score:+.2f}  (-1=very negative, +1=very positive)
  Complexity: {linguistic.complexity}/100
  Manipulation score: {linguistic.manipulation_score}/100

Stage 3 — Contradiction engine:
{contra_text}

Stage 4 — Dark Triad pre-score:
  Narcissism: {dark_triad.narcissism}/100
  Machiavellianism: {dark_triad.machiavellianism}/100
  Psychopathy: {dark_triad.psychopathy}/100

RAW PROFILE DATA
================
Bio: "{profile.bio}"

Categorical fields:
  smokes: {profile.smokes}
  drinks: {profile.drinks}
  diet:   {profile.diet}
  drugs:  {profile.drugs}
  status: {profile.status}
  age:    {profile.age}
  picture_filename: {profile.picture_filename or 'none'}
  picture_size_kb: {profile.picture_size_kb}

TASK: Perform Stage 5 Chain-of-Thought semantic audit. Refine all scores using your full linguistic understanding. Cite specific bio phrases in cot_reasoning."""


def _call_openrouter(user_msg: str) -> tuple[dict, str]:
    """
    Call OpenRouter using the OpenAI-compatible API.
    OpenRouter gives access to 300+ models with one key.
    The base_url swap is the only difference from calling OpenAI directly.
    """
    api_key    = os.getenv("OPENROUTER_API_KEY", "")
    model_name = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct")

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",       # ← the only OpenRouter-specific line
        default_headers={
            "HTTP-Referer": "https://redflagai.local", # optional — shown in OpenRouter dashboard
            "X-Title":      "RedFlagAI",               # optional — shown in OpenRouter dashboard
        },
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system",  "content": SYSTEM_PROMPT},
            {"role": "user",    "content": user_msg},
        ],
        temperature=0.3,   # lower = more deterministic JSON output
        max_tokens=1500,
    )

    raw   = response.choices[0].message.content or ""
    clean = re.sub(r"```json|```", "", raw).strip()
    return json.loads(clean), model_name


def _call_gemini(user_msg: str) -> tuple[dict, str]:
    """
    Fallback: call Google Gemini if GEMINI_API_KEY is set instead.
    Uses the same openai SDK pointed at Gemini's OpenAI-compatible endpoint.
    """
    api_key = os.getenv("GEMINI_API_KEY", "")

    client = OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    model_name = "gemini-1.5-flash"
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.3,
        max_tokens=1500,
    )

    raw   = response.choices[0].message.content or ""
    clean = re.sub(r"```json|```", "", raw).strip()
    return json.loads(clean), model_name


# ── MAIN AUDIT ORCHESTRATOR ────────────────────────────────────────────────────

async def run_audit(profile: ProfileInput) -> AuditResult:
    """
    Orchestrates the full 5-stage pipeline.

    Stages 1–4 run locally in Python (fast, no API cost).
    YOLO detects image objects when a picture is supplied.
    Stage 5 calls the configured LLM provider (OpenRouter or Gemini).
    """

    # ── Local stages (no network calls) ───────────────────────────
    s1_score, s1_passed = stage1_statistical(profile)
    linguistic          = stage2_linguistic(profile.bio)
    contradictions      = stage3_contradictions(profile)
    dark_triad          = stage4_dark_triad(profile.bio)

    # ── YOLO visual detection ───────────────────────────────────
    visual_flags = []
    detected_objects = []
    visual_ai_explanation = ""

    if profile.picture and _YOLO_AVAILABLE:
        tmp_path = None
        try:
            header, b64 = profile.picture.split(",", 1) if "," in profile.picture else ("", profile.picture)
            img_bytes = base64.b64decode(b64)

            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
                tf.write(img_bytes)
                tmp_path = tf.name

            model_path = os.getenv("YOLO_MODEL", "yolov8n.pt")
            model = YOLO(model_path)
            results = model.predict(source=tmp_path, verbose=False)
            det = results[0]
            names = model.names if hasattr(model, "names") else {}

            if hasattr(det, "boxes") and len(det.boxes) > 0:
                for box in det.boxes:
                    try:
                        cls = int(box.cls.cpu().numpy()[0])
                    except Exception:
                        continue
                    name = names.get(cls, str(cls)).lower()
                    if isinstance(name, bytes):
                        name = name.decode()
                    name_key = name.replace("_", " ")
                    detected_objects.append(name_key)
                    if name_key in VISUAL_FLAG_MAP:
                        cat, txt, sev = VISUAL_FLAG_MAP[name_key]
                        visual_flags.append(Flag(category=cat, text=txt, severity=sev))

            detected_objects = list(dict.fromkeys(detected_objects))
            if detected_objects:
                visual_ai_explanation = explain_yolo_objects_with_ai(detected_objects, profile)

        except Exception as e:
            print("YOLO detection failed:", e)
            visual_flags = []
            detected_objects = []
            visual_ai_explanation = ""

        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    # ── Stage 5: LLM provider selection ───────────────────────────
    user_msg = _build_user_message(
        profile, s1_score, s1_passed, linguistic, contradictions, dark_triad
    )

    if detected_objects:
        user_msg += "\n\nYOLO DETECTED OBJECTS:\n"
        user_msg += ", ".join(detected_objects)

    if visual_flags:
        user_msg += "\n\nVISUAL FLAGS FROM YOLO MAP:\n"
        user_msg += "\n".join(
            f"- {v.text} [category: {v.category}, severity: {v.severity}]"
            for v in visual_flags
        )

    if visual_ai_explanation:
        user_msg += "\n\nAI VISUAL EXPLANATION:\n"
        user_msg += visual_ai_explanation

    openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
    gemini_key     = os.getenv("GEMINI_API_KEY", "")

    if openrouter_key and openrouter_key != "your-openrouter-api-key-here":
        data, model_used = _call_openrouter(user_msg)
        provider_used    = "openrouter"
    elif gemini_key and gemini_key != "your-gemini-api-key-here":
        data, model_used = _call_gemini(user_msg)
        provider_used    = "gemini"
    else:
        raise RuntimeError("No AI API key configured. Set OPENROUTER_API_KEY in backend/.env")

    # ── Assemble final result ─────────────────────────────────────
    return AuditResult(
        overall_risk=data.get("overall_risk", 50),
        financial_risk=data.get("financial_risk", 20),
        emotional_risk=data.get("emotional_risk", 30),
        identity_risk=data.get("identity_risk", 20),
        dark_triad=DarkTriad(**data.get("dark_triad", {
            "narcissism":       dark_triad.narcissism,
            "machiavellianism": dark_triad.machiavellianism,
            "psychopathy":      dark_triad.psychopathy,
        })),
        linguistic=LinguisticFeatures(**data.get("linguistic", {
            "sentiment_score":    linguistic.sentiment_score,
            "complexity":         linguistic.complexity,
            "manipulation_score": linguistic.manipulation_score,
        })),
        contradictions=[Contradiction(**c) for c in data.get("contradictions", [])] or contradictions,
        flags=[Flag(**f) for f in data.get("flags", [])],
        visual_flags=visual_flags,
        cot_reasoning=data.get("cot_reasoning", ""),
        educational_tip=data.get("educational_tip", ""),
        precision_estimate=data.get("precision_estimate", 0.80),
        recall_estimate=data.get("recall_estimate", 0.72),
        iou_estimate=data.get("iou_estimate", 0.61),
        stage1_score=s1_score,
        stage1_passed=s1_passed,
        provider=provider_used,
        model_used=model_used,
        picture_b64=profile.picture,
        visual_ai_explanation=visual_ai_explanation,
    )

"""
audit.py — 5-stage ensemble pipeline for dating profile red flag detection.

Stage 1: Statistical screening   (rule-based SVM proxy on tabular fields)
Stage 2: Linguistic fingerprinting (sentiment, complexity, manipulation)
Stage 3: Contradiction engine    (stated fields vs bio text)
Stage 4: Dark Triad scoring      (narcissism, machiavellianism, psychopathy)
Stage 5: LLM semantic reasoning  (Claude CoT audit)
"""

import re
import os
import anthropic
from pydantic import BaseModel
from typing import Optional


# ── PYDANTIC MODELS ────────────────────────────────────────────────────────────

class ProfileInput(BaseModel):
    bio: str
    smokes: str = "no"
    drinks: str = "socially"
    diet: str = "anything"
    drugs: str = "never"
    status: str = "single"
    age: int = 28


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
    severity: str  # high | medium | low


class Flag(BaseModel):
    category: str  # emotional | financial | identity | behavioral | veracity
    text: str
    severity: str  # high | medium | low


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
    stage1_score: int       # from statistical screening
    stage1_passed: bool     # whether profile passed Stage 1 clean


# ── STAGE 1: STATISTICAL SCREENING ────────────────────────────────────────────

def stage1_statistical(profile: ProfileInput) -> tuple[int, bool]:
    """
    Proxy for SVM/LR trained on OkCupid tabular features.
    In production: load a pickled sklearn model and call .predict_proba().
    Here: rule-based scoring that mirrors what a trained model would learn.
    Returns (risk_score 0-100, passed_clean bool)
    """
    score = 0

    # Drug use signals
    if profile.drugs == "often":
        score += 25
    elif profile.drugs == "sometimes":
        score += 12

    # Drinking pattern
    if profile.drinks == "often":
        score += 15
    elif profile.drinks == "desperately":
        score += 20

    # Smoking contradiction base
    if profile.smokes == "yes":
        score += 10
    elif profile.smokes == "sometimes":
        score += 5

    # Age-based risk (very young or very old with "available" status)
    if profile.age < 22 and profile.status in ("available", "seeing someone"):
        score += 10
    if profile.age > 50 and profile.status == "available":
        score += 8

    # Status signals
    if profile.status == "seeing someone":
        score += 20

    # Diet contradiction potential
    if profile.diet in ("strictly vegan", "strictly anything"):
        score += 3  # minor — strict diets sometimes correlate with rigidity

    score = min(score, 100)
    passed = score < 40
    return score, passed


# ── STAGE 2: LINGUISTIC FINGERPRINTING ────────────────────────────────────────

# Manipulation vocabulary — patterns found in fraudulent/manipulative profiles
MANIPULATION_PATTERNS = [
    r"\ball or nothing\b", r"\bno half measures\b",
    r"\bimmediately (deleted|blocked|gone)\b",
    r"\bpass my test\b", r"\byou (can't|cannot) lie to me\b",
    r"\byou feel like (my )?soulmate\b", r"\binstant connection\b",
    r"\belectric connection\b", r"\bmove mountains\b",
    r"\boperating at my level\b", r"\balpha\b",
    r"\bi attract\b", r"\bi don.t chase\b",
    r"\bweak personalities\b", r"\bemotional neediness\b",
    r"\bintimidating\b.*\blevel\b",
    r"\bselective about who i give\b",
    r"\btext you good morning and goodnight\b",
    r"\btreated like royalt\b",
]

NEGATIVE_PATTERNS = [
    r"\bgames\b", r"\bdeleted\b", r"\bblocked\b",
    r"\bdrama\b", r"\btoxic\b", r"\bweak\b",
    r"\bneediness\b", r"\btest\b.*\bpass\b",
]

POSITIVE_PATTERNS = [
    r"\bhonest\b", r"\bcommunicat\b", r"\bkind\b",
    r"\brespect\b", r"\bbalanced\b", r"\bopen\b",
    r"\bfun\b", r"\blaugh\b", r"\bfriend\b",
    r"\bshare\b", r"\blearn\b",
]


def stage2_linguistic(bio: str) -> LinguisticFeatures:
    """
    Rule-based linguistic analysis proxy.
    In production: use NLTK VADER + textstat + HuggingFace classifiers.
    """
    bio_lower = bio.lower()
    words = bio_lower.split()
    word_count = max(len(words), 1)

    # Sentiment: positive - negative keyword ratio
    pos = sum(1 for p in POSITIVE_PATTERNS if re.search(p, bio_lower))
    neg = sum(1 for p in NEGATIVE_PATTERNS if re.search(p, bio_lower))
    raw_sent = (pos - neg) / max(pos + neg, 1)
    # Scale to -1..1 with centre bias
    sentiment = max(-1.0, min(1.0, round(raw_sent * 0.8, 2)))

    # Complexity: avg word length + sentence variety proxy
    avg_word_len = sum(len(w) for w in words) / word_count
    sentence_count = max(len(re.split(r'[.!?]+', bio)), 1)
    avg_sentence_len = word_count / sentence_count
    complexity = int(min(100, (avg_word_len * 8) + (avg_sentence_len * 0.5)))

    # Manipulation score
    manip_hits = sum(1 for p in MANIPULATION_PATTERNS if re.search(p, bio_lower))
    # First-person singular rate
    first_person = len(re.findall(r'\b(i|me|my|mine|myself)\b', bio_lower))
    first_person_rate = first_person / word_count
    manipulation = int(min(100, manip_hits * 18 + first_person_rate * 60))

    return LinguisticFeatures(
        sentiment_score=sentiment,
        complexity=complexity,
        manipulation_score=manipulation,
    )


# ── STAGE 3: CONTRADICTION ENGINE ─────────────────────────────────────────────

CONTRADICTION_RULES = [
    {
        "field": "smokes",
        "values": ["no"],
        "patterns": [
            r"\bcigar\b", r"\bsmoking\b", r"\bcigarette\b",
            r"\bsmoke\b", r"\bvaping\b", r"\bvape\b",
        ],
        "labels": {
            "no": "smokes: no",
        },
        "severity": "high",
    },
    {
        "field": "drugs",
        "values": ["never"],
        "patterns": [
            r"\b(weed|marijuana|cannabis|420|high|stoned|molly|mdma|cocaine|coke)\b",
            r"\bget high\b", r"\bsmoke up\b",
        ],
        "labels": {"never": "drugs: never"},
        "severity": "high",
    },
    {
        "field": "drinks",
        "values": ["never"],
        "patterns": [
            r"\b(beer|wine|whiskey|whisky|cocktail|drunk|drinking|bar|pub|booze|vodka)\b",
        ],
        "labels": {"never": "drinks: never"},
        "severity": "medium",
    },
    {
        "field": "diet",
        "values": ["vegan", "strictly vegan"],
        "patterns": [
            r"\b(meat|chicken|burger|steak|bacon|fish|sushi|seafood|beef|pork)\b",
        ],
        "labels": {
            "vegan": "diet: vegan",
            "strictly vegan": "diet: strictly vegan",
        },
        "severity": "medium",
    },
    {
        "field": "status",
        "values": ["single"],
        "patterns": [
            r"\b(girlfriend|boyfriend|partner|wife|husband|spouse|fiancee|fiancé)\b",
        ],
        "labels": {"single": "status: single"},
        "severity": "high",
    },
]


def stage3_contradictions(profile: ProfileInput) -> list[Contradiction]:
    bio_lower = profile.bio.lower()
    found = []
    profile_dict = {
        "smokes": profile.smokes,
        "drinks": profile.drinks,
        "diet": profile.diet,
        "drugs": profile.drugs,
        "status": profile.status,
    }

    for rule in CONTRADICTION_RULES:
        field_val = profile_dict.get(rule["field"], "")
        if field_val not in rule["values"]:
            continue
        for pattern in rule["patterns"]:
            match = re.search(pattern, bio_lower)
            if match:
                # Extract surrounding context (up to 8 words each side)
                start = max(0, match.start() - 40)
                end = min(len(profile.bio), match.end() + 40)
                excerpt = "…" + profile.bio[start:end].strip() + "…"
                found.append(Contradiction(
                    stated=rule["labels"].get(field_val, f"{rule['field']}: {field_val}"),
                    detected=excerpt,
                    severity=rule["severity"],
                ))
                break  # one contradiction per rule

    return found


# ── STAGE 4: DARK TRIAD ────────────────────────────────────────────────────────

NARCISSISM_PATTERNS = [
    r"\balpha\b", r"\boperating at my level\b",
    r"\bi attract\b", r"\bi don.t chase\b",
    r"\bintimidating\b", r"\bselective about\b",
    r"\bmy time is (my )?most valuable\b",
    r"\bnot everyone (can|could)\b",
    r"\bmost people (aren.t|can.t)\b",
]

MACH_PATTERNS = [
    r"\bpass my test\b", r"\bi know every game\b",
    r"\byou cannot lie to me\b", r"\bcan read people\b",
    r"\bi know what i want and.*how to get it\b",
    r"\bcharming when i want\b", r"\bif you pass\b",
    r"\bdifferent side of me\b",
]

PSYCHOPATHY_PATTERNS = [
    r"\bimmediately (deleted|blocked)\b",
    r"\bweak personalities\b", r"\bemotional neediness\b",
    r"\bnot here to make friends\b",
    r"\bi prefer those who can keep up\b",
    r"\bno time for\b", r"\bdon.t have time for\b",
]


def stage4_dark_triad(bio: str) -> DarkTriad:
    bio_lower = bio.lower()

    narc = sum(1 for p in NARCISSISM_PATTERNS if re.search(p, bio_lower))
    mach = sum(1 for p in MACH_PATTERNS if re.search(p, bio_lower))
    psyc = sum(1 for p in PSYCHOPATHY_PATTERNS if re.search(p, bio_lower))

    # Scale hits to 0-100 (cap at 5 hits = 100)
    narcissism = min(100, int(narc * 22 + 5))
    machiavellianism = min(100, int(mach * 25 + 5))
    psychopathy = min(100, int(psyc * 28 + 5))

    return DarkTriad(
        narcissism=narcissism,
        machiavellianism=machiavellianism,
        psychopathy=psychopathy,
    )


# ── STAGE 5: LLM CHAIN-OF-THOUGHT ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert dating profile auditing AI with specialisation in forensic linguistics, behavioural psychology, and deceptive communication detection.

You receive a structured summary from a 4-stage pre-processing pipeline, plus the raw profile. Your job is Stage 5: semantic Chain-of-Thought reasoning.

Respond ONLY with a valid JSON object — absolutely no markdown, no backticks, no preamble, no trailing text.

Required schema:
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
  "cot_reasoning": "<3-4 sentences citing specific phrases from the bio. Explain the psychological mechanism. Be analytical.>",
  "educational_tip": "<2-3 sentences of constructive, non-shaming commentary written to the profile author.>",
  "precision_estimate": <float 0.0-1.0>,
  "recall_estimate": <float 0.0-1.0>,
  "iou_estimate": <float 0.0-1.0>
}

Scoring rules:
- overall_risk: 5-20 = healthy, 20-45 = mild concerns, 45-70 = significant flags, 70-100 = severe
- financial_risk: signals of financial manipulation or instability exploitation
- emotional_risk: love-bombing, emotional coercion, co-dependency, intermittent reinforcement
- identity_risk: deception, inconsistency, catfishing patterns, identity dissociation
- Refine the dark_triad and linguistic scores the pipeline gave you — use your semantic understanding
- contradictions: incorporate pipeline findings and add any new ones you detect
- flags: be specific, cite bio text, cover all dimensions
- precision/recall/iou: your calibrated estimate for a profile with this signal clarity"""


def build_user_message(
    profile: ProfileInput,
    stage1_score: int,
    stage1_passed: bool,
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
  Risk score: {stage1_score}/100
  Passed clean: {stage1_passed}

Stage 2 — Linguistic fingerprinting:
  Sentiment: {linguistic.sentiment_score:+.2f} (-1=very negative, +1=very positive)
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
  diet: {profile.diet}
  drugs: {profile.drugs}
  status: {profile.status}
  age: {profile.age}

TASK: Perform Stage 5 Chain-of-Thought semantic audit. Refine all scores using your full linguistic understanding. Cite specific bio phrases in your reasoning."""


# ── MAIN AUDIT ORCHESTRATOR ────────────────────────────────────────────────────

async def run_audit(profile: ProfileInput) -> AuditResult:
    """
    Orchestrates the full 5-stage pipeline.
    """
    # Stage 1
    s1_score, s1_passed = stage1_statistical(profile)

    # Stage 2
    linguistic = stage2_linguistic(profile.bio)

    # Stage 3
    contradictions = stage3_contradictions(profile)

    # Stage 4
    dark_triad = stage4_dark_triad(profile.bio)

    # Stage 5: Claude LLM
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    client = anthropic.Anthropic(api_key=api_key)

    user_msg = build_user_message(
        profile, s1_score, s1_passed, linguistic, contradictions, dark_triad
    )

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )

    raw = "".join(block.text for block in message.content if hasattr(block, "text"))
    # Strip any accidental markdown fences
    clean = re.sub(r"```json|```", "", raw).strip()

    import json
    data = json.loads(clean)

    # Merge LLM output with pipeline data (LLM is authoritative on scores,
    # but we always include Stage 1-4 signal for transparency)
    result = AuditResult(
        overall_risk=data.get("overall_risk", 50),
        financial_risk=data.get("financial_risk", 20),
        emotional_risk=data.get("emotional_risk", 30),
        identity_risk=data.get("identity_risk", 20),
        dark_triad=DarkTriad(**data.get("dark_triad", {
            "narcissism": dark_triad.narcissism,
            "machiavellianism": dark_triad.machiavellianism,
            "psychopathy": dark_triad.psychopathy,
        })),
        linguistic=LinguisticFeatures(**data.get("linguistic", {
            "sentiment_score": linguistic.sentiment_score,
            "complexity": linguistic.complexity,
            "manipulation_score": linguistic.manipulation_score,
        })),
        contradictions=[
            Contradiction(**c) for c in data.get("contradictions", [])
        ] or contradictions,
        flags=[Flag(**f) for f in data.get("flags", [])],
        cot_reasoning=data.get("cot_reasoning", ""),
        educational_tip=data.get("educational_tip", ""),
        precision_estimate=data.get("precision_estimate", 0.80),
        recall_estimate=data.get("recall_estimate", 0.72),
        iou_estimate=data.get("iou_estimate", 0.61),
        stage1_score=s1_score,
        stage1_passed=s1_passed,
    )

    return result

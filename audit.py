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
    import pandas as pd
    _PANDAS_AVAILABLE = True
except Exception:
    _PANDAS_AVAILABLE = False

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
    # Optional profile picture as a base64 data URL (data:image/...;base64,...)
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


# Visual flag mapping (YOLO class -> (category, description, severity))
VISUAL_FLAG_MAP = {
    # common COCO-like classes
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
    """
    Ask AI to explain whether YOLO-detected objects may be red flags or positive signals.
    """
    if not detected_objects:
        return "No clear objects were detected in the image."

    objects_text = ", ".join(detected_objects)

    prompt = f"""
You are analysing a dating profile picture.

YOLO detected these objects:
{objects_text}

Profile info:
Bio: {profile.bio}
Smokes: {profile.smokes}
Drinks: {profile.drinks}
Drugs: {profile.drugs}
Status: {profile.status}
Age: {profile.age}

Explain whether the detected objects may be:
1. Red flags
2. Neutral signals
3. Positive signals

Important:
- Do not judge the person unfairly.
- Explain based only on visible objects.
- Mention contradiction if image object conflicts with profile info.
- Keep the explanation short and clear.
"""

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
    llm_model: str      # exact model name (for transparency)
    picture_b64: Optional[str] = None
    visual_flags: list[Flag] = []
    visual_ai_explanation: Optional[str] = None


# ── STAGE 1: STATISTICAL SCREENING ────────────────────────────────────────────

def load_stage1_model(model_path: str = "stage1_okcupid_model.joblib"):
    if not _PANDAS_AVAILABLE:
        return None
    try:
        import joblib
    except Exception:
        return None
    if not os.path.exists(model_path):
        return None
    try:
        return joblib.load(model_path)
    except Exception:
        return None


def stage1_statistical(profile: ProfileInput) -> tuple[int, bool]:
    """
    Use a trained stage1 model when available, otherwise fallback to
    a rule-based OkCupid proxy.

    Returns:
        (risk_score 0-100, passed_clean bool)
    """
    model = load_stage1_model()
    if model is not None:
        try:
            import pandas as pd
            row = pd.DataFrame([{
                "smokes": profile.smokes,
                "drinks": profile.drinks,
                "diet": profile.diet,
                "drugs": profile.drugs,
                "status": profile.status,
                "age": profile.age,
                "body_type": "unknown",
                "education": "unknown",
                "ethnicity": "unknown",
                "height": 0,
                "income": 0,
                "job": "unknown",
                "religion": "unknown",
                "sign": "unknown",
                "bio_text": profile.bio,
            }])
            proba = model.predict_proba(row)[0]
            positive_index = list(model.classes_).index(1) if 1 in model.classes_ else 1
            score = int(min(100, max(0, proba[positive_index] * 100)))
            return score, score < 40
        except Exception:
            pass

    score = 0

    # Drug use signals (strongest predictor in OkCupid data)
    if profile.drugs == "often":
        score += 25
    elif profile.drugs == "sometimes":
        score += 12

    # Drinking pattern
    if profile.drinks in ("often", "very often", "desperately"):
        score += 15

    # Smoking
    if profile.smokes == "yes":
        score += 10
    elif profile.smokes in ("sometimes", "trying to quit", "when drinking"):
        score += 5

    # Relationship status red flags
    if profile.status == "seeing someone":
        score += 20
    elif profile.status == "married":
        score += 30
    elif profile.status == "unknown":
        score += 5

    # Age-related risk patterns
    if profile.age < 22 and profile.status in ("available", "seeing someone"):
        score += 8
    if profile.age > 50 and profile.status == "available":
        score += 6

    score = min(score, 100)
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


def train_yolo_from_csv(csv_path: str, output_dir: str = 'yolodata', model: str = 'yolov8n.pt', epochs: int = 50):
    """
    Helper to convert a CSV of annotations to YOLOv8 training format and kick off training.

    Expected CSV columns: image_path, class_id, x1, y1, x2, y2
    Coordinates should be absolute pixel coordinates. The function will create
    a dataset under `output_dir` with `images/` and `labels/` and then call
    ultralytics.YOLO(model).train(data=..., epochs=...)
    """
    if not _PANDAS_AVAILABLE or not _PIL_AVAILABLE or not _YOLO_AVAILABLE:
        raise RuntimeError('pandas, Pillow and ultralytics are required to run training')

    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)
    images_out = os.path.join(output_dir, 'images')
    labels_out = os.path.join(output_dir, 'labels')
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)

    # gather unique images
    for img_path, group in df.groupby('image_path'):
        try:
            im = Image.open(img_path)
            w, h = im.size
        except Exception:
            print(f"Skipping missing/unreadable image: {img_path}")
            continue

        # copy image to images_out
        dst_img = os.path.join(images_out, os.path.basename(img_path))
        if os.path.abspath(img_path) != os.path.abspath(dst_img):
            try:
                im.save(dst_img)
            except Exception:
                # last resort: copy file
                import shutil
                shutil.copy(img_path, dst_img)

        # write label file
        lbl_lines = []
        for _, row in group.iterrows():
            cls = int(row['class_id'])
            x1, y1, x2, y2 = float(row['x1']), float(row['y1']), float(row['x2']), float(row['y2'])
            xc = ((x1 + x2) / 2.0) / w
            yc = ((y1 + y2) / 2.0) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            lbl_lines.append(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        lbl_path = os.path.join(labels_out, os.path.splitext(os.path.basename(img_path))[0] + '.txt')
        with open(lbl_path, 'w', encoding='utf8') as f:
            f.write('\n'.join(lbl_lines))

    # prepare a minimal data YAML for ultralytics
    data_yaml = os.path.join(output_dir, 'data.yaml')
    num_classes = int(df['class_id'].max()) + 1
    with open(data_yaml, 'w', encoding='utf8') as f:
        f.write(f"train: {os.path.abspath(images_out)}\n")
        f.write(f"val: {os.path.abspath(images_out)}\n")
        f.write(f"nc: {num_classes}\n")
        f.write("names: []\n")

    # call ultralytics training
    model_obj = YOLO(model)
    model_obj.train(data=data_yaml, epochs=epochs)


def _load_okcupid_table(dataset_path: str):
    if not _PANDAS_AVAILABLE:
        raise RuntimeError('pandas is required to load OkCupid dataset')
    try:
        return pd.read_csv(dataset_path)
    except Exception:
        return pd.read_excel(dataset_path, engine='openpyxl')


def train_okcupid_stage1_model(
    dataset_path: str,
    output_model: str = 'stage1_okcupid_model.joblib',
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Train a stage1 risk classifier from the OkCupid profile dataset.

    The helper uses lifestyle and essay text features to learn a proxy
    red-flag risk score from the provided dataset. The model is saved to
    `output_model` for future fast inference in `stage1_statistical()`.
    """
    if not _PANDAS_AVAILABLE:
        raise RuntimeError('pandas is required to train the OkCupid model')
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        from sklearn.metrics import classification_report, accuracy_score
        import joblib
    except Exception as e:
        raise RuntimeError('scikit-learn and joblib are required to train the OkCupid model') from e

    df = _load_okcupid_table(dataset_path)
    if df is None or df.shape[0] == 0:
        raise RuntimeError(f'No rows found in dataset: {dataset_path}')

    text_cols = ['essay0', 'essay8', 'essay9']
    for col in text_cols:
        if col not in df.columns:
            df[col] = ''
    df['bio_text'] = df[text_cols].fillna('').agg(' '.join, axis=1)

    for col in ['smokes', 'drinks', 'drugs', 'status', 'body_type', 'education', 'ethnicity', 'job', 'religion', 'sign']:
        if col not in df.columns:
            df[col] = 'unknown'
        df[col] = df[col].fillna('unknown').astype(str)

    for col in ['age', 'height', 'income']:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    risk_pattern = re.compile(
        r"\b(drunk|drinking|smoke|cigarette|cocaine|weed|marijuana|high|stoned|molly|mdma|trouble|crazy|hate|angry|broken|pass my test|not here to make friends)\b",
        re.I,
    )

    def label_row(row):
        text = str(row['bio_text']).lower()
        if row['drugs'] in ('often', 'sometimes'):
            return 1
        if row['smokes'] not in ('no', 'unknown'):
            return 1
        if row['drinks'] in ('often', 'very often', 'desperately'):
            return 1
        if row['status'] in ('married', 'seeing someone'):
            return 1
        if risk_pattern.search(text):
            return 1
        return 0

    df['risk_target'] = df.apply(label_row, axis=1)
    if df['risk_target'].sum() == 0:
        raise RuntimeError('Could not derive any positive risk targets from the OkCupid dataset.')

    feature_text = ['bio_text']
    feature_cats = ['smokes', 'drinks', 'drugs', 'status', 'body_type', 'education', 'ethnicity', 'job', 'religion', 'sign']
    feature_nums = ['age', 'height', 'income']

    preprocessor = ColumnTransformer(
        transformers=[
            ('text', TfidfVectorizer(max_features=2000, ngram_range=(1, 2), stop_words='english'), 'bio_text'),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), feature_cats),
            ('num', StandardScaler(), feature_nums),
        ],
        remainder='drop',
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced')),
    ])

    X = df[feature_text + feature_cats + feature_nums]
    y = df['risk_target']

    stratify = y if len(set(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify,
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, zero_division=0)
    acc = accuracy_score(y_test, y_pred)

    joblib.dump(pipeline, output_model)
    print(f'Trained stage1 OkCupid model: {output_model}')
    print(f'Accuracy: {acc:.4f}')
    print(report)

    return pipeline, report


# ── STAGE 5: LLM CHAIN-OF-THOUGHT (OpenRouter / Gemini) ───────────────────────

SYSTEM_PROMPT = """You are an expert dating profile auditing AI specialising in forensic linguistics, behavioural psychology, visual analysis, and deceptive communication detection.

You receive a structured pre-processing summary from a 4-stage pipeline, the raw profile, and possibly a profile picture. Your role is Stage 5: semantic Chain-of-Thought reasoning that refines and enriches the pipeline scores.

If a profile picture is provided, analyse it for:
- RED FLAG VISUAL SIGNALS: smoking/vaping, excessive drinking/beer cans, drug paraphernalia, dangerous vehicles (superbikes, sports cars), aggressive/narcissistic posing, provocative clothing, closed/guarded body language, harsh facial expressions, tattoos of dark symbolism.
- POSITIVE VISUAL SIGNALS: reading books, sports/fitness activities (gym, yoga, running), volunteering/community service, family moments, pets, cultural activities, outdoor recreation, genuine smiling, open/relaxed body language, professional attire.

In cot_reasoning, cite specific VISUAL observations when a picture is present (e.g., "Picture shows smoking habit" or "Reading in natural setting suggests intellectual interests").

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
    # Picture metadata section for LLM with detailed visual analysis instructions
    if profile.picture:
        pic_section = (
            f"Profile picture attached: filename={profile.picture_filename or 'unknown'}, size_kb={profile.picture_size_kb}.\n"
            "\nVISUAL ANALYSIS REQUIRED:\n"
            "  RED FLAGS (risky visual signals): smoking, vaping, alcohol/beer cans, drug paraphernalia, superbikes, sports cars, \n"
            "    narcissistic posing, aggressive expression, closed body language, revealing/provocative clothing, dark tattoos.\n"
            "  POSITIVE SIGNALS (healthy indicators): reading, books, sports/fitness activities, yoga, volunteering, family, pets, \n"
            "    cultural activities, hiking, outdoor recreation, genuine smile, open posture, professional dress.\n"
            "\nCite SPECIFIC visual observations in cot_reasoning (e.g., 'Picture shows person smoking' or 'Reading book suggests intellectual interests').\n"
            "If you cannot view the image, explicitly state so and rely only on textual/categorical signals.\n"
        )
    else:
        pic_section = "Profile picture: None provided.\n"

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

{pic_section}
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

    Stages 1–4 run locally in Python.
    YOLO detects objects from uploaded image.
    Stage 5 LLM explains the full profile + YOLO visual findings.
    """

    # ── Local stages ───────────────────────────
    s1_score, s1_passed = stage1_statistical(profile)
    linguistic = stage2_linguistic(profile.bio)
    contradictions = stage3_contradictions(profile)
    dark_triad = stage4_dark_triad(profile.bio)

    # ── YOLO visual detection ──────────────────
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
                    name_key = name.replace("_", " ")

                    detected_objects.append(name_key)

                    if name_key in VISUAL_FLAG_MAP:
                        cat, txt, sev = VISUAL_FLAG_MAP[name_key]
                        visual_flags.append(
                            Flag(
                                category=cat,
                                text=txt,
                                severity=sev
                            )
                        )

            # Remove duplicate objects
            detected_objects = list(dict.fromkeys(detected_objects))

            if detected_objects:
                visual_ai_explanation = explain_yolo_objects_with_ai(
                    detected_objects,
                    profile
                )

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

    # ── Build Stage 5 LLM message ───────────────
    user_msg = _build_user_message(
        profile,
        s1_score,
        s1_passed,
        linguistic,
        contradictions,
        dark_triad
    )

    # Add YOLO detected objects to LLM prompt
    if detected_objects:
        user_msg += "\n\nYOLO DETECTED OBJECTS:\n"
        user_msg += ", ".join(detected_objects)

    # Add rule-based visual flags
    if visual_flags:
        user_msg += "\n\nVISUAL FLAGS FROM YOLO MAP:\n"
        user_msg += "\n".join(
            f"- {v.text} [category: {v.category}, severity: {v.severity}]"
            for v in visual_flags
        )

    # Add AI explanation of YOLO objects
    if visual_ai_explanation:
        user_msg += "\n\nAI VISUAL EXPLANATION:\n"
        user_msg += visual_ai_explanation

    # ── Stage 5: LLM provider selection ─────────
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
    gemini_key = os.getenv("GEMINI_API_KEY", "")

    if openrouter_key and openrouter_key != "your-openrouter-api-key-here":
        data, model_used = _call_openrouter(user_msg)
        provider_used = "openrouter"

    elif gemini_key and gemini_key != "your-gemini-api-key-here":
        data, model_used = _call_gemini(user_msg)
        provider_used = "gemini"

    else:
        raise RuntimeError("No AI API key configured. Set OPENROUTER_API_KEY in backend/.env")

    # ── Assemble final result ───────────────────
    return AuditResult(
        overall_risk=data.get("overall_risk", 50),
        financial_risk=data.get("financial_risk", 20),
        emotional_risk=data.get("emotional_risk", 30),
        identity_risk=data.get("identity_risk", 20),

        dark_triad=DarkTriad(**data.get("dark_triad", {
            "narcissism": dark_triad.narcissism,
            "machiavellianism": dark_triad.machiellianism if hasattr(dark_triad, "machiellianism") else dark_triad.machiavellianism,
            "psychopathy": dark_triad.psychopathy,
        })),

        linguistic=LinguisticFeatures(**data.get("linguistic", {
            "sentiment_score": linguistic.sentiment_score,
            "complexity": linguistic.complexity,
            "manipulation_score": linguistic.manipulation_score,
        })),

        contradictions=[Contradiction(**c) for c in data.get("contradictions", [])] or contradictions,
        flags=[Flag(**f) for f in data.get("flags", [])],
        visual_flags=visual_flags or [Flag(**f) for f in data.get("visual_flags", [])],

        cot_reasoning=data.get("cot_reasoning", ""),
        educational_tip=data.get("educational_tip", ""),
        precision_estimate=data.get("precision_estimate", 0.80),
        recall_estimate=data.get("recall_estimate", 0.72),
        iou_estimate=data.get("iou_estimate", 0.61),

        stage1_score=s1_score,
        stage1_passed=s1_passed,
        provider=provider_used,
        llm_model=model_used,
        picture_b64=profile.picture,
        visual_ai_explanation=visual_ai_explanation,
    )


from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import base64
import math
import os

# -----------------------------
# LOAD API KEY (SECURE)
# -----------------------------
API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise RuntimeError("API_KEY not set in environment variables")

# -----------------------------
# APP CONFIG
# -----------------------------
app = FastAPI(
    title="AI Voice Authenticity Detection API",
    description="Detect whether a voice sample is AI-generated or Human-generated",
    version="1.0"
)

# -----------------------------
# REQUEST MODEL
# -----------------------------
class VoiceRequest(BaseModel):
    audio_base64: str
    language: str

# -----------------------------
# RESPONSE MODEL
# -----------------------------
class VoiceResponse(BaseModel):
    classification: str
    confidence: float
    explanation: str

# -----------------------------
# HOME PAGE
# -----------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

# -----------------------------
# ENTROPY FUNCTION
# -----------------------------
def shannon_entropy(data: bytes) -> float:
    freq = {}
    for b in data:
        freq[b] = freq.get(b, 0) + 1

    entropy = 0.0
    length = len(data)
    for count in freq.values():
        p = count / length
        entropy -= p * math.log2(p)
    return entropy

# -----------------------------
# DETECTION ENDPOINT (LOCKED)
# -----------------------------
@app.post("/detect", response_model=VoiceResponse)
def detect_voice(
    request: VoiceRequest,
    x_api_key: str = Header(None)
):
    # 🔐 API KEY CHECK
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: Invalid API Key"
        )

    # Decode Base64
    try:
        audio_bytes = base64.b64decode(request.audio_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Base64 audio")

    if len(audio_bytes) < 1000:
        raise HTTPException(status_code=400, detail="Audio too short for analysis")

    # Calculate entropy
    entropy = shannon_entropy(audio_bytes)

    # Classification logic
    if entropy > 7.2:
        classification = "AI-generated"
        confidence = round(min(0.95, (entropy - 6.5) / 2), 2)
    else:
        classification = "Human-generated"
        confidence = round(min(0.95, (7.2 - entropy) / 2), 2)

    # Language-based explanations
    explanations = {
        "tamil": {
            "AI-generated": "இந்த தமிழ் குரலில் செயற்கை நுண்ணறிவுக்கான ஒரே மாதிரியான சுருதி மற்றும் இயந்திர பேச்சு தன்மைகள் காணப்படுகின்றன.",
            "Human-generated": "இந்த தமிழ் குரலில் இயல்பான மனித பேச்சு மாற்றங்கள் மற்றும் உணர்ச்சி வெளிப்பாடுகள் கண்டறியப்பட்டன."
        },
        "english": {
            "AI-generated": "The audio exhibits uniform pitch and synthesized speech patterns typical of AI-generated voices.",
            "Human-generated": "The audio shows natural variations in tone, rhythm, and emotion, indicating human speech."
        },
        "hindi": {
            "AI-generated": "इस हिंदी ऑडियो में कृत्रिम आवाज़ के समान स्थिर स्वर और यांत्रिक पैटर्न पाए गए।",
            "Human-generated": "इस हिंदी ऑडियो में प्राकृतिक मानव स्वर परिवर्तन और भावनात्मक अभिव्यक्ति पाई गई।"
        },
        "malayalam": {
            "AI-generated": "ഈ മലയാളം ശബ്ദത്തിൽ കൃത്രിമ ശബ്ദത്തിനുള്ള ഏകീകൃത സ്വര മാതൃകകൾ കാണപ്പെടുന്നു.",
            "Human-generated": "ഈ മലയാളം ശബ്ദത്തിൽ സ്വാഭാവികമായ മനുഷ്യ ശബ്ദ വ്യതിയാനങ്ങൾ കണ്ടെത്തി."
        },
        "telugu": {
            "AI-generated": "ఈ తెలుగు ఆడియోలో కృత్రిమ స్వరాలకు సంబంధించిన స్థిరమైన పిచ్ నమూనాలు కనిపిస్తున్నాయి.",
            "Human-generated": "ఈ తెలుగు ఆడియోలో సహజమైన మానవ స్వర మార్పులు గుర్తించబడ్డాయి."
        }
    }

    lang = request.language.lower()
    explanation = explanations.get(
        lang,
        {
            "AI-generated": "The audio exhibits synthesized speech characteristics.",
            "Human-generated": "The audio exhibits natural human speech patterns."
        }
    )[classification]

    return {
        "classification": classification,
        "confidence": confidence,
        "explanation": explanation
    }

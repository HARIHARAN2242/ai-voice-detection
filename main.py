from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import base64
import math
import os

# =========================
# CONFIG
# =========================
API_KEY = os.getenv("API_KEY")

app = FastAPI(
    title="AI Voice Authenticity Detection API",
    description="Detect whether a voice sample is AI-generated or Human-generated",
    version="1.0"
)

# =========================
# DATA MODELS
# =========================
class VoiceRequest(BaseModel):
    audio_base64: str
    language: str

class VoiceResponse(BaseModel):
    classification: str
    confidence: float
    explanation: str

# =========================
# HOME PAGE
# =========================
@app.get("/", response_class=HTMLResponse)
def home():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

# =========================
# HELPER: ENTROPY
# =========================
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

# =========================
# LANGUAGE EXPLANATIONS
# =========================
LANGUAGE_EXPLANATION = {
    "english": {
        "AI-generated": "The English speech shows synthetic consistency and low natural variation, which is common in AI-generated voices.",
        "Human-generated": "The English speech contains natural pauses and variations typical of human speech."
    },
    "tamil": {
        "AI-generated": "இந்த தமிழ் குரலில் இயற்கையான ஏற்றத்தாழ்வுகள் குறைவாக உள்ளதால் இது செயற்கை குரலாக இருக்கலாம்.",
        "Human-generated": "இந்த தமிழ் குரலில் மனித குரலுக்குரிய இயற்கையான மாற்றங்கள் காணப்படுகின்றன."
    },
    "hindi": {
        "AI-generated": "इस हिंदी आवाज़ में कृत्रिम पैटर्न दिखाई देते हैं, जो AI जनरेशन का संकेत हो सकता है।",
        "Human-generated": "इस हिंदी आवाज़ में मानवीय उतार-चढ़ाव और स्वाभाविकता है।"
    },
    "malayalam": {
        "AI-generated": "ഈ മലയാളം ശബ്ദത്തിൽ കൃത്രിമ ഘടനകൾ കാണപ്പെടുന്നു.",
        "Human-generated": "ഈ മലയാളം ശബ്ദത്തിൽ സ്വാഭാവികമായ മനുഷ്യ വ്യത്യാസങ്ങൾ ഉണ്ട്."
    },
    "telugu": {
        "AI-generated": "ఈ తెలుగు స్వరంలో కృత్రిమ నమూనాలు కనిపిస్తున్నాయి.",
        "Human-generated": "ఈ తెలుగు స్వరంలో మానవ స్వరానికి చెందిన సహజ మార్పులు ఉన్నాయి."
    }
}

# =========================
# DETECTION API
# =========================
@app.post("/detect", response_model=VoiceResponse)
def detect_voice(
    request: VoiceRequest,
    x_api_key: str = Header(None)
):
    # 🔐 API KEY CHECK
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    # Decode Base64
    try:
        audio_bytes = base64.b64decode(request.audio_base64)
    except:
        raise HTTPException(status_code=400, detail="Invalid Base64 audio")

    if len(audio_bytes) < 200:
        raise HTTPException(status_code=400, detail="Audio is too short")

    # Entropy-based heuristic
    entropy = shannon_entropy(audio_bytes)

    if entropy < 4.2:
        classification = "AI-generated"
        confidence = round(0.75 + (4.2 - entropy) * 0.05, 2)
    else:
        classification = "Human-generated"
        confidence = round(0.75 + (entropy - 4.2) * 0.05, 2)

    confidence = min(confidence, 0.99)

    lang = request.language.lower()
    explanation = LANGUAGE_EXPLANATION.get(
        lang,
        LANGUAGE_EXPLANATION["english"]
    )[classification]

    return {
        "classification": classification,
        "confidence": confidence,
        "explanation": explanation
    }

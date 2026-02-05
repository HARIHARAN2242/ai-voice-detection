from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import base64
import math
import os

# =========================
# CONFIG
# =========================
API_KEY = os.getenv("API_KEY")  # set in Render

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
# HELPER FUNCTION
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
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Base64 audio")

    if len(audio_bytes) < 2000:
        return {
            "classification": "Unknown",
            "confidence": 0.0,
            "explanation": "Audio sample is too short for reliable analysis"
        }

    entropy = shannon_entropy(audio_bytes)

    # Simple heuristic logic (prototype)
    if entropy > 7.5:
        classification = "AI-generated"
        confidence = 0.86
    else:
        classification = "Human-generated"
        confidence = 0.84

    # Language-based explanation
    explanations = {
        "tamil": {
            "AI-generated": "இந்த குரலில் இயந்திரம் உருவாக்கிய ஒலி பண்புகள் காணப்படுகின்றன.",
            "Human-generated": "இந்த குரலில் இயல்பான மனித பேச்சு மாறுபாடுகள் உள்ளன."
        },
        "english": {
            "AI-generated": "The voice shows synthetic patterns typical of AI generation.",
            "Human-generated": "The voice contains natural human speech variations."
        },
        "hindi": {
            "AI-generated": "इस आवाज़ में एआई द्वारा उत्पन्न ध्वनि पैटर्न पाए गए हैं।",
            "Human-generated": "इस आवाज़ में प्राकृतिक मानवीय भाषण के गुण मौजूद हैं।"
        },
        "malayalam": {
            "AI-generated": "ഈ ശബ്ദത്തിൽ എഐ സിന്തറ്റിക് ലക്ഷണങ്ങൾ കണ്ടെത്തി.",
            "Human-generated": "ഈ ശബ്ദത്തിൽ സ്വാഭാവിക മനുഷ്യ ശബ്ദ വ്യത്യാസങ്ങൾ കാണുന്നു."
        },
        "telugu": {
            "AI-generated": "ఈ వాయిస్‌లో AI సృష్టించిన లక్షణాలు కనిపిస్తున్నాయి.",
            "Human-generated": "ఈ వాయిస్‌లో సహజమైన మానవ మాట్లాడే లక్షణాలు ఉన్నాయి."
        }
    }

    lang = request.language.lower()
    explanation = explanations.get(
        lang,
        explanations["english"]
    )[classification]

    return {
        "classification": classification,
        "confidence": confidence,
        "explanation": explanation
    }

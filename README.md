

# AI-Generated Voice Detection API

This project detects whether a given voice sample is **AI-generated** or **Human-generated**.

## Features
- Accepts Base64-encoded MP3 audio
- Supports multiple languages:
  - English
  - Tamil
  - Hindi
  - Malayalam
  - Telugu
- Returns:
  - Classification
  - Confidence score
  - Explanation

## Tech Stack
- Python
- FastAPI
- HTML, CSS
- JavaScript

## API Endpoint
POST `/detect`

### Request Body
```json
{
  "audio_base64": "<base64_audio>",
  "language": "English"
}

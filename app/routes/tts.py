"""
TTS Routes - OpenAI-compatible Text-to-Speech API.

Implements:
- POST /v1/audio/speech
- GET /v1/audio/voices
"""

from typing import Optional, List, Dict
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import Response

from app.config import settings
from app.models.tts_models import TTSRequest
from app.services.tts_service import TTSService


router = APIRouter(prefix="/v1/audio", tags=["Text-to-Speech"])

# Global service instances (initialized on startup)
_tts_services: Dict[str, TTSService] = {}


def get_tts_service(model_id: Optional[str] = None) -> TTSService:
    """Get TTS service for the given model ID."""
    if not _tts_services:
        raise HTTPException(
            status_code=503,
            detail="TTS service not initialized"
        )

    if model_id:
        # Route by model ID
        if model_id == "vibevoice-1.5b" and "1.5b" in _tts_services:
            service = _tts_services["1.5b"]
        elif model_id in ("vibevoice-realtime", "tts-1", "tts-1-hd") and "0.5b" in _tts_services:
            service = _tts_services["0.5b"]
        else:
            # Use whatever is available
            service = next(iter(_tts_services.values()))
    else:
        # Default: prefer 0.5b for backward compat
        service = _tts_services.get("0.5b") or next(iter(_tts_services.values()))

    if not service.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="TTS model not loaded"
        )
    return service


def init_tts_service() -> Dict[str, TTSService]:
    """Initialize TTS service(s) on startup based on configuration."""
    global _tts_services
    if not settings.tts_enabled:
        print("[TTS] Service disabled by configuration")
        return {}

    model_type = settings.tts_model_type.lower()

    if model_type in ("0.5b", "both"):
        print("[TTS] Initializing 0.5B streaming model...")
        service_05b = TTSService(model_type="0.5b")
        service_05b.load()
        _tts_services["0.5b"] = service_05b

    if model_type in ("1.5b", "both"):
        print("[TTS] Initializing 1.5B full model...")
        service_15b = TTSService(model_type="1.5b")
        service_15b.load()
        _tts_services["1.5b"] = service_15b

    return _tts_services


# Media type mapping
MEDIA_TYPES = {
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
}


@router.post("/speech")
async def create_speech(
    request: TTSRequest,
):
    """
    Generate speech from text (OpenAI-compatible).

    Returns audio in the requested format (default: wav).
    """
    # Validate input
    if not request.input or not request.input.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty")

    if len(request.input) > 4096:
        raise HTTPException(
            status_code=400,
            detail="Input text too long. Maximum 4096 characters."
        )

    # Validate format
    if request.response_format not in MEDIA_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid response_format. Must be one of: {list(MEDIA_TYPES.keys())}"
        )

    # Get service based on requested model
    service = get_tts_service(model_id=request.model)

    try:
        audio_bytes = service.synthesize(
            text=request.input,
            voice=request.voice,
            speed=request.speed,
            response_format=request.response_format,
        )

        media_type = MEDIA_TYPES.get(request.response_format, "audio/wav")

        return Response(
            content=audio_bytes,
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename=speech.{request.response_format}"
            }
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Speech synthesis failed: {e}")


@router.get("/voices")
async def list_voices() -> dict:
    """
    List available voices from all loaded TTS services.

    Note: This is a non-standard endpoint for convenience.
    """
    if not _tts_services:
        raise HTTPException(status_code=503, detail="TTS service not initialized")

    # Aggregate voices from all loaded services
    all_voices = {}
    for model_type, service in _tts_services.items():
        if service.is_loaded:
            for v in service.get_available_voices():
                if v not in all_voices:
                    all_voices[v] = {
                        "voice_id": v,
                        "name": v.title(),
                        "language": "en",
                        "models": [service.model_id],
                    }
                else:
                    all_voices[v]["models"].append(service.model_id)

    return {
        "voices": list(all_voices.values()),
        "default_voice": settings.default_voice,
    }

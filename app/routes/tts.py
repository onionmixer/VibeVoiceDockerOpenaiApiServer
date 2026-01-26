"""
TTS Routes - OpenAI-compatible Text-to-Speech API.

Implements:
- POST /v1/audio/speech
"""

from typing import Optional, List
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import Response

from app.config import settings
from app.models.tts_models import TTSRequest
from app.services.tts_service import TTSService


router = APIRouter(prefix="/v1/audio", tags=["Text-to-Speech"])

# Global service instance (initialized on startup)
_tts_service: Optional[TTSService] = None


def get_tts_service() -> TTSService:
    """Dependency to get TTS service."""
    if _tts_service is None:
        raise HTTPException(
            status_code=503,
            detail="TTS service not initialized"
        )
    if not _tts_service.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="TTS model not loaded"
        )
    return _tts_service


def init_tts_service() -> Optional[TTSService]:
    """Initialize TTS service on startup."""
    global _tts_service
    if not settings.tts_enabled:
        print("[TTS] Service disabled by configuration")
        return None

    _tts_service = TTSService()
    _tts_service.load()
    return _tts_service


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
    service: TTSService = Depends(get_tts_service),
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
async def list_voices(
    service: TTSService = Depends(get_tts_service),
) -> dict:
    """
    List available voices.

    Note: This is a non-standard endpoint for convenience.
    """
    voices = service.get_available_voices()

    return {
        "voices": [
            {
                "voice_id": v,
                "name": v.title(),
                "language": "en",
            }
            for v in voices
        ],
        "default_voice": settings.default_voice,
    }

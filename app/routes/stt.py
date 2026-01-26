"""
STT Routes - OpenAI-compatible Speech-to-Text API.

Implements:
- POST /v1/audio/transcriptions
"""

from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import PlainTextResponse, JSONResponse

from app.config import settings
from app.services.stt_service import STTService


router = APIRouter(prefix="/v1/audio", tags=["Speech-to-Text"])

# Global service instance (initialized on startup)
_stt_service: Optional[STTService] = None


def get_stt_service() -> STTService:
    """Dependency to get STT service."""
    if _stt_service is None:
        raise HTTPException(
            status_code=503,
            detail="STT service not initialized"
        )
    if not _stt_service.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="STT model not loaded"
        )
    return _stt_service


def init_stt_service() -> Optional[STTService]:
    """Initialize STT service on startup."""
    global _stt_service
    if not settings.asr_enabled:
        print("[STT] Service disabled by configuration")
        return None

    _stt_service = STTService()
    _stt_service.load()
    return _stt_service


@router.post("/transcriptions")
async def create_transcription(
    file: UploadFile = File(..., description="Audio file to transcribe"),
    model: str = Form(default="vibevoice-asr", description="Model ID"),
    language: Optional[str] = Form(default=None, description="Language code (ISO 639-1)"),
    prompt: Optional[str] = Form(default=None, description="Optional prompt/context"),
    response_format: str = Form(default="json", description="Response format"),
    temperature: float = Form(default=0.0, description="Sampling temperature"),
    service: STTService = Depends(get_stt_service),
):
    """
    Transcribe audio to text (OpenAI-compatible).

    Supported audio formats: mp3, mp4, mpeg, mpga, m4a, wav, webm

    Response formats:
    - json: {"text": "..."}
    - text: Plain text
    - verbose_json: Includes timestamps and segments
    - srt: SubRip subtitle format
    - vtt: WebVTT subtitle format
    """
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # Read file content
    try:
        audio_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")

    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    # Validate response format
    valid_formats = ["json", "text", "verbose_json", "srt", "vtt"]
    if response_format not in valid_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid response_format. Must be one of: {valid_formats}"
        )

    try:
        result = service.transcribe(
            audio_bytes=audio_bytes,
            filename=file.filename,
            language=language,
            prompt=prompt,
            response_format=response_format,
            temperature=temperature,
        )

        # Return appropriate response type
        if response_format == "text":
            return PlainTextResponse(content=result.get("text", ""))
        elif response_format in ("srt", "vtt"):
            media_type = "text/srt" if response_format == "srt" else "text/vtt"
            return PlainTextResponse(content=result, media_type=media_type)
        else:
            return JSONResponse(content=result)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")


@router.post("/translations")
async def create_translation(
    file: UploadFile = File(..., description="Audio file to translate"),
    model: str = Form(default="vibevoice-asr", description="Model ID"),
    prompt: Optional[str] = Form(default=None, description="Optional prompt/context"),
    response_format: str = Form(default="json", description="Response format"),
    temperature: float = Form(default=0.0, description="Sampling temperature"),
    service: STTService = Depends(get_stt_service),
):
    """
    Translate audio to English text (OpenAI-compatible).

    Note: VibeVoice-ASR handles multilingual audio natively,
    so this endpoint functions similarly to transcriptions
    but attempts to output in English.
    """
    # Read file content
    try:
        audio_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")

    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        # Force English output
        result = service.transcribe(
            audio_bytes=audio_bytes,
            filename=file.filename or "audio.wav",
            language="en",
            prompt=prompt,
            response_format=response_format,
            temperature=temperature,
        )

        if response_format == "text":
            return PlainTextResponse(content=result.get("text", ""))
        elif response_format in ("srt", "vtt"):
            media_type = "text/srt" if response_format == "srt" else "text/vtt"
            return PlainTextResponse(content=result, media_type=media_type)
        else:
            return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {e}")

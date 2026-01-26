"""
API Routes for OpenAI-compatible endpoints.
"""

from app.routes.tts import router as tts_router
from app.routes.stt import router as stt_router

__all__ = ["tts_router", "stt_router"]

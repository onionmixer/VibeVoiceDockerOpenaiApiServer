"""
VibeVoice Docker OpenAI API Server

Main FastAPI application providing OpenAI-compatible TTS and STT APIs.
"""

import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.config import settings
from app.routes import tts_router, stt_router
from app.routes.stt import init_stt_service
from app.routes.tts import init_tts_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    # Startup
    print("=" * 60)
    print(f"VibeVoice OpenAI API Server v{__version__}")
    print("=" * 60)
    print(f"Device: {settings.device}")
    print(f"Attention: {settings.attn_implementation}")
    print(f"ASR Enabled: {settings.asr_enabled}")
    print(f"TTS Enabled: {settings.tts_enabled}")
    print("=" * 60)

    # Initialize services
    if settings.asr_enabled:
        print("\n[Startup] Initializing STT service...")
        try:
            init_stt_service()
            print("[Startup] STT service ready")
        except Exception as e:
            print(f"[Startup] STT service failed: {e}")

    if settings.tts_enabled:
        print("\n[Startup] Initializing TTS service...")
        try:
            init_tts_service()
            print("[Startup] TTS service ready")
        except Exception as e:
            print(f"[Startup] TTS service failed: {e}")

    print("\n" + "=" * 60)
    print(f"Server ready at http://{settings.api_host}:{settings.api_port}")
    print("=" * 60 + "\n")

    yield

    # Shutdown
    print("\n[Shutdown] Cleaning up...")


# Create FastAPI app
app = FastAPI(
    title="VibeVoice OpenAI API Server",
    description="OpenAI-compatible TTS and STT API using VibeVoice models",
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


# Include routers
app.include_router(stt_router)
app.include_router(tts_router)


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "VibeVoice OpenAI API Server",
        "version": __version__,
        "endpoints": {
            "tts": "/v1/audio/speech",
            "stt": "/v1/audio/transcriptions",
            "docs": "/docs",
        },
        "status": "running",
    }


# Health check endpoint
@app.get("/health")
async def health():
    """Health check endpoint."""
    from app.routes.stt import _stt_service
    from app.routes.tts import _tts_service

    stt_status = "disabled"
    if settings.asr_enabled:
        stt_status = "ready" if (_stt_service and _stt_service.is_loaded) else "loading"

    tts_status = "disabled"
    if settings.tts_enabled:
        tts_status = "ready" if (_tts_service and _tts_service.is_loaded) else "loading"

    return {
        "status": "healthy",
        "services": {
            "stt": stt_status,
            "tts": tts_status,
        }
    }


# Models endpoint (OpenAI-compatible)
@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible)."""
    models = []

    if settings.asr_enabled:
        models.append({
            "id": "vibevoice-asr",
            "object": "model",
            "created": 1700000000,
            "owned_by": "microsoft",
            "permission": [],
            "root": "vibevoice-asr",
            "parent": None,
        })

    if settings.tts_enabled:
        models.append({
            "id": "vibevoice-realtime",
            "object": "model",
            "created": 1700000000,
            "owned_by": "microsoft",
            "permission": [],
            "root": "vibevoice-realtime",
            "parent": None,
        })

    return {
        "object": "list",
        "data": models,
    }


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": str(exc),
                "type": type(exc).__name__,
                "code": "internal_error",
            }
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )

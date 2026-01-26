# VibeVoice OpenAI API Server

OpenAI-compatible TTS (Text-to-Speech) and STT (Speech-to-Text) API server using Microsoft VibeVoice models.

## Features

- **OpenAI-compatible API**: Drop-in replacement for OpenAI's audio APIs
- **STT (Speech-to-Text)**: Using VibeVoice-ASR (7B) with 60-minute long-form support
- **TTS (Text-to-Speech)**: Using VibeVoice-Realtime (0.5B) with ~200ms latency
- **Multi-architecture**: Supports both x86_64 (NVIDIA CUDA) and ARM64 (DGX Spark)
- **Open WebUI Integration**: Ready-to-use with Open WebUI

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/audio/speech` | POST | Text-to-Speech synthesis |
| `/v1/audio/transcriptions` | POST | Speech-to-Text transcription |
| `/v1/audio/translations` | POST | Speech translation to English |
| `/v1/audio/voices` | GET | List available voices |
| `/v1/models` | GET | List available models |
| `/health` | GET | Health check |

## Quick Start

### 1. Download Models

```bash
# Install HuggingFace CLI
pip install huggingface_hub

# Download ASR model (~17GB)
huggingface-cli download microsoft/VibeVoice-ASR \
    --local-dir /path/to/models/VibeVoice-ASR

# Download TTS model (~2GB)
huggingface-cli download microsoft/VibeVoice-Realtime-0.5B \
    --local-dir /path/to/models/VibeVoice-Realtime-0.5B
```

### 2. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit with your model path
# MODEL_DIR=/path/to/your/models
```

### 3. Run with Docker Compose

#### x86_64 (Standard NVIDIA GPU)

```bash
export MODEL_DIR=/path/to/your/models
docker compose up -d --build
```

#### DGX Spark (ARM64)

```bash
export MODEL_DIR=/path/to/your/models
docker compose -f docker-compose.dgx-spark.yml up -d --build
```

#### With Open WebUI

```bash
export MODEL_DIR=/path/to/your/models
docker compose -f docker-compose.with-openwebui.yml up -d --build
```

## API Usage Examples

### Text-to-Speech

```bash
curl -X POST http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vibevoice-realtime",
    "input": "Hello, how are you today?",
    "voice": "carter",
    "response_format": "wav"
  }' \
  --output speech.wav
```

### Speech-to-Text

```bash
curl -X POST http://localhost:8080/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=vibevoice-asr" \
  -F "response_format=json"
```

### Verbose Transcription (with timestamps)

```bash
curl -X POST http://localhost:8080/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=vibevoice-asr" \
  -F "response_format=verbose_json"
```

## Open WebUI Configuration

### Environment Variables

```bash
# TTS Settings
AUDIO_TTS_ENGINE=openai
AUDIO_TTS_OPENAI_API_BASE_URL=http://vibevoice-api:8080/v1
AUDIO_TTS_OPENAI_API_KEY=not-needed
AUDIO_TTS_MODEL=vibevoice-realtime
AUDIO_TTS_VOICE=carter

# STT Settings
AUDIO_STT_ENGINE=openai
AUDIO_STT_OPENAI_API_BASE_URL=http://vibevoice-api:8080/v1
AUDIO_STT_OPENAI_API_KEY=not-needed
AUDIO_STT_MODEL=vibevoice-asr
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEVICE` | `cuda` | Device (cuda, cpu) |
| `ATTN_IMPLEMENTATION` | `flash_attention_2` | Attention (flash_attention_2, sdpa) |
| `ASR_ENABLED` | `true` | Enable STT service |
| `TTS_ENABLED` | `true` | Enable TTS service |
| `ASR_MODEL_PATH` | `/models/VibeVoice-ASR` | ASR model path |
| `TTS_MODEL_PATH` | `/models/VibeVoice-Realtime-0.5B` | TTS model path |
| `DEFAULT_VOICE` | `carter` | Default TTS voice |
| `API_PORT` | `8080` | API server port |

### Platform-Specific Settings

| Platform | `ATTN_IMPLEMENTATION` | Notes |
|----------|----------------------|-------|
| x86_64 (CUDA) | `flash_attention_2` | Recommended for performance |
| DGX Spark (ARM64) | `sdpa` | Required (Flash Attention not supported) |

## GPU Requirements

| Model | VRAM | Notes |
|-------|------|-------|
| VibeVoice-ASR (7B) | ~17GB | bfloat16 |
| VibeVoice-Realtime (0.5B) | ~2-4GB | Lightweight |
| Both models | ~20GB+ | Consider separate deployment |

## Model Directory Structure

```
/path/to/models/
├── VibeVoice-ASR/
│   ├── config.json
│   ├── model-00001-of-00008.safetensors
│   ├── model-00002-of-00008.safetensors
│   ├── ... (8 files total)
│   └── model.safetensors.index.json
├── VibeVoice-Realtime-0.5B/
│   ├── config.json
│   ├── model.safetensors
│   └── ...
└── voices/                    # TTS voice presets (optional)
    └── streaming_model/
        ├── carter.pt
        ├── wayne.pt
        └── ...
```

## Supported Audio Formats

### Input (STT)
- mp3, mp4, mpeg, mpga, m4a, wav, webm

### Output (TTS)
- wav (default), mp3, opus, aac, flac, pcm

## License

- This project: MIT License
- VibeVoice models: MIT License (research/development recommended)

## References

- [VibeVoice GitHub](https://github.com/microsoft/VibeVoice)
- [VibeVoice-ASR on HuggingFace](https://huggingface.co/microsoft/VibeVoice-ASR)
- [VibeVoice-Realtime on HuggingFace](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B)
- [Open WebUI](https://docs.openwebui.com/)

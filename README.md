# VibeVoice OpenAI API Server

OpenAI-compatible TTS (Text-to-Speech) and STT (Speech-to-Text) API server using Microsoft VibeVoice models.

## Features

- **OpenAI-compatible API**: Drop-in replacement for OpenAI's audio APIs
- **STT (Speech-to-Text)**: Using VibeVoice-ASR (7B) with 60-minute long-form support
- **TTS (Text-to-Speech)**: Dual model support
  - **VibeVoice-Realtime (0.5B)**: ~200ms latency, 25 voice presets, streaming
  - **VibeVoice-1.5B**: Higher quality, 9 voice presets, CFG-based
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

# Download TTS 0.5B streaming model (~2GB)
huggingface-cli download microsoft/VibeVoice-Realtime-0.5B \
    --local-dir /path/to/models/VibeVoice-Realtime-0.5B

# Download TTS 1.5B full model (~5.4GB) from Microsoft
huggingface-cli download microsoft/VibeVoice \
    --local-dir /path/to/models/VibeVoice-1.5B
```

### 2. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit with your model path
# MODEL_DIR=/path/to/your/models
```

### 3. Run with Docker Compose

#### Using run-test.sh (Recommended)

```bash
# TTS 0.5B only (default)
MODEL_DIR=/path/to/your/models ./run-test.sh

# TTS 1.5B only
MODEL_DIR=/path/to/your/models ./run-test.sh --model 1.5b

# Both TTS models
MODEL_DIR=/path/to/your/models ./run-test.sh --model both

# With ASR enabled
MODEL_DIR=/path/to/your/models ./run-test.sh --asr

# Skip build (image already built)
MODEL_DIR=/path/to/your/models ./run-test.sh --model 1.5b --skip-build

# Stop containers
./run-test.sh --down
```

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

### Text-to-Speech (0.5B Streaming)

```bash
curl -X POST http://localhost:8899/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vibevoice-realtime",
    "input": "Hello, how are you today?",
    "voice": "carter",
    "response_format": "wav"
  }' \
  --output speech.wav
```

### Text-to-Speech (1.5B Full)

```bash
curl -X POST http://localhost:8899/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vibevoice-1.5b",
    "input": "Hello, how are you today?",
    "voice": "alice",
    "response_format": "wav"
  }' \
  --output speech.wav
```

### Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8899/v1", api_key="unused")

response = client.audio.speech.create(
    model="vibevoice-1.5b",
    input="Hello, how are you today?",
    voice="alice",
)
response.stream_to_file("speech.wav")
```

### Speech-to-Text

```bash
curl -X POST http://localhost:8899/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=vibevoice-asr" \
  -F "response_format=json"
```

### Verbose Transcription (with timestamps)

```bash
curl -X POST http://localhost:8899/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=vibevoice-asr" \
  -F "response_format=verbose_json"
```

## Open WebUI Configuration

### Environment Variables

```bash
# TTS Settings
AUDIO_TTS_ENGINE=openai
AUDIO_TTS_OPENAI_API_BASE_URL=http://vibevoice-api:8899/v1
AUDIO_TTS_OPENAI_API_KEY=not-needed
AUDIO_TTS_MODEL=vibevoice-1.5b        # or vibevoice-realtime for 0.5B
AUDIO_TTS_VOICE=alice                  # or carter, etc.

# STT Settings
AUDIO_STT_ENGINE=openai
AUDIO_STT_OPENAI_API_BASE_URL=http://vibevoice-api:8899/v1
AUDIO_STT_OPENAI_API_KEY=not-needed
AUDIO_STT_MODEL=vibevoice-asr
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEVICE` | `cuda` | Device (cuda, cpu) |
| `ATTN_IMPLEMENTATION` | `sdpa` | Attention (sdpa, flash_attention_2) |
| `ASR_ENABLED` | `true` | Enable STT service |
| `TTS_ENABLED` | `true` | Enable TTS service |
| `TTS_MODEL_TYPE` | `0.5b` | TTS model selection: `0.5b`, `1.5b`, or `both` |
| `ASR_MODEL_PATH` | `/models/VibeVoice-ASR` | ASR model path |
| `TTS_MODEL_PATH` | `/models/VibeVoice-Realtime-0.5B` | TTS 0.5B model path |
| `TTS_1_5B_MODEL_PATH` | `/models/VibeVoice-1.5B` | TTS 1.5B model path |
| `DEFAULT_VOICE` | `carter` | Default TTS voice |
| `API_PORT` | `8899` | Host-side API port (container internal: 8080) |

### Platform-Specific Settings

| Platform | `ATTN_IMPLEMENTATION` | Notes |
|----------|----------------------|-------|
| x86_64 (Ampere+: RTX 30xx, A100, ...) | `flash_attention_2` | Best performance |
| x86_64 (Pre-Ampere: GTX 16xx, RTX 20xx, ...) | `sdpa` | Required (Flash Attention not supported) |
| DGX Spark (ARM64) | `sdpa` | Required |

> **Note**: Flash Attention 2 requires NVIDIA Ampere architecture (SM 80) or newer. For pre-Ampere GPUs (Turing, Volta, etc.), use `sdpa`.

## GPU Requirements

| Model | VRAM (float16) | Notes |
|-------|----------------|-------|
| VibeVoice-ASR (7B) | ~14GB | STT only |
| VibeVoice-Realtime (0.5B) | ~2GB | Low-latency streaming TTS |
| VibeVoice-1.5B | ~4GB + ~1GB inference | High-quality TTS (6GB minimum, 8GB+ recommended) |
| ASR + 0.5B TTS | ~16GB+ | Consider separate deployment |
| ASR + 1.5B TTS | ~18GB+ | Consider separate deployment |

## Model Directory Structure

```
/path/to/models/
├── VibeVoice-ASR/                    # ASR model (~17GB)
│   ├── config.json
│   ├── model-00001-of-00008.safetensors
│   ├── ... (8 shards total)
│   └── model.safetensors.index.json
├── VibeVoice-Realtime-0.5B/         # TTS 0.5B streaming model (~1.9GB)
│   ├── config.json
│   ├── model.safetensors
│   └── ...
├── VibeVoice-1.5B/                   # TTS 1.5B full model (~5.4GB)
│   ├── config.json
│   ├── model-00001-of-00003.safetensors
│   ├── model-00002-of-00003.safetensors
│   ├── model-00003-of-00003.safetensors
│   └── model.safetensors.index.json
└── voices/
    ├── streaming_model/              # 0.5B voice presets (25x .pt)
    │   ├── en-Carter_man.pt
    │   ├── en-Davis_man.pt
    │   └── ...
    └── full_model/                   # 1.5B voice presets (9x .wav)
        ├── en-Alice_woman.wav
        ├── en-Carter_man.wav
        ├── en-Frank_man.wav
        └── ...
```

## Voice Presets

### Available Voices

| Model | Voice | File | Language |
|-------|-------|------|----------|
| 0.5B Streaming | carter, davis, emma, frank, grace, mike | `.pt` files | English |
| 0.5B Streaming | samuel | `.pt` file | Indian English |
| 0.5B Streaming | *(+ 18 more)* | `.pt` files | de, fr, it, jp, kr, nl, pl, pt, sp |
| 1.5B Full | alice, carter, frank, maya | `.wav` files | English |
| 1.5B Full | mary | `.wav` file | English (with BGM) |
| 1.5B Full | samuel | `.wav` file | Indian English |
| 1.5B Full | anchen | `.wav` file | Chinese (with BGM) |
| 1.5B Full | bowen, xinran | `.wav` files | Chinese |

### 1.5B Voice Preset Requirements

The 1.5B model voice presets (`.wav` files in `voices/full_model/`) must meet the following specifications:

| Property | Required Value |
|----------|---------------|
| Sample Rate | **24,000 Hz (24kHz)** |
| Channels | 1 (Mono) |
| Bit Depth | 16-bit PCM |
| Duration | **8 seconds recommended** (max ~9s for 6GB VRAM GPUs) |

> **Important**: The original voice samples from Microsoft's repository are 24kHz but 24~30 seconds long. These **must be trimmed to ~8 seconds** before use. Using the original long files will cause **CUDA Out of Memory** errors on GPUs with limited VRAM (e.g., 6GB). Voice audio is cached in CPU RAM after the first request and resampled to 24kHz automatically if needed.

#### Preparing Voice Presets

Use the following script to trim voice preset files to 8 seconds:

```python
import wave
import numpy as np
import os

voice_dir = '/path/to/models/voices/full_model'
TARGET_DURATION = 8.0  # seconds

for fname in sorted(os.listdir(voice_dir)):
    if not fname.endswith('.wav'):
        continue
    path = os.path.join(voice_dir, fname)
    with wave.open(path, 'rb') as w:
        rate = w.getframerate()
        total_frames = w.getnframes()
        duration = total_frames / rate
        raw = w.readframes(total_frames)

    if duration <= TARGET_DURATION:
        continue

    audio = np.frombuffer(raw, dtype=np.int16)
    max_samples = int(TARGET_DURATION * rate)
    audio = audio[:max_samples]

    with wave.open(path, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(audio.tobytes())
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
- [VibeVoice-1.5B on HuggingFace](https://huggingface.co/microsoft/VibeVoice) - Microsoft official model
- [shijincai/VibeVoice](https://github.com/shijincai/VibeVoice) - Restored 1.5B inference code
- [Open WebUI](https://docs.openwebui.com/)

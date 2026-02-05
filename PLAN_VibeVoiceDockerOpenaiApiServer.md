# VibeVoice Docker API Server 개발 계획

## 프로젝트 개요

MeloTTS-Docker-API-Server를 참고하여 VibeVoice 모델을 활용한 Open WebUI 호환 음성 API 서버를 구축하는 프로젝트입니다.

**지원 모델**:
- **STT**: VibeVoice-ASR (7B) — 음성→텍스트
- **TTS**: VibeVoice-Realtime-0.5B — 텍스트→음성 (실시간 스트리밍, 저지연)
- **TTS**: VibeVoice-1.5B — 텍스트→음성 (고품질 전체 모델)

> VibeVoice-TTS 1.5B 모델의 추론 코드가 Microsoft에 의해 제거되었으나(2025-09), shijincai fork에서 복원된 코드를 활용하여 지원합니다.

**구현 디렉터리**: `VibeVoiceDockerOpenaiApiServer/`

---

## 분석 결과 요약

### 1. MeloTTS-Docker-API-Server 분석

| 항목 | 내용 |
|------|------|
| 프레임워크 | FastAPI |
| 엔드포인트 | `POST /convert/tts` |
| 요청 형식 | `{"text": "...", "speed": ..., "language": "...", "speaker_id": "..."}` |
| 응답 형식 | WAV 파일 |
| Docker | Python 3.9 기반, MeloTTS 의존성 |
| **Open WebUI 호환** | ❌ (OpenAI TTS API 형식 아님) |

### 2. VibeVoice 분석 (본 프로젝트에서 사용하는 모델)

| 모델 | 크기 | 기능 | 상태 |
|------|------|------|------|
| **VibeVoice-ASR** | 7B (17.3GB) | 60분 긴 오디오 STT, 화자분리, 타임스탬프 | ✅ 사용 |
| **VibeVoice-Realtime-0.5B** | 0.5B (1.9GB) | 실시간 스트리밍 TTS (~200ms 지연), 25개 음성 프리셋 | ✅ 사용 |
| **VibeVoice-1.5B** | 1.5B (5.4GB) | 고품질 TTS, 9개 음성 프리셋 (.wav), CFG 기반 | ✅ 사용 |

> **듀얼 모델 지원**: `TTS_MODEL_TYPE` 환경 변수로 "0.5b", "1.5b", "both" 중 선택. 기본값은 "0.5b" (하위 호환).
> 1.5B 추론 코드는 [shijincai/VibeVoice](https://github.com/shijincai/VibeVoice) fork에서 복원.

### 3. Open WebUI API 규격

**TTS API** (OpenAI 호환):
- 엔드포인트: `POST /v1/audio/speech`
- 요청: `{"model": "...", "voice": "...", "input": "텍스트", "speed": 1.0}`
- 응답: 오디오 바이너리 (mp3/wav)

**STT API** (OpenAI 호환):
- 엔드포인트: `POST /v1/audio/transcriptions`
- 요청: `multipart/form-data` (file + model)
- 응답: `{"text": "인식된 텍스트"}`

### 4. Open WebUI API 규격 충족 방식

| 기능 | 사용 모델 | 네이티브 호환 | 본 프로젝트의 해결 |
|------|-----------|---------------|-------------------|
| STT | VibeVoice-ASR (7B) | ❌ (자체 API) | ✅ OpenAI 호환 REST 래퍼 구현 |
| TTS | VibeVoice-Realtime-0.5B | ❌ (WebSocket 전용) | ✅ OpenAI 호환 REST 래퍼 구현 |

**결론**: VibeVoice 모델은 Open WebUI API 규격을 직접 만족하지 않으므로, 본 프로젝트에서 OpenAI 호환 REST API 래퍼를 구현하여 해결.

---

## 모델 다운로드 및 설정 가이드

> **현재 상태**: 모든 모델 데이터 다운로드 완료 ✅ (2026-02-05)

### VibeVoice-ASR 모델 파일 구조 ✅

[HuggingFace microsoft/VibeVoice-ASR](https://huggingface.co/microsoft/VibeVoice-ASR)에서 다운로드해야 하는 파일 목록:

| 파일명 | 크기 | 설명 |
|--------|------|------|
| `config.json` | 3.52 KB | 모델 설정 파일 |
| `model-00001-of-00008.safetensors` | 2.49 GB | 모델 가중치 파트 1 |
| `model-00002-of-00008.safetensors` | 2.39 GB | 모델 가중치 파트 2 |
| `model-00003-of-00008.safetensors` | 2.47 GB | 모델 가중치 파트 3 |
| `model-00004-of-00008.safetensors` | 2.47 GB | 모델 가중치 파트 4 |
| `model-00005-of-00008.safetensors` | 2.50 GB | 모델 가중치 파트 5 |
| `model-00006-of-00008.safetensors` | 2.48 GB | 모델 가중치 파트 6 |
| `model-00007-of-00008.safetensors` | 1.46 GB | 모델 가중치 파트 7 |
| `model-00008-of-00008.safetensors` | 1.09 GB | 모델 가중치 파트 8 |
| `model.safetensors.index.json` | 120 KB | 모델 가중치 인덱스 |
| `README.md` | 2.56 KB | 모델 설명서 |

**총 크기**: 약 17.3 GB

### VibeVoice-Realtime-0.5B 모델 파일 구조 ✅

[HuggingFace microsoft/VibeVoice-Realtime-0.5B](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B)에서 다운로드:

| 파일명 | 크기 | 설명 |
|--------|------|------|
| `config.json` | 2.1 KB | 모델 설정 파일 |
| `preprocessor_config.json` | 360 B | 프로세서 설정 파일 |
| `model.safetensors` | 1.9 GB | 모델 가중치 |
| `README.md` | 10 KB | 모델 설명서 |

**총 크기**: 약 1.9 GB

> **참고**: VibeVoice-ASR 레포에는 `preprocessor_config.json`이 제공되지 않습니다.
> ASR 프로세서는 기본값 (`speech_tok_compress_ratio=3200`, `target_sample_rate=24000`)으로 폴백합니다.

### 다운로드 방법

#### 방법 1: HuggingFace CLI 사용 (권장)

```bash
# HuggingFace CLI 설치
pip install huggingface_hub

# 로그인 (토큰 필요 시)
huggingface-cli login

# 모델 전체 다운로드
huggingface-cli download microsoft/VibeVoice-ASR --local-dir /path/to/models/VibeVoice-ASR

# TTS 모델도 필요한 경우
huggingface-cli download microsoft/VibeVoice-Realtime-0.5B --local-dir /path/to/models/VibeVoice-Realtime-0.5B
```

#### 방법 2: Git LFS 사용

```bash
# Git LFS 설치
sudo apt-get install git-lfs
git lfs install

# 모델 클론
git clone https://huggingface.co/microsoft/VibeVoice-ASR /path/to/models/VibeVoice-ASR
git clone https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B /path/to/models/VibeVoice-Realtime-0.5B
```

#### 방법 3: Python 스크립트 사용

```python
from huggingface_hub import snapshot_download

# ASR 모델 다운로드
snapshot_download(
    repo_id="microsoft/VibeVoice-ASR",
    local_dir="/path/to/models/VibeVoice-ASR",
    local_dir_use_symlinks=False
)

# TTS 모델 다운로드
snapshot_download(
    repo_id="microsoft/VibeVoice-Realtime-0.5B",
    local_dir="/path/to/models/VibeVoice-Realtime-0.5B",
    local_dir_use_symlinks=False
)
```

### 모델 디렉터리 구조 (Docker 외부 - MODEL_DIR)

Docker 컨테이너는 `${MODEL_DIR}:/models:ro`로 마운트됩니다. MODEL_DIR은 다음 구조를 가져야 합니다:

```
MODEL_DIR/
├── VibeVoice-ASR/                   # ASR 모델 (17GB)
│   ├── config.json
│   ├── model-00001-of-00008.safetensors
│   ├── ... (8 shards)
│   ├── model.safetensors.index.json
│   └── README.md
├── VibeVoice-Realtime-0.5B/        # TTS 0.5B 스트리밍 모델 (1.9GB)
│   ├── config.json
│   ├── preprocessor_config.json
│   ├── model.safetensors
│   └── README.md
├── VibeVoice-1.5B/                  # TTS 1.5B 전체 모델 (5.4GB)
│   ├── config.json
│   ├── preprocessor_config.json
│   ├── model-00001-of-00003.safetensors
│   ├── model-00002-of-00003.safetensors
│   ├── model-00003-of-00003.safetensors
│   ├── model.safetensors.index.json
│   └── README.md
└── voices/
    ├── streaming_model/             # 0.5B 음성 프리셋 (25개 .pt)
    │   ├── en-Carter_man.pt
    │   ├── en-Davis_man.pt
    │   └── ... (총 25개, 10개 언어)
    └── full_model/                  # 1.5B 음성 프리셋 (9개 .wav)
        ├── en-Alice_woman.wav
        ├── en-Carter_man.wav
        ├── en-Frank_man.wav
        ├── en-Mary_woman_bgm.wav
        ├── en-Maya_woman.wav
        ├── in-Samuel_man.wav
        ├── zh-Anchen_man_bgm.wav
        ├── zh-Bowen_man.wav
        └── zh-Xinran_woman.wav
```

#### MODEL_DIR 심볼릭 링크 구성 예시

모델 데이터가 분산된 경우 심볼릭 링크로 구성할 수 있습니다:

```bash
BASE=/path/to/VibeVoiceDockerApiServer
mkdir -p $BASE/models/voices

ln -sfn $BASE/VibeVoice-ASR_Data      $BASE/models/VibeVoice-ASR
ln -sfn $BASE/VibeVoice-Realtime-0.5B $BASE/models/VibeVoice-Realtime-0.5B
ln -sfn $BASE/VibeVoice/demo/voices/streaming_model $BASE/models/voices/streaming_model
```

### 모델 다운로드 확인

```bash
# 파일 크기 확인
du -sh /path/to/models/VibeVoice-ASR
# 예상 출력: 약 17G

du -sh /path/to/models/VibeVoice-Realtime-0.5B
# 예상 출력: 약 1-2G
```

---

## 프로젝트 구조 (구현 완료)

```
VibeVoiceDockerOpenaiApiServer/
├── app/
│   ├── __init__.py              # 패키지 초기화, 버전 정보
│   ├── main.py                  # FastAPI 메인 앱, 라이프사이클 관리
│   ├── config.py                # pydantic-settings 기반 환경 변수 설정
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── tts.py               # POST /v1/audio/speech
│   │   └── stt.py               # POST /v1/audio/transcriptions, translations
│   ├── services/
│   │   ├── __init__.py
│   │   ├── tts_service.py       # VibeVoice-Realtime TTS 래퍼
│   │   └── stt_service.py       # VibeVoice-ASR 래퍼
│   └── models/
│       ├── __init__.py
│       ├── tts_models.py        # TTSRequest, TTSVoice 등
│       └── stt_models.py        # TranscriptionResponse, Segment 등
├── Dockerfile                   # 멀티 아키텍처 (amd64/arm64)
├── docker-compose.yml           # x86_64 기본 구성
├── docker-compose.dgx-spark.yml # DGX Spark (ARM64) 구성
├── docker-compose.with-openwebui.yml  # Open WebUI 통합 구성
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## 구현 작업 목록

### Phase 1: STT API (VibeVoice-ASR) ✅ 완료

| 작업 | 상태 | 구현 파일 |
|------|------|-----------|
| OpenAI 호환 `/v1/audio/transcriptions` 엔드포인트 | ✅ | `app/routes/stt.py` |
| multipart/form-data 요청 처리 | ✅ | `app/routes/stt.py` |
| file, model, language, response_format 파라미터 | ✅ | `app/routes/stt.py` |
| VibeVoice-ASR 모델 로딩 | ✅ | `app/services/stt_service.py` |
| VibeVoiceASRProcessor 초기화 | ✅ | `app/services/stt_service.py` |
| 다양한 오디오 포맷 지원 (librosa) | ✅ | `app/services/stt_service.py` |
| json 응답 형식 | ✅ | `app/services/stt_service.py` |
| text 응답 형식 | ✅ | `app/services/stt_service.py` |
| verbose_json 응답 형식 (타임스탬프, 화자) | ✅ | `app/services/stt_service.py` |
| srt 자막 형식 | ✅ | `app/services/stt_service.py` |
| vtt 자막 형식 | ✅ | `app/services/stt_service.py` |
| `/v1/audio/translations` 엔드포인트 | ✅ | `app/routes/stt.py` |

### Phase 2: TTS API (VibeVoice-Realtime) ✅ 완료

| 작업 | 상태 | 구현 파일 |
|------|------|-----------|
| OpenAI 호환 `/v1/audio/speech` 엔드포인트 | ✅ | `app/routes/tts.py` |
| JSON 요청 처리 (TTSRequest) | ✅ | `app/models/tts_models.py` |
| model, voice, input, speed, response_format 파라미터 | ✅ | `app/models/tts_models.py` |
| VibeVoice-Realtime 모델 로딩 | ✅ | `app/services/tts_service.py` |
| VibeVoiceStreamingProcessor 초기화 | ✅ | `app/services/tts_service.py` |
| 동기식 생성 (REST 변환) | ✅ | `app/services/tts_service.py` |
| 음성 프리셋 로딩 및 캐싱 | ✅ | `app/services/tts_service.py` |
| OpenAI 음성명 → VibeVoice 매핑 | ✅ | `app/services/tts_service.py` |
| wav 응답 형식 | ✅ | `app/services/tts_service.py` |
| mp3 응답 형식 (ffmpeg) | ✅ | `app/services/tts_service.py` |
| opus, aac, flac, pcm 응답 형식 | ✅ | `app/services/tts_service.py` |
| `/v1/audio/voices` 목록 엔드포인트 | ✅ | `app/routes/tts.py` |

### Phase 3: Docker화 (멀티 아키텍처 지원) ✅ 완료

| 작업 | 상태 | 구현 파일 |
|------|------|-----------|
| Dockerfile (x86_64 + ARM64) | ✅ | `Dockerfile` |
| docker-compose.yml (x86_64) | ✅ | `docker-compose.yml` |
| docker-compose.dgx-spark.yml (ARM64) | ✅ | `docker-compose.dgx-spark.yml` |
| docker-compose.with-openwebui.yml | ✅ | `docker-compose.with-openwebui.yml` |
| 환경 변수 설정 | ✅ | `.env.example` |
| 외부 모델 볼륨 마운트 | ✅ | `docker-compose*.yml` |
| Health check 엔드포인트 | ✅ | `app/main.py` |
| `/v1/models` 엔드포인트 | ✅ | `app/main.py` |

### Phase 4: 모델 데이터 및 환경 설정 ✅ 완료

| 작업 | 상태 | 비고 |
|------|------|------|
| VibeVoice-ASR 모델 다운로드 (17GB, 8 shards) | ✅ | `VibeVoice-ASR_Data/` |
| `model.safetensors.index.json` 다운로드 | ✅ | 샤드 인덱스 파일 |
| VibeVoice-Realtime-0.5B 모델 다운로드 (1.9GB) | ✅ | `VibeVoice-Realtime-0.5B/` |
| 음성 프리셋 확보 (25개 .pt 파일) | ✅ | `VibeVoice/demo/voices/streaming_model/` |
| MODEL_DIR 심볼릭 링크 구성 | ✅ | `models/` 디렉터리 |
| 기본 dtype을 float16으로 변경 | ✅ | VRAM 절약 및 호환성 |
| batch_size 설정 추가 | ✅ | 기본값 1 |

### Phase 5: Docker 빌드/실행 테스트 및 Open WebUI 연동 ✅ 완료 (일부 진행 중)

| 작업 | 상태 | 비고 |
|------|------|------|
| Open WebUI 설정 가이드 | ✅ | `README.md` 포함 |
| Docker 빌드 테스트 | ✅ | `nvcr.io/nvidia/pytorch:25.01-py3` 베이스, 빌드 성공 |
| Docker 실행 테스트 (TTS 1.5B) | ✅ | GTX 1660 Ti (6GB), 전체 API 테스트 6/6 통과 |
| GPU device_ids 설정 | ✅ | `nvidia-smi` 인덱스 기준 `device_ids: ['0']` |
| Attention 구현 호환성 수정 | ✅ | Pre-Ampere GPU: `flash_attention_2` → `sdpa` 변경 |
| 기본 호스트 포트 변경 | ✅ | `8080` → `8899` |
| run-test.sh 테스트 스크립트 검증 | ✅ | `--model 1.5b --skip-build` 등 옵션 동작 확인 |
| Open WebUI 통합 테스트 | ⏳ | STT/TTS 연동 테스트 필요 |
| 성능 최적화 | ⏳ | 테스트 결과에 따라 진행 |

---

## 구현된 API 엔드포인트

| 엔드포인트 | 메서드 | 설명 | 상태 |
|------------|--------|------|------|
| `/` | GET | API 정보 | ✅ |
| `/health` | GET | 헬스 체크 | ✅ |
| `/v1/models` | GET | 모델 목록 (OpenAI 호환) | ✅ |
| `/v1/audio/speech` | POST | TTS (텍스트→음성) | ✅ |
| `/v1/audio/transcriptions` | POST | STT (음성→텍스트) | ✅ |
| `/v1/audio/translations` | POST | 번역 (음성→영어 텍스트) | ✅ |
| `/v1/audio/voices` | GET | 사용 가능한 음성 목록 | ✅ |
| `/docs` | GET | Swagger UI | ✅ |
| `/redoc` | GET | ReDoc | ✅ |

---

## Docker 구성 (멀티 아키텍처 지원)

### 지원 플랫폼

| 플랫폼 | 아키텍처 | 베이스 이미지 | GPU |
|--------|----------|---------------|-----|
| **일반 x86_64 서버** | amd64 | `nvcr.io/nvidia/pytorch:25.01-py3` | NVIDIA CUDA |
| **NVIDIA DGX Spark** | arm64 (Grace CPU) | `nvcr.io/nvidia/pytorch:24.08-py3` | Grace Blackwell |

### 기본 추론 설정

| 항목 | 환경 변수 | 기본값 | 설명 |
|------|-----------|--------|------|
| ASR dtype | `ASR_DTYPE` | `float16` | ASR 모델 정밀도 |
| TTS dtype | `TTS_DTYPE` | `float16` | TTS 모델 정밀도 |
| Batch size | `BATCH_SIZE` | `1` | 추론 배치 크기 |
| Attention | `ATTN_IMPLEMENTATION` | `sdpa` | 기본값. Ampere+ GPU에서만 `flash_attention_2` 사용 가능 |
| Host 포트 | `API_PORT` | `8899` | 호스트 노출 포트 (컨테이너 내부: 8080) |

> **참고**: 기본 dtype은 `float16` (FP16)입니다. bfloat16이 필요한 경우 환경 변수로 오버라이드할 수 있습니다.
> **참고**: Flash Attention 2는 NVIDIA Ampere 아키텍처(SM 80) 이상에서만 지원됩니다. Pre-Ampere GPU(Turing: GTX 16xx/RTX 20xx, Volta: V100 등)에서는 반드시 `sdpa`를 사용해야 합니다.

### 실행 방법

#### x86_64 환경 (일반 NVIDIA GPU 서버)

```bash
# run-test.sh 사용 (권장) — 빌드/실행/테스트 자동 수행
cd VibeVoiceDockerOpenaiApiServer
MODEL_DIR=/path/to/your/models ./run-test.sh --model 1.5b

# 또는 docker compose 직접 사용
MODEL_DIR=/path/to/your/models docker compose up -d --build

# bfloat16으로 실행하려면
MODEL_DIR=/path/to/your/models ASR_DTYPE=bfloat16 TTS_DTYPE=bfloat16 docker compose up -d --build
```

#### DGX Spark 환경 (ARM64)

```bash
# DGX Spark 전용 compose 파일로 실행 (SDPA 자동 적용)
MODEL_DIR=/path/to/your/models docker compose -f docker-compose.dgx-spark.yml up -d --build
```

#### Open WebUI와 함께 실행

```bash
# Open WebUI 통합 실행
MODEL_DIR=/path/to/your/models docker compose -f docker-compose.with-openwebui.yml up -d --build
```

---

## 기술 스택

| 구분 | 기술 |
|------|------|
| 프레임워크 | FastAPI |
| 베이스 이미지 (x86_64) | `nvcr.io/nvidia/pytorch:25.01-py3` |
| 베이스 이미지 (DGX Spark) | `nvcr.io/nvidia/pytorch:24.08-py3` |
| ML 프레임워크 | PyTorch, Transformers |
| 설정 관리 | pydantic-settings |
| 오디오 처리 | librosa, scipy, ffmpeg |
| 주요 의존성 | vibevoice[tts,asr], flash-attn (x86_64), uvicorn, python-multipart |

### requirements.txt

```
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
python-multipart>=0.0.6
pydantic>=2.0.0
pydantic-settings>=2.0.0
python-dotenv>=1.0.0
pydub>=0.25.1
librosa>=0.10.0
scipy>=1.10.0
soundfile>=0.12.0
numpy>=1.24.0
```

---

## Open WebUI 설정 예시

### 환경 변수 설정

```bash
# TTS 설정
AUDIO_TTS_ENGINE=openai
AUDIO_TTS_OPENAI_API_BASE_URL=http://vibevoice-api:8899/v1
AUDIO_TTS_OPENAI_API_KEY=not-needed
AUDIO_TTS_MODEL=vibevoice-realtime
AUDIO_TTS_VOICE=carter

# STT 설정
AUDIO_STT_ENGINE=openai
AUDIO_STT_OPENAI_API_BASE_URL=http://vibevoice-api:8899/v1
AUDIO_STT_OPENAI_API_KEY=not-needed
AUDIO_STT_MODEL=vibevoice-asr
```

---

## 주의사항

### GPU 요구사항

| 모델 | VRAM (float16) | VRAM (bfloat16) | 비고 |
|------|----------------|-----------------|------|
| VibeVoice-ASR (7B) | ~14GB | ~17GB | float16 기본 |
| VibeVoice-Realtime (0.5B) | ~1-2GB | ~2-4GB | 경량 모델 |
| VibeVoice-1.5B | ~3-4GB | ~5-6GB | 고품질 모델 |
| **ASR + 0.5B** | ~16GB+ | ~20GB+ | 분리 배포 권장 |
| **ASR + 1.5B** | ~18GB+ | ~23GB+ | 분리 배포 권장 |
| **ASR + both TTS** | ~20GB+ | ~25GB+ | 대용량 GPU 필요 |

### 플랫폼별 고려사항

#### x86_64 (일반 NVIDIA GPU)
- **Ampere+ (SM 80+)**: RTX 30xx, A100, RTX 40xx 등 — `flash_attention_2` 사용 가능 (최적 성능)
- **Pre-Ampere (SM < 80)**: GTX 16xx, RTX 20xx, V100 등 — `sdpa` 사용 필수
- CUDA Compute Capability 7.0+ 권장

#### DGX Spark (ARM64 Grace Blackwell)
- Flash Attention 2 대신 **SDPA (Scaled Dot-Product Attention)** 사용
- 128GB 통합 메모리로 대용량 모델 로딩 가능
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 설정 권장
- Compute Capability sm_121 지원 PyTorch 필요

### 성능 고려사항

1. **정밀도 (dtype)**:
   - 기본: `float16` (FP16) — VRAM 절약, 대부분의 GPU 호환
   - 대안: `bfloat16` — 더 넓은 dynamic range, Ampere+ GPU 권장
   - 대안: `float32` — 최고 정밀도, VRAM 2배 사용

2. **Attention 구현**:
   - x86_64 Ampere+: `flash_attention_2` (최적 성능)
   - x86_64 Pre-Ampere: `sdpa` (필수, 기본값)
   - DGX Spark: `sdpa` (필수)

3. **Batch Size**: 기본값 `1`. 단일 요청 순차 처리.

4. **모델 분리 배포**: GPU 메모리 제약 시 ASR/TTS 서버 분리

5. **외부 모델 스토리지**:
   - 모델 파일은 Docker 외부에 저장하여 이미지 크기 최소화
   - NFS, NVMe SSD 등 고속 스토리지 권장

### 라이선스

- VibeVoice: MIT 라이선스
- 연구/개발 목적 권장 (상용 사용 시 추가 검토 필요)

---

## 참고 자료

### VibeVoice
- [VibeVoice GitHub](https://github.com/microsoft/VibeVoice)
- [VibeVoice-ASR HuggingFace](https://huggingface.co/microsoft/VibeVoice-ASR)
- [VibeVoice-Realtime HuggingFace](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B)

### Open WebUI
- [Open WebUI TTS Integration](https://docs.openwebui.com/features/audio/text-to-speech/openai-tts-integration/)
- [Open WebUI STT Integration](https://docs.openwebui.com/features/audio/speech-to-text/openai-stt-integration/)

### NVIDIA DGX Spark
- [DGX Spark User Guide](https://docs.nvidia.com/dgx/dgx-spark/dgx-spark.pdf)
- [DGX Spark NGC Guide](https://docs.nvidia.com/dgx/dgx-spark/ngc.html)
- [NVIDIA Container Runtime for Docker](https://docs.nvidia.com/dgx/dgx-spark/nvidia-container-runtime-for-docker.html)

### 기타
- [MeloTTS-Docker-API-Server](https://github.com/timhagel/MeloTTS-Docker-API-Server)
- [Docker Model Runner on DGX Spark](https://www.docker.com/blog/new-nvidia-dgx-spark-docker-model-runner/)

---

## 변경 이력

| 날짜 | 버전 | 내용 |
|------|------|------|
| 2026-01-26 | 1.0 | 초기 계획 수립 |
| 2026-01-26 | 1.1 | 모델 다운로드 가이드, DGX Spark 지원, 외부 모델 마운트 추가 |
| 2026-01-26 | 2.0 | **구현 완료**: Phase 1-3 완료, 프로젝트 구조 확정, 구현 상태 반영 |
| 2026-02-05 | 2.1 | **모델 데이터 완료**: ASR index 파일, Realtime-0.5B 모델 다운로드, MODEL_DIR 심볼릭 링크 구성 |
| 2026-02-05 | 2.2 | **기본 dtype 변경**: bfloat16 → float16 (FP16), batch_size=1 설정 추가, Phase 4 추가 |
| 2026-02-05 | 2.3 | **모델 지원 범위 명확화**: VibeVoice-TTS 1.5B 미지원 명시, 0.5B Realtime만 TTS 지원 |
| 2026-02-06 | 3.0 | **듀얼 모델 지원**: VibeVoice-1.5B TTS 추가 (MS 공식 모델 + shijincai fork 추론 코드), TTS_MODEL_TYPE 환경 변수, 모델별 라우팅, WAV 음성 프리셋 |
| 2026-02-06 | 3.1 | **Docker 빌드/실행 테스트 완료**: GTX 1660 Ti 환경에서 TTS 1.5B 전체 테스트 6/6 통과. GPU device_ids 설정(`['0']`), Attention `sdpa`로 변경(Pre-Ampere 호환), 기본 호스트 포트 `8899`로 변경, README/PLAN 문서 갱신 |

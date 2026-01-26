# VibeVoice Docker API Server 개발 계획

## 프로젝트 개요

MeloTTS-Docker-API-Server를 참고하여 VibeVoice-ASR을 활용한 Open WebUI 호환 음성 API 서버를 구축하는 프로젝트입니다.

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

### 2. VibeVoice 분석

| 모델 | 크기 | 기능 | 코드 상태 |
|------|------|------|-----------|
| **VibeVoice-ASR** | 7B (17.3GB) | 60분 긴 오디오 STT, 화자분리, 타임스탬프 | ✅ 사용가능 |
| **VibeVoice-TTS** | 1.5B | 90분 TTS, 다중화자 | ❌ 코드 제거됨 (2025-09) |
| **VibeVoice-Realtime** | 0.5B | 실시간 스트리밍 TTS (~200ms 지연) | ✅ 사용가능 |

**VibeVoice-TTS 코드 제거 이유**: Microsoft의 책임있는 AI 원칙에 따라 악용 방지를 위해 제거됨.

### 3. Open WebUI API 규격

**TTS API** (OpenAI 호환):
- 엔드포인트: `POST /v1/audio/speech`
- 요청: `{"model": "...", "voice": "...", "input": "텍스트", "speed": 1.0}`
- 응답: 오디오 바이너리 (mp3/wav)

**STT API** (OpenAI 호환):
- 엔드포인트: `POST /v1/audio/transcriptions`
- 요청: `multipart/form-data` (file + model)
- 응답: `{"text": "인식된 텍스트"}`

### 4. VibeVoice만으로 Open WebUI TTS API 규격 만족 여부

| 모델 | TTS 가능 여부 | Open WebUI 호환 |
|------|---------------|-----------------|
| VibeVoice-ASR | ❌ ASR 모델 | N/A |
| VibeVoice-TTS | ❌ 코드 제거됨 | N/A |
| VibeVoice-Realtime | ✅ TTS 가능 | ❌ WebSocket 전용 (REST API 래퍼 필요) |

**결론**: VibeVoice만으로는 Open WebUI API 규격을 **직접 만족하지 않음**. OpenAI 호환 REST API 래퍼를 구현해야 함.

---

## 모델 다운로드 및 설정 가이드

### VibeVoice-ASR 모델 파일 구조

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

### 모델 디렉터리 구조 (Docker 외부)

```
/path/to/models/
├── VibeVoice-ASR/
│   ├── config.json
│   ├── model-00001-of-00008.safetensors
│   ├── model-00002-of-00008.safetensors
│   ├── model-00003-of-00008.safetensors
│   ├── model-00004-of-00008.safetensors
│   ├── model-00005-of-00008.safetensors
│   ├── model-00006-of-00008.safetensors
│   ├── model-00007-of-00008.safetensors
│   ├── model-00008-of-00008.safetensors
│   ├── model.safetensors.index.json
│   └── README.md
├── VibeVoice-Realtime-0.5B/
│   ├── config.json
│   ├── model.safetensors
│   └── ...
└── voices/                          # TTS 음성 프리셋 (선택)
    └── streaming_model/
        ├── carter.pt
        ├── wayne.pt
        └── ...
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

### Phase 4: Open WebUI 연동 ⏳ 진행 중

| 작업 | 상태 | 비고 |
|------|------|------|
| Open WebUI 설정 가이드 | ✅ | `README.md` 포함 |
| 통합 테스트 | ⏳ | 실제 환경에서 테스트 필요 |
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

### 실행 방법

#### x86_64 환경 (일반 NVIDIA GPU 서버)

```bash
# 환경 변수 설정
export MODEL_DIR=/path/to/your/models

# 빌드 및 실행
cd VibeVoiceDockerOpenaiApiServer
docker compose up -d --build
```

#### DGX Spark 환경 (ARM64)

```bash
# 환경 변수 설정
export MODEL_DIR=/path/to/your/models

# DGX Spark 전용 compose 파일로 실행
docker compose -f docker-compose.dgx-spark.yml up -d --build
```

#### Open WebUI와 함께 실행

```bash
# 환경 변수 설정
export MODEL_DIR=/path/to/your/models

# Open WebUI 통합 실행
docker compose -f docker-compose.with-openwebui.yml up -d --build
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
AUDIO_TTS_OPENAI_API_BASE_URL=http://vibevoice-api:8080/v1
AUDIO_TTS_OPENAI_API_KEY=not-needed
AUDIO_TTS_MODEL=vibevoice-realtime
AUDIO_TTS_VOICE=carter

# STT 설정
AUDIO_STT_ENGINE=openai
AUDIO_STT_OPENAI_API_BASE_URL=http://vibevoice-api:8080/v1
AUDIO_STT_OPENAI_API_KEY=not-needed
AUDIO_STT_MODEL=vibevoice-asr
```

---

## 주의사항

### GPU 요구사항

| 모델 | VRAM 요구량 | 비고 |
|------|-------------|------|
| VibeVoice-ASR (7B) | ~17GB | bfloat16 기준 |
| VibeVoice-Realtime (0.5B) | ~2-4GB | 경량 모델 |
| **동시 로딩** | ~20GB+ | 분리 배포 권장 |

### 플랫폼별 고려사항

#### x86_64 (일반 NVIDIA GPU)
- Flash Attention 2 지원 (최적 성능)
- CUDA Compute Capability 7.0+ 권장 (V100, RTX 20xx 이상)

#### DGX Spark (ARM64 Grace Blackwell)
- Flash Attention 2 대신 **SDPA (Scaled Dot-Product Attention)** 사용
- 128GB 통합 메모리로 대용량 모델 로딩 가능
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 설정 권장
- Compute Capability sm_121 지원 PyTorch 필요

### 성능 고려사항

1. **Attention 구현**:
   - x86_64: `flash_attention_2` (권장)
   - DGX Spark: `sdpa` (필수)

2. **모델 분리 배포**: GPU 메모리 제약 시 ASR/TTS 서버 분리

3. **외부 모델 스토리지**:
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

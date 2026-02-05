# VibeVoice Docker API Server - 배포 및 테스트 가이드

## 사전 요구사항

| 항목 | 요구사항 |
|------|----------|
| OS | Linux (Ubuntu 20.04+, RHEL 8+ 등) |
| Docker | 24.0+ (Docker Engine + Docker Compose v2) |
| NVIDIA Driver | 535+ (CUDA 12.x) |
| NVIDIA Container Toolkit | `nvidia-ctk` 설치 필요 |
| GPU VRAM | 최소 4GB (TTS only) ~ 24GB+ (ASR + 양 TTS 모델) |
| 디스크 | ~25GB (모델 포함) |

## 디렉터리 구조

전체 `VibeVoiceDockerApiServer/` 디렉터리를 대상 서버로 복사합니다.

```
VibeVoiceDockerApiServer/
├── VibeVoice/                       # 라이브러리 소스 (Docker 빌드 시 사용)
├── VibeVoiceDockerOpenaiApiServer/  # API 서버 코드 + Docker 설정
│   ├── app/                         # FastAPI 앱
│   ├── Dockerfile
│   ├── docker-compose.yml           # x86_64 기본
│   ├── docker-compose.dgx-spark.yml # ARM64 DGX Spark
│   ├── docker-compose.with-openwebui.yml
│   ├── .env.example
│   ├── run-test.sh                  # 테스트 스크립트
│   └── DEPLOY.md                    # 이 문서
└── models/                          # 모델 가중치 (~24GB)
    ├── VibeVoice-ASR/               # STT 7B (17GB)
    ├── VibeVoice-Realtime-0.5B/     # TTS 0.5B (1.9GB)
    ├── VibeVoice-1.5B/              # TTS 1.5B (5.1GB)
    └── voices/
        ├── streaming_model/         # 0.5B 프리셋 (.pt, 25개)
        └── full_model/              # 1.5B 프리셋 (.wav, 9개)
```

## 빠른 시작

```bash
cd VibeVoiceDockerApiServer/VibeVoiceDockerOpenaiApiServer
./run-test.sh
```

## 수동 실행

### 1. Docker 빌드 및 실행

```bash
cd VibeVoiceDockerApiServer/VibeVoiceDockerOpenaiApiServer

# x86_64 환경 (기본)
MODEL_DIR=../models docker compose up -d --build

# DGX Spark (ARM64)
MODEL_DIR=../models docker compose -f docker-compose.dgx-spark.yml up -d --build
```

### 2. TTS 모델 선택

환경 변수 `TTS_MODEL_TYPE`으로 제어합니다:

| 값 | 설명 | VRAM |
|----|------|------|
| `0.5b` (기본) | 0.5B 스트리밍 모델만 로드 | ~2GB |
| `1.5b` | 1.5B 전체 모델만 로드 | ~4GB |
| `both` | 양쪽 모두 로드 | ~6GB |

```bash
# 1.5B 모델만 사용
MODEL_DIR=../models TTS_MODEL_TYPE=1.5b docker compose up -d --build

# 양쪽 모두
MODEL_DIR=../models TTS_MODEL_TYPE=both docker compose up -d --build

# TTS만 (ASR 비활성화, VRAM 절약)
MODEL_DIR=../models ASR_ENABLED=false docker compose up -d --build
```

### 3. API 테스트

```bash
# 헬스 체크
curl http://localhost:8080/health

# 모델 목록
curl http://localhost:8080/v1/models

# 음성 목록
curl http://localhost:8080/v1/audio/voices

# TTS - 0.5B (기본)
curl -X POST http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"vibevoice-realtime","input":"Hello, world!","voice":"carter"}' \
  -o test_05b.wav

# TTS - 1.5B
curl -X POST http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"vibevoice-1.5b","input":"Hello, world!","voice":"alice"}' \
  -o test_15b.wav

# STT
curl -X POST http://localhost:8080/v1/audio/transcriptions \
  -F "file=@test_05b.wav" \
  -F "model=vibevoice-asr"
```

### 4. 로그 확인

```bash
docker compose logs -f vibevoice-api
```

### 5. 중지

```bash
docker compose down
```

## 환경 변수 참고

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `MODEL_DIR` | `./models` | 모델 디렉터리 경로 (호스트) |
| `DEVICE` | `cuda` | 디바이스 (cuda/cpu) |
| `ATTN_IMPLEMENTATION` | `flash_attention_2` | Attention 구현 (DGX Spark: `sdpa`) |
| `ASR_ENABLED` | `true` | STT 서비스 활성화 |
| `TTS_ENABLED` | `true` | TTS 서비스 활성화 |
| `TTS_MODEL_TYPE` | `0.5b` | TTS 모델 선택 (`0.5b`/`1.5b`/`both`) |
| `ASR_DTYPE` | `float16` | ASR 정밀도 |
| `TTS_DTYPE` | `float16` | TTS 정밀도 |
| `API_PORT` | `8080` | API 포트 |

## 트러블슈팅

### NVIDIA Container Toolkit 미설치
```
docker: Error response from daemon: could not select device driver
```
→ `nvidia-ctk` 설치: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

### Flash Attention 빌드 실패 (ARM64)
자동으로 SDPA로 폴백합니다. DGX Spark에서는 `ATTN_IMPLEMENTATION=sdpa`가 자동 설정됩니다.

### VRAM 부족
- `ASR_ENABLED=false`로 ASR 비활성화
- `TTS_MODEL_TYPE=0.5b`로 경량 모델만 사용
- `TTS_DTYPE=float16` 확인

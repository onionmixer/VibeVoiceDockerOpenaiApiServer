#!/usr/bin/env bash
set -euo pipefail

#=============================================================================
# VibeVoice Docker API Server - Build, Run & Test Script
#
# Usage:
#   ./run-test.sh                  # 기본 (0.5b TTS only)
#   ./run-test.sh --model 1.5b     # 1.5B TTS only
#   ./run-test.sh --model both     # 양쪽 TTS 모델
#   ./run-test.sh --asr             # ASR 활성화 (기본: 비활성화)
#   ./run-test.sh --dgx-spark      # DGX Spark (ARM64)
#   ./run-test.sh --skip-build     # 빌드 건너뛰기 (이미 빌드됨)
#   ./run-test.sh --down           # 컨테이너 중지 및 제거
#=============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-$PROJECT_ROOT/models}"

# Defaults
TTS_MODEL_TYPE="${TTS_MODEL_TYPE:-0.5b}"
ASR_ENABLED="${ASR_ENABLED:-false}"
COMPOSE_FILE="docker-compose.yml"
SKIP_BUILD=false
DO_DOWN=false
API_PORT="${API_PORT:-8899}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            TTS_MODEL_TYPE="$2"; shift 2 ;;
        --asr)
            ASR_ENABLED=true; shift ;;
        --dgx-spark)
            COMPOSE_FILE="docker-compose.dgx-spark.yml"; shift ;;
        --skip-build)
            SKIP_BUILD=true; shift ;;
        --down)
            DO_DOWN=true; shift ;;
        --port)
            API_PORT="$2"; shift 2 ;;
        -h|--help)
            head -15 "$0" | tail -12; exit 0 ;;
        *)
            echo "Unknown option: $1"; exit 1 ;;
    esac
done

API_URL="http://localhost:${API_PORT}"

#-----------------------------------------------------------------------------
# Helper functions
#-----------------------------------------------------------------------------
log()  { echo -e "\033[1;34m[INFO]\033[0m $*"; }
ok()   { echo -e "\033[1;32m[PASS]\033[0m $*"; }
fail() { echo -e "\033[1;31m[FAIL]\033[0m $*"; }
sep()  { echo "────────────────────────────────────────────────────────────"; }

#-----------------------------------------------------------------------------
# Down mode
#-----------------------------------------------------------------------------
if $DO_DOWN; then
    log "Stopping containers..."
    cd "$SCRIPT_DIR"
    MODEL_DIR="$MODEL_DIR" docker compose -f "$COMPOSE_FILE" down
    log "Done."
    exit 0
fi

#-----------------------------------------------------------------------------
# Pre-flight checks
#-----------------------------------------------------------------------------
sep
log "VibeVoice Docker API Server - Test Runner"
sep
log "Project root : $PROJECT_ROOT"
log "Model dir    : $MODEL_DIR"
log "TTS model    : $TTS_MODEL_TYPE"
log "ASR enabled  : $ASR_ENABLED"
log "Compose file : $COMPOSE_FILE"
log "API port     : $API_PORT"
sep

# Check Docker
if ! command -v docker &>/dev/null; then
    fail "Docker not found. Install Docker first."
    exit 1
fi
log "Docker: $(docker --version)"

# Check NVIDIA runtime
if docker info 2>/dev/null | grep -q "nvidia"; then
    log "NVIDIA runtime: available"
elif command -v nvidia-smi &>/dev/null; then
    log "NVIDIA GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
else
    fail "NVIDIA GPU/runtime not detected. GPU is required."
    exit 1
fi

# Check model directory
if [ ! -d "$MODEL_DIR" ]; then
    fail "Model directory not found: $MODEL_DIR"
    exit 1
fi

CHECKS_OK=true
if [ "$ASR_ENABLED" = "true" ] && [ ! -d "$MODEL_DIR/VibeVoice-ASR" ]; then
    fail "ASR model not found: $MODEL_DIR/VibeVoice-ASR"
    CHECKS_OK=false
fi
if [ "$TTS_MODEL_TYPE" = "0.5b" ] || [ "$TTS_MODEL_TYPE" = "both" ]; then
    if [ ! -d "$MODEL_DIR/VibeVoice-Realtime-0.5B" ]; then
        fail "0.5B model not found: $MODEL_DIR/VibeVoice-Realtime-0.5B"
        CHECKS_OK=false
    fi
fi
if [ "$TTS_MODEL_TYPE" = "1.5b" ] || [ "$TTS_MODEL_TYPE" = "both" ]; then
    if [ ! -d "$MODEL_DIR/VibeVoice-1.5B" ]; then
        fail "1.5B model not found: $MODEL_DIR/VibeVoice-1.5B"
        CHECKS_OK=false
    fi
fi

if ! $CHECKS_OK; then
    fail "Required models missing. Aborting."
    exit 1
fi
ok "All model directories found."

#-----------------------------------------------------------------------------
# Build & Start
#-----------------------------------------------------------------------------
sep
cd "$SCRIPT_DIR"

if ! $SKIP_BUILD; then
    log "Building Docker image..."
    MODEL_DIR="$MODEL_DIR" \
    TTS_MODEL_TYPE="$TTS_MODEL_TYPE" \
    ASR_ENABLED="$ASR_ENABLED" \
    API_PORT="$API_PORT" \
        docker compose -f "$COMPOSE_FILE" build
    ok "Docker build complete."
fi

log "Starting container..."
MODEL_DIR="$MODEL_DIR" \
TTS_MODEL_TYPE="$TTS_MODEL_TYPE" \
ASR_ENABLED="$ASR_ENABLED" \
API_PORT="$API_PORT" \
    docker compose -f "$COMPOSE_FILE" up -d

#-----------------------------------------------------------------------------
# Wait for health
#-----------------------------------------------------------------------------
sep
log "Waiting for server to start (model loading may take 1-3 min)..."

MAX_WAIT=300
ELAPSED=0
while [ $ELAPSED -lt $MAX_WAIT ]; do
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$API_URL/health" 2>/dev/null || echo "000")
    if [ "$HTTP_CODE" = "200" ]; then
        break
    fi
    sleep 5
    ELAPSED=$((ELAPSED + 5))
    printf "\r  Waiting... %ds / %ds (HTTP: %s)" "$ELAPSED" "$MAX_WAIT" "$HTTP_CODE"
done
echo ""

if [ $ELAPSED -ge $MAX_WAIT ]; then
    fail "Server failed to start within ${MAX_WAIT}s"
    log "Container logs:"
    docker compose -f "$COMPOSE_FILE" logs --tail=50
    exit 1
fi
ok "Server is healthy."

#-----------------------------------------------------------------------------
# API Tests
#-----------------------------------------------------------------------------
sep
log "Running API tests..."
sep

TESTS_PASSED=0
TESTS_FAILED=0
OUTPUT_DIR=$(mktemp -d)

run_test() {
    local name="$1"
    shift
    if "$@" >/dev/null 2>&1; then
        ok "$name"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        fail "$name"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
}

# 1) Health check
run_test "GET /health" \
    curl -sf "$API_URL/health"

# 2) Models list
run_test "GET /v1/models" \
    curl -sf "$API_URL/v1/models"

# 3) Root endpoint
run_test "GET /" \
    curl -sf "$API_URL/"

# 4) Voices list
run_test "GET /v1/audio/voices" \
    curl -sf "$API_URL/v1/audio/voices"

# 5) TTS 0.5B
if [ "$TTS_MODEL_TYPE" = "0.5b" ] || [ "$TTS_MODEL_TYPE" = "both" ]; then
    log "Testing TTS 0.5B (vibevoice-realtime)..."
    TTS_05B_OUT="$OUTPUT_DIR/test_05b.wav"
    if curl -sf -X POST "$API_URL/v1/audio/speech" \
        -H "Content-Type: application/json" \
        -d '{"model":"vibevoice-realtime","input":"Hello, this is a test of the zero point five billion model.","voice":"carter"}' \
        -o "$TTS_05B_OUT" && [ -s "$TTS_05B_OUT" ]; then
        FILE_SIZE=$(stat -c%s "$TTS_05B_OUT" 2>/dev/null || stat -f%z "$TTS_05B_OUT" 2>/dev/null)
        ok "TTS 0.5B synthesis → $TTS_05B_OUT ($FILE_SIZE bytes)"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        fail "TTS 0.5B synthesis"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
fi

# 6) TTS 1.5B
if [ "$TTS_MODEL_TYPE" = "1.5b" ] || [ "$TTS_MODEL_TYPE" = "both" ]; then
    log "Testing TTS 1.5B (vibevoice-1.5b)..."
    TTS_15B_OUT="$OUTPUT_DIR/test_15b.wav"
    if curl -sf -X POST "$API_URL/v1/audio/speech" \
        -H "Content-Type: application/json" \
        -d '{"model":"vibevoice-1.5b","input":"Hello, this is a test of the one point five billion model.","voice":"alice"}' \
        -o "$TTS_15B_OUT" && [ -s "$TTS_15B_OUT" ]; then
        FILE_SIZE=$(stat -c%s "$TTS_15B_OUT" 2>/dev/null || stat -f%z "$TTS_15B_OUT" 2>/dev/null)
        ok "TTS 1.5B synthesis → $TTS_15B_OUT ($FILE_SIZE bytes)"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        fail "TTS 1.5B synthesis"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
fi

# 7) STT (use generated audio if available)
if [ "$ASR_ENABLED" = "true" ]; then
    STT_INPUT=""
    [ -f "$OUTPUT_DIR/test_05b.wav" ] && STT_INPUT="$OUTPUT_DIR/test_05b.wav"
    [ -z "$STT_INPUT" ] && [ -f "$OUTPUT_DIR/test_15b.wav" ] && STT_INPUT="$OUTPUT_DIR/test_15b.wav"

    if [ -n "$STT_INPUT" ]; then
        log "Testing STT (vibevoice-asr)..."
        STT_RESULT=$(curl -sf -X POST "$API_URL/v1/audio/transcriptions" \
            -F "file=@$STT_INPUT" \
            -F "model=vibevoice-asr" 2>/dev/null || echo "")
        if [ -n "$STT_RESULT" ]; then
            ok "STT transcription → $STT_RESULT"
            TESTS_PASSED=$((TESTS_PASSED + 1))
        else
            fail "STT transcription"
            TESTS_FAILED=$((TESTS_FAILED + 1))
        fi
    else
        log "Skipping STT test (no audio file generated)"
    fi
fi

# 8) Error handling test (empty input)
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$API_URL/v1/audio/speech" \
    -H "Content-Type: application/json" \
    -d '{"model":"vibevoice-realtime","input":"","voice":"carter"}' 2>/dev/null || echo "000")
if [ "$HTTP_CODE" = "400" ]; then
    ok "Empty input validation → HTTP $HTTP_CODE"
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    fail "Empty input validation → expected 400, got $HTTP_CODE"
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi

#-----------------------------------------------------------------------------
# Summary
#-----------------------------------------------------------------------------
sep
log "Test Results"
sep
echo ""
echo "  Passed: $TESTS_PASSED"
echo "  Failed: $TESTS_FAILED"
echo "  Output: $OUTPUT_DIR/"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    ok "All tests passed!"
else
    fail "$TESTS_FAILED test(s) failed."
fi

sep
log "Server is running at $API_URL"
log "Swagger docs: $API_URL/docs"
log "Stop: ./run-test.sh --down"
sep

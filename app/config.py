"""
Configuration management for VibeVoice API Server.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Device settings
    device: str = "cuda"
    attn_implementation: str = "flash_attention_2"  # or "sdpa" for DGX Spark

    # Model paths
    asr_model_path: str = "/models/VibeVoice-ASR"
    tts_model_path: str = "/models/VibeVoice-Realtime-0.5B"
    voices_path: str = "/models/voices"

    # Service toggles
    asr_enabled: bool = True
    tts_enabled: bool = True

    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8080

    # Model loading settings
    asr_dtype: str = "float16"  # float16, bfloat16, or float32
    tts_dtype: str = "float16"
    batch_size: int = 1  # Inference batch size

    # ASR settings
    asr_max_new_tokens: int = 32768
    asr_temperature: float = 0.0
    asr_top_p: float = 1.0

    # TTS settings (0.5B streaming model)
    tts_cfg_scale: float = 1.5
    tts_inference_steps: int = 5
    tts_sample_rate: int = 24000

    # TTS model type: "0.5b", "1.5b", or "both"
    tts_model_type: str = "0.5b"

    # 1.5B full model settings
    tts_1_5b_model_path: str = "/models/VibeVoice-1.5B"
    tts_1_5b_cfg_scale: float = 1.3
    tts_1_5b_inference_steps: int = 10

    # Default voice for TTS
    default_voice: str = "carter"

    class Config:
        env_prefix = ""
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_torch_dtype(dtype_str: str):
    """Convert string dtype to torch dtype."""
    import torch
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return dtype_map.get(dtype_str, torch.float16)

"""
TTS Service - VibeVoice-Realtime wrapper for OpenAI-compatible API.
"""

import os
import io
import copy
import tempfile
from typing import Optional, Dict, List, Iterator
from pathlib import Path

import numpy as np
import torch

from app.config import settings, get_torch_dtype


class TTSService:
    """
    Text-to-Speech service using VibeVoice-Realtime model.
    """

    def __init__(self):
        self.model = None
        self.processor = None
        self.device = None
        self.dtype = None
        self.voice_presets: Dict[str, Path] = {}
        self._voice_cache: Dict[str, object] = {}
        self._loaded = False
        self.sample_rate = settings.tts_sample_rate

    def load(self) -> None:
        """Load the TTS model and processor."""
        if self._loaded:
            return

        from vibevoice.modular.modeling_vibevoice_streaming_inference import (
            VibeVoiceStreamingForConditionalGenerationInference,
        )
        from vibevoice.processor.vibevoice_streaming_processor import (
            VibeVoiceStreamingProcessor,
        )

        model_path = settings.tts_model_path
        self.device = settings.device
        self.dtype = get_torch_dtype(settings.tts_dtype)
        attn_impl = settings.attn_implementation

        print(f"[TTS] Loading VibeVoice-Realtime from {model_path}")
        print(f"[TTS] Device: {self.device}, dtype: {self.dtype}, attn: {attn_impl}")

        # Load processor
        self.processor = VibeVoiceStreamingProcessor.from_pretrained(model_path)

        # Determine dtype and attention based on device
        if self.device == "mps":
            load_dtype = torch.float32
            attn_impl = "sdpa"
        elif self.device == "cpu":
            load_dtype = torch.float32
            attn_impl = "sdpa"
        else:
            load_dtype = self.dtype

        # Load model
        try:
            self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                model_path,
                torch_dtype=load_dtype,
                device_map=self.device if self.device in ("cuda", "cpu", "auto") else None,
                attn_implementation=attn_impl,
            )
        except Exception as e:
            print(f"[TTS] Failed with {attn_impl}, trying SDPA: {e}")
            self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                model_path,
                torch_dtype=load_dtype,
                device_map=self.device if self.device in ("cuda", "cpu", "auto") else None,
                attn_implementation="sdpa",
            )

        if self.device == "mps":
            self.model.to("mps")

        self.model.eval()
        self.model.set_ddpm_inference_steps(num_steps=settings.tts_inference_steps)

        # Load voice presets
        self._load_voice_presets()

        self._loaded = True
        print("[TTS] Model loaded successfully")

    def _load_voice_presets(self) -> None:
        """Load available voice presets from voices directory."""
        voices_dir = Path(settings.voices_path) / "streaming_model"

        if not voices_dir.exists():
            # Try relative to TTS model
            voices_dir = Path(settings.tts_model_path).parent / "voices" / "streaming_model"

        if not voices_dir.exists():
            print(f"[TTS] Warning: Voices directory not found at {voices_dir}")
            return

        for pt_file in voices_dir.rglob("*.pt"):
            name = pt_file.stem.lower()
            self.voice_presets[name] = pt_file
            print(f"[TTS] Found voice preset: {name}")

        # Map OpenAI voice names to available presets
        self._voice_mapping = {
            "alloy": "carter",
            "echo": "wayne",
            "fable": "carter",
            "onyx": "wayne",
            "nova": "carter",
            "shimmer": "wayne",
        }

    def _get_voice_path(self, voice: str) -> Optional[Path]:
        """Get voice preset path for given voice name."""
        voice_lower = voice.lower()

        # Direct match
        if voice_lower in self.voice_presets:
            return self.voice_presets[voice_lower]

        # OpenAI voice mapping
        if voice_lower in self._voice_mapping:
            mapped = self._voice_mapping[voice_lower]
            if mapped in self.voice_presets:
                return self.voice_presets[mapped]

        # Partial match
        for name, path in self.voice_presets.items():
            if voice_lower in name or name in voice_lower:
                return path

        # Fallback to first available
        if self.voice_presets:
            return next(iter(self.voice_presets.values()))

        return None

    def _load_voice_cache(self, voice: str) -> Optional[object]:
        """Load and cache voice preset."""
        voice_lower = voice.lower()

        if voice_lower in self._voice_cache:
            return self._voice_cache[voice_lower]

        voice_path = self._get_voice_path(voice)
        if voice_path is None:
            return None

        print(f"[TTS] Loading voice preset: {voice_path}")
        prefilled_outputs = torch.load(
            voice_path,
            map_location=self.device,
            weights_only=False,
        )
        self._voice_cache[voice_lower] = prefilled_outputs
        return prefilled_outputs

    def synthesize(
        self,
        text: str,
        voice: str = "carter",
        speed: float = 1.0,
        response_format: str = "wav",
    ) -> bytes:
        """
        Synthesize speech from text.

        Args:
            text: Text to synthesize
            voice: Voice preset name
            speed: Speech speed (currently not implemented for VibeVoice)
            response_format: Output format (wav, mp3, etc.)

        Returns:
            Audio bytes in requested format
        """
        if not self._loaded:
            raise RuntimeError("TTS model not loaded. Call load() first.")

        if not text.strip():
            raise ValueError("Input text cannot be empty")

        # Clean text
        text = text.replace("'", "'").replace('"', '"').replace('"', '"')

        # Get voice preset
        prefilled_outputs = self._load_voice_cache(voice)
        if prefilled_outputs is None:
            raise ValueError(f"Voice preset '{voice}' not found")

        # Prepare inputs
        inputs = self.processor.process_input_with_cached_prompt(
            text=text.strip(),
            cached_prompt=prefilled_outputs,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        # Move to device
        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=settings.tts_cfg_scale,
                tokenizer=self.processor.tokenizer,
                generation_config={"do_sample": False},
                verbose=False,
                all_prefilled_outputs=copy.deepcopy(prefilled_outputs),
            )

        # Get audio output
        if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
            audio = outputs.speech_outputs[0]
            if torch.is_tensor(audio):
                audio = audio.detach().cpu().to(torch.float32).numpy()
            else:
                audio = np.asarray(audio, dtype=np.float32)

            if audio.ndim > 1:
                audio = audio.reshape(-1)

            # Normalize
            peak = np.max(np.abs(audio)) if audio.size else 0.0
            if peak > 1.0:
                audio = audio / peak

            # Convert to requested format
            return self._convert_audio(audio, response_format)
        else:
            raise RuntimeError("No audio output generated")

    def _convert_audio(self, audio: np.ndarray, format: str) -> bytes:
        """Convert audio array to specified format."""
        import scipy.io.wavfile as wav_io

        # First create WAV
        audio_int16 = (audio * 32767).astype(np.int16)
        wav_buffer = io.BytesIO()
        wav_io.write(wav_buffer, self.sample_rate, audio_int16)
        wav_bytes = wav_buffer.getvalue()

        if format == "wav":
            return wav_bytes

        elif format == "pcm":
            # Raw PCM (16-bit signed, little-endian)
            return audio_int16.tobytes()

        elif format in ("mp3", "opus", "aac", "flac"):
            # Use ffmpeg for conversion
            return self._convert_with_ffmpeg(wav_bytes, format)

        else:
            # Default to WAV
            return wav_bytes

    def _convert_with_ffmpeg(self, wav_bytes: bytes, format: str) -> bytes:
        """Convert audio using ffmpeg."""
        import subprocess

        format_args = {
            "mp3": ["-f", "mp3", "-acodec", "libmp3lame", "-ab", "192k"],
            "opus": ["-f", "opus", "-acodec", "libopus", "-ab", "128k"],
            "aac": ["-f", "adts", "-acodec", "aac", "-ab", "192k"],
            "flac": ["-f", "flac", "-acodec", "flac"],
        }

        args = format_args.get(format, ["-f", "wav"])

        cmd = [
            "ffmpeg",
            "-f", "wav",
            "-i", "pipe:0",
            *args,
            "-ar", str(self.sample_rate),
            "-ac", "1",
            "pipe:1",
        ]

        try:
            result = subprocess.run(
                cmd,
                input=wav_bytes,
                capture_output=True,
                check=True,
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            print(f"[TTS] ffmpeg conversion failed: {e.stderr.decode()}")
            # Fallback to WAV
            return wav_bytes
        except FileNotFoundError:
            print("[TTS] ffmpeg not found, returning WAV")
            return wav_bytes

    def get_available_voices(self) -> List[str]:
        """Get list of available voice presets."""
        return list(self.voice_presets.keys())

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

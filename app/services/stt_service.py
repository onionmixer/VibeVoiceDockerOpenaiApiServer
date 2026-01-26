"""
STT Service - VibeVoice-ASR wrapper for OpenAI-compatible API.
"""

import os
import io
import time
import tempfile
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path

import numpy as np
import torch
import librosa

from app.config import settings, get_torch_dtype
from app.models.stt_models import (
    TranscriptionResponse,
    TranscriptionVerboseResponse,
    TranscriptionSegment,
)


class STTService:
    """
    Speech-to-Text service using VibeVoice-ASR model.
    """

    def __init__(self):
        self.model = None
        self.processor = None
        self.device = None
        self.dtype = None
        self._loaded = False

    def load(self) -> None:
        """Load the ASR model and processor."""
        if self._loaded:
            return

        from vibevoice.modular.modeling_vibevoice_asr import (
            VibeVoiceASRForConditionalGeneration,
        )
        from vibevoice.processor.vibevoice_asr_processor import (
            VibeVoiceASRProcessor,
        )

        model_path = settings.asr_model_path
        self.device = settings.device
        self.dtype = get_torch_dtype(settings.asr_dtype)
        attn_impl = settings.attn_implementation

        print(f"[STT] Loading VibeVoice-ASR from {model_path}")
        print(f"[STT] Device: {self.device}, dtype: {self.dtype}, attn: {attn_impl}")

        # Load processor
        self.processor = VibeVoiceASRProcessor.from_pretrained(
            model_path,
            language_model_pretrained_name="Qwen/Qwen2.5-7B"
        )

        # Load model
        try:
            self.model = VibeVoiceASRForConditionalGeneration.from_pretrained(
                model_path,
                dtype=self.dtype,
                device_map=self.device if self.device == "auto" else None,
                attn_implementation=attn_impl,
                trust_remote_code=True,
            )
        except Exception as e:
            print(f"[STT] Failed with {attn_impl}, trying SDPA: {e}")
            self.model = VibeVoiceASRForConditionalGeneration.from_pretrained(
                model_path,
                dtype=self.dtype,
                device_map=self.device if self.device == "auto" else None,
                attn_implementation="sdpa",
                trust_remote_code=True,
            )

        if self.device != "auto":
            self.model = self.model.to(self.device)

        self.model.eval()
        self._loaded = True
        print("[STT] Model loaded successfully")

    def _load_audio(self, audio_bytes: bytes, filename: str) -> Tuple[np.ndarray, int]:
        """
        Load audio from bytes, handling various formats.
        Returns (audio_array, sample_rate).
        """
        # Save to temp file for librosa to handle format detection
        suffix = Path(filename).suffix if filename else ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            # Load with librosa (handles format conversion)
            audio, sr = librosa.load(tmp_path, sr=24000, mono=True)
            return audio, sr
        finally:
            os.unlink(tmp_path)

    def _prepare_generation_config(
        self,
        max_new_tokens: int = None,
        temperature: float = None,
        top_p: float = None,
    ) -> dict:
        """Prepare generation configuration."""
        max_tokens = max_new_tokens or settings.asr_max_new_tokens
        temp = temperature if temperature is not None else settings.asr_temperature
        p = top_p if top_p is not None else settings.asr_top_p

        config = {
            "max_new_tokens": max_tokens,
            "pad_token_id": self.processor.pad_id,
            "eos_token_id": self.processor.tokenizer.eos_token_id,
        }

        if temp > 0:
            config["do_sample"] = True
            config["temperature"] = temp
            config["top_p"] = p
        else:
            config["do_sample"] = False

        return config

    def transcribe(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: str = "json",
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Transcribe audio to text.

        Args:
            audio_bytes: Raw audio file bytes
            filename: Original filename (for format detection)
            language: Language hint (optional)
            prompt: Context/prompt for transcription (optional)
            response_format: Output format (json, text, verbose_json, srt, vtt)
            temperature: Sampling temperature

        Returns:
            Transcription result dict
        """
        if not self._loaded:
            raise RuntimeError("STT model not loaded. Call load() first.")

        start_time = time.time()

        # Load and preprocess audio
        audio_array, sample_rate = self._load_audio(audio_bytes, filename)
        audio_duration = len(audio_array) / sample_rate

        # Process input
        inputs = self.processor(
            audio=audio_array,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
            add_generation_prompt=True,
        )

        # Move to device
        inputs = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        # Generate
        generation_config = self._prepare_generation_config(temperature=temperature)

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **generation_config)

        # Decode
        input_length = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0, input_length:]

        # Find EOS and trim
        eos_positions = (
            generated_ids == self.processor.tokenizer.eos_token_id
        ).nonzero(as_tuple=True)[0]
        if len(eos_positions) > 0:
            generated_ids = generated_ids[: eos_positions[0] + 1]

        generated_text = self.processor.decode(generated_ids, skip_special_tokens=True)

        # Parse segments if available
        try:
            segments_data = self.processor.post_process_transcription(generated_text)
        except Exception:
            segments_data = []

        generation_time = time.time() - start_time

        # Build response based on format
        if response_format == "text":
            return {"text": generated_text}

        elif response_format == "verbose_json":
            segments = []
            for i, seg in enumerate(segments_data):
                segments.append(
                    TranscriptionSegment(
                        id=i,
                        seek=0,
                        start=float(seg.get("start_time", 0) or 0),
                        end=float(seg.get("end_time", 0) or 0),
                        text=seg.get("text", ""),
                        tokens=[],
                        temperature=temperature,
                        avg_logprob=0.0,
                        compression_ratio=1.0,
                        no_speech_prob=0.0,
                        speaker=seg.get("speaker_id"),
                    )
                )

            return TranscriptionVerboseResponse(
                task="transcribe",
                language=language or "en",
                duration=audio_duration,
                text=generated_text,
                segments=segments,
            ).model_dump()

        elif response_format == "srt":
            return self._format_srt(segments_data, generated_text)

        elif response_format == "vtt":
            return self._format_vtt(segments_data, generated_text)

        else:  # json (default)
            return TranscriptionResponse(text=generated_text).model_dump()

    def _format_srt(self, segments: List[Dict], fallback_text: str) -> str:
        """Format transcription as SRT subtitle."""
        if not segments:
            return f"1\n00:00:00,000 --> 00:00:10,000\n{fallback_text}\n"

        lines = []
        for i, seg in enumerate(segments, 1):
            start = self._seconds_to_srt_time(float(seg.get("start_time", 0) or 0))
            end = self._seconds_to_srt_time(float(seg.get("end_time", 0) or 0))
            text = seg.get("text", "")
            speaker = seg.get("speaker_id", "")
            if speaker:
                text = f"[{speaker}] {text}"
            lines.append(f"{i}\n{start} --> {end}\n{text}\n")

        return "\n".join(lines)

    def _format_vtt(self, segments: List[Dict], fallback_text: str) -> str:
        """Format transcription as WebVTT subtitle."""
        if not segments:
            return f"WEBVTT\n\n00:00:00.000 --> 00:00:10.000\n{fallback_text}\n"

        lines = ["WEBVTT", ""]
        for seg in segments:
            start = self._seconds_to_vtt_time(float(seg.get("start_time", 0) or 0))
            end = self._seconds_to_vtt_time(float(seg.get("end_time", 0) or 0))
            text = seg.get("text", "")
            speaker = seg.get("speaker_id", "")
            if speaker:
                text = f"<v {speaker}>{text}"
            lines.append(f"{start} --> {end}")
            lines.append(text)
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _seconds_to_srt_time(seconds: float) -> str:
        """Convert seconds to SRT time format (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    @staticmethod
    def _seconds_to_vtt_time(seconds: float) -> str:
        """Convert seconds to VTT time format (HH:MM:SS.mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

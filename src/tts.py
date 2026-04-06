"""Platform-aware Kokoro TTS: mlx-audio on Apple Silicon, kokoro-onnx elsewhere."""

import os
import platform
import re
import sys
from pathlib import Path

import numpy as np

# Chinese Unicode ranges
_CHINESE_RE = re.compile(
    r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]"
)

DEFAULT_VOICE_EN = "af_heart"
DEFAULT_VOICE_ZH = "zf_xiaobei"


def _detect_voice(text: str) -> str:
    """Auto-detect language and return appropriate voice."""
    chinese_chars = len(_CHINESE_RE.findall(text))
    total_alpha = sum(1 for c in text if c.isalpha())
    if total_alpha == 0:
        return DEFAULT_VOICE_EN
    if chinese_chars / total_alpha > 0.3:
        return DEFAULT_VOICE_ZH
    return DEFAULT_VOICE_EN


def _is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


class TTSBackend:
    """Unified TTS interface."""

    sample_rate: int = 24000

    def generate(self, text: str, voice: str | None = None, speed: float = 1.1) -> np.ndarray:
        raise NotImplementedError


class MLXBackend(TTSBackend):
    """mlx-audio backend (Apple Silicon GPU via MLX)."""

    def __init__(self):
        from mlx_audio.tts.generate import load_model

        self._model = load_model("mlx-community/Kokoro-82M-bf16")
        self.sample_rate = self._model.sample_rate
        # Warmup: triggers pipeline init (phonemizer, spacy, etc.)
        list(self._model.generate(text="Hello", voice=DEFAULT_VOICE_EN, speed=1.0))

    def generate(self, text: str, voice: str | None = None, speed: float = 1.1) -> np.ndarray:
        if voice is None:
            voice = _detect_voice(text)
        lang_code = voice[0]  # first char of voice name: 'a'=English, 'z'=Chinese, etc.
        results = list(self._model.generate(text=text, voice=voice, speed=speed, lang_code=lang_code))
        return np.concatenate([np.array(r.audio) for r in results])


class ONNXBackend(TTSBackend):
    """kokoro-onnx backend (ONNX Runtime, CPU)."""

    def __init__(self):
        import kokoro_onnx
        from huggingface_hub import hf_hub_download

        model_path = hf_hub_download("fastrtc/kokoro-onnx", "kokoro-v1.0.onnx")
        voices_path = hf_hub_download("fastrtc/kokoro-onnx", "voices-v1.0.bin")

        self._model = kokoro_onnx.Kokoro(model_path, voices_path)
        self.sample_rate = 24000

    def generate(self, text: str, voice: str | None = None, speed: float = 1.1) -> np.ndarray:
        if voice is None:
            voice = _detect_voice(text)
        pcm, _sr = self._model.create(text, voice=voice, speed=speed)
        return pcm


def load() -> TTSBackend:
    """Load the best available TTS backend for this platform."""
    if _is_apple_silicon() and not os.environ.get("KOKORO_ONNX"):
        try:
            backend = MLXBackend()
            print(f"TTS: mlx-audio (Apple GPU, sample_rate={backend.sample_rate})")
            return backend
        except ImportError:
            print("TTS: mlx-audio not installed, falling back to kokoro-onnx")

    backend = ONNXBackend()
    print(f"TTS: kokoro-onnx (CPU, sample_rate={backend.sample_rate})")
    return backend

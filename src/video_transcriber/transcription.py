"""Local, original-language transcription with a lazily loaded faster-whisper model."""

from __future__ import annotations

import importlib
import inspect
import math
import time
from pathlib import Path
from typing import Callable

from .errors import DependencyError, TranscriptionError


def _error_hint(exc: Exception, *, device: str, offline: bool) -> str:
    detail = str(exc)
    lowered = detail.lower()
    if "metadata_errors" in lowered:
        hint = "PyAV is incompatible with faster-whisper. Run python -m pip install -r requirements.txt to restore compatible versions."
    elif isinstance(exc, MemoryError) or any(token in lowered for token in ("out of memory", "bad_alloc")):
        hint = "Try a smaller model (for example --model small) or close other memory-intensive programs."
    elif any(token in lowered for token in ("cuda", "cublas", "cudnn", "float16")) or device == "cuda":
        hint = "Check compatible NVIDIA CUDA/cuDNN libraries, or retry --device cpu --compute-type int8."
    elif offline:
        hint = "Offline mode requires a complete model in the chosen cache. Download it once without --offline."
    else:
        hint = (
            "Check the model name/cache and your network for the initial model download; "
            "CPU users can retry --device cpu --compute-type int8."
        )
    return f"{detail[:2000]} {hint}"


def transcribe_audio(
    audio_path: Path,
    *,
    model: str = "small",
    language: str | None = None,
    device: str = "cpu",
    compute_type: str = "int8",
    task: str = "transcribe",
    model_dir: Path | None = None,
    offline: bool = False,
    cpu_threads: int = 0,
    progress: Callable[[str], None] = print,
) -> dict:
    """Consume every ASR segment before returning a complete transcript.

    VAD skips non-speech and faster-whisper restores timestamps to the original
    audio timeline. Language detection per decoding window is enabled when the
    installed version supports it, unless the caller forces a language.
    ``translate`` is the explicit Whisper task that produces English text.
    """
    audio_path = Path(audio_path).expanduser().resolve()
    if not audio_path.is_file() or audio_path.stat().st_size == 0:
        raise TranscriptionError(f"Audio file is missing or empty: {audio_path}")
    if task not in {"transcribe", "translate"}:
        raise TranscriptionError("Task must be 'transcribe' or explicit English translation ('translate').")
    if device not in {"cpu", "cuda", "auto"}:
        raise TranscriptionError("Device must be cpu, cuda, or auto; Apple Metal/MPS is not supported.")
    if cpu_threads < 0:
        raise TranscriptionError("CPU thread count must be zero (automatic) or a positive integer.")
    if not str(model).strip():
        raise TranscriptionError("Choose a Whisper model, such as small, medium, or large-v3.")
    aliases = {"telugu": "te", "english": "en", "hindi": "hi", "auto": None}
    if language is not None:
        language = language.strip().lower()
        language = aliases.get(language, language) or None
    if str(model).endswith(".en") and language not in {None, "en"}:
        raise TranscriptionError("An English-only .en model cannot transcribe this language. Use small, medium, or large-v3.")
    try:
        whisper_model = importlib.import_module("faster_whisper").WhisperModel
    except (ImportError, OSError, RuntimeError) as exc:
        raise DependencyError(
            "faster-whisper or a native dependency is unavailable. Run "
            "python -m pip install -r requirements.txt in your Python 3.11+ environment. "
            f"Details: {exc}"
        ) from exc
    try:
        if model_dir is not None:
            model_dir = Path(model_dir).expanduser().resolve()
            model_dir.mkdir(parents=True, exist_ok=True)
        progress(f"Loading Whisper {model} on {device} ({compute_type})…")
        if not offline:
            progress("The first run may download model weights; later runs reuse the local cache.")
        engine = whisper_model(
            str(model), device=device, compute_type=compute_type, cpu_threads=cpu_threads,
            download_root=str(model_dir) if model_dir is not None else None,
            local_files_only=offline,
        )
    except Exception as exc:
        raise TranscriptionError(f"Could not load Whisper: {_error_hint(exc, device=device, offline=offline)}") from exc

    if language and language not in engine.supported_languages:
        raise TranscriptionError(f"Language '{language}' is unsupported by model '{model}'. Use a supported language code or a multilingual model.")
    parameters = inspect.signature(engine.transcribe).parameters
    multilingual_detection = language is None and "multilingual" in parameters and not str(model).endswith(".en")
    options = {
        "language": language,
        "task": task,
        "beam_size": 5,
        "vad_filter": True,
        "condition_on_previous_text": False,
        "without_timestamps": False,
    }
    if "multilingual" in parameters:
        options["multilingual"] = multilingual_detection
    if multilingual_detection:
        progress("Detecting language per audio window; mixed-language speech remains best-effort.")
    elif language is None:
        progress("Using initial language detection; this model/version may miss later language switches.")
    if task == "translate":
        progress("Explicit translation requested: transcript text will be in English.")
    progress("Recognizing speech locally; long recordings can take substantial time on CPU…")

    segments = []
    generated_segments = None
    try:
        generated_segments, info = engine.transcribe(str(audio_path), **options)
        duration = float(info.duration)
        last_report = time.monotonic()
        for segment in generated_segments:
            text = segment.text.strip()
            if not text:
                continue
            start, end = float(segment.start), float(segment.end)
            if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start:
                raise TranscriptionError("Whisper returned invalid timestamps; no complete transcript was saved.")
            segments.append({"start": start, "end": end, "text": text})
            if len(segments) == 1 or time.monotonic() - last_report >= 10:
                suffix = f" / {duration:.0f}s ({min(100, end / duration * 100):.0f}%)" if duration > 0 else ""
                progress(f"Transcribing: {end:.0f}s{suffix}, {len(segments)} speech segments")
                last_report = time.monotonic()
        if not segments:
            raise TranscriptionError(
                "No recognizable speech was found. Audio may be silent, music-only, too quiet, or unclear. "
                "No speech text was invented; try clearer media or a larger model."
            )
        result = {
            "language": info.language,
            "language_probability": float(info.language_probability),
            "duration": duration,
            "duration_after_vad": float(getattr(info, "duration_after_vad", duration)),
            "segments": segments,
            "model": str(model),
            "device": device,
            "compute_type": compute_type,
            "task": task,
            "multilingual_detection": multilingual_detection,
        }
    except TranscriptionError:
        raise
    except Exception as exc:
        raise TranscriptionError(
            "Transcription failed before completion; no partial transcript was exported. "
            + _error_hint(exc, device=device, offline=offline)
        ) from exc
    finally:
        if generated_segments is not None and hasattr(generated_segments, "close"):
            generated_segments.close()
    progress(f"Transcription complete: {len(segments)} segments; detected source language: {result['language']}.")
    return result

"""Inspect local media and atomically extract speech-ready audio using FFmpeg."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable

from .errors import AudioError, DependencyError


def check_ffmpeg() -> None:
    """Require both command-line tools on PATH before starting a download."""
    missing = [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]
    if missing:
        raise DependencyError(
            f"Missing {', '.join(missing)} on PATH. Install FFmpeg (which includes "
            "ffprobe), reopen your terminal, and see the platform instructions in README.md."
        )


def _local_file(path: Path) -> Path:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise AudioError(f"Media file does not exist or is not a regular file: {path}")
    if path.stat().st_size == 0:
        raise AudioError(f"Media file is empty: {path}")
    return path


def _positive_float(value: object) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) and number > 0 else 0.0
    except (ValueError, TypeError):
        return 0.0


def probe_audio(path: Path) -> dict:
    """Return the first audio stream's properties, rejecting unusable media."""
    path = _local_file(path)
    if shutil.which("ffprobe") is None:
        raise DependencyError("ffprobe is missing from PATH. Install FFmpeg; see README.md.")
    command = [
        "ffprobe", "-v", "error", "-protocol_whitelist", "file,pipe",
        "-select_streams", "a:0", "-show_entries",
        "stream=codec_name,sample_rate,channels,duration:format=duration",
        "-of", "json", str(path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=120)
    except subprocess.TimeoutExpired as exc:
        raise AudioError(f"ffprobe timed out inspecting {path.name}; check that the file is valid.") from exc
    except OSError as exc:
        raise AudioError(f"Cannot run ffprobe: {exc}") from exc
    if result.returncode:
        raise AudioError(f"Cannot read audio in {path.name}: {result.stderr.strip()[-2000:]}")
    try:
        data = json.loads(result.stdout)
        streams = data.get("streams", [])
        if not streams:
            raise AudioError(f"No audio stream found in {path.name}. Choose media containing speech.")
        stream = streams[0]
        duration = _positive_float(stream.get("duration")) or _positive_float(
            data.get("format", {}).get("duration")
        )
        sample_rate = int(stream.get("sample_rate", 0))
        channels = int(stream.get("channels", 0))
    except (ValueError, TypeError, AttributeError, IndexError) as exc:
        raise AudioError(f"ffprobe returned invalid audio metadata for {path.name}.") from exc
    if duration <= 0 or sample_rate <= 0 or channels <= 0:
        raise AudioError(f"Audio in {path.name} is empty or has no usable duration/sample format.")
    return {
        "duration": duration,
        "sample_rate": sample_rate,
        "channels": channels,
        "codec": stream.get("codec_name", "unknown"),
    }


def extract_audio(source: Path, target: Path, progress: Callable[[str], None] = print) -> dict:
    """Write mono 16 kHz PCM WAV; interrupted/failed runs never replace a good file.

    RF64 is enabled automatically when the audio exceeds the RIFF size limit.
    FFmpeg writes progress to a pipe and diagnostics to a temporary file so neither
    pipe can fill up while processing long media.
    """
    check_ffmpeg()
    source = _local_file(source)
    target = Path(target).expanduser().resolve()
    if source == target:
        raise AudioError("Audio output must be different from the original media file.")
    source_info = probe_audio(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    progress(f"Extracting audio from {source.name} (mono, 16 kHz PCM WAV)…")
    temporary: Path | None = None
    process = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{target.stem}-", suffix=".wav", dir=target.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
        command = [
            "ffmpeg", "-hide_banner", "-nostdin", "-y", "-loglevel", "error",
            "-protocol_whitelist", "file,pipe", "-i", str(source), "-map", "0:a:0",
            "-vn", "-sn", "-dn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            "-rf64", "auto", "-f", "wav", "-progress", "pipe:1", "-nostats", str(temporary),
        ]
        with tempfile.TemporaryFile(mode="w+b") as diagnostics:
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=diagnostics, text=True, bufsize=1
            )
            last_report = time.monotonic()
            assert process.stdout is not None
            try:
                for line in process.stdout:
                    key, _, value = line.strip().partition("=")
                    if key == "out_time_us" and time.monotonic() - last_report >= 5:
                        seconds = _positive_float(value) / 1_000_000
                        fraction = min(100, seconds / source_info["duration"] * 100)
                        progress(f"Extracting audio: {fraction:.0f}% ({seconds:.0f} seconds processed)")
                        last_report = time.monotonic()
                returncode = process.wait()
            finally:
                # In particular, KeyboardInterrupt must stop FFmpeg before deleting its output.
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                process.stdout.close()
            if returncode:
                diagnostics.seek(0, 2)
                length = diagnostics.tell()
                diagnostics.seek(max(0, length - 4000))
                detail = diagnostics.read().decode("utf-8", errors="replace").strip()
                raise AudioError(f"FFmpeg could not extract audio from {source.name}: {detail}")
        info = probe_audio(temporary)
        if info["sample_rate"] != 16000 or info["channels"] != 1 or info["codec"] != "pcm_s16le":
            raise AudioError("FFmpeg produced an unexpected audio format; the original media was kept.")
        temporary.replace(target)
        progress(f"Audio ready: {target}")
        return info
    except OSError as exc:
        raise AudioError(f"Unable to extract/save audio: {exc}. Check disk space and file permissions.") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

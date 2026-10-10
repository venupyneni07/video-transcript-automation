"""Offline checks including genuinely generated local audio and real FFmpeg."""

import io
import json
import math
import shutil
import struct
import subprocess
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from video_transcriber import audio
from video_transcriber.errors import AudioError, DependencyError


HAS_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


@pytest.fixture
def stereo_audio(tmp_path):
    """A locally synthesized tone, not a downloaded or personal recording."""
    path = tmp_path / "a tone; $(literal).wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(44100)
        samples = (
            struct.pack("<hh", value, value)
            for value in (int(4000 * math.sin(2 * math.pi * 440 * n / 44100)) for n in range(22050))
        )
        handle.writeframes(b"".join(samples))
    return path


@pytest.mark.skipif(not HAS_FFMPEG, reason="FFmpeg and ffprobe are not installed")
def test_real_extraction_produces_whisper_format(stereo_audio, tmp_path):
    target = tmp_path / "output" / "audio.wav"
    messages = []
    result = audio.extract_audio(stereo_audio, target, progress=messages.append)
    assert result["sample_rate"] == 16000
    assert result["channels"] == 1
    assert result["codec"] == "pcm_s16le"
    assert result["duration"] == pytest.approx(0.5, abs=0.01)
    assert stereo_audio.is_file()
    assert target.is_file()
    assert not list(target.parent.glob(".audio-*.wav"))
    assert any("Audio ready" in message for message in messages)


@pytest.mark.skipif(not HAS_FFMPEG, reason="FFmpeg and ffprobe are not installed")
def test_video_without_audio_is_rejected(tmp_path):
    video = tmp_path / "silent-video.mp4"
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "color=black:s=16x16:d=0.1", "-an", str(video)],
        check=True, capture_output=True,
    )
    with pytest.raises(AudioError, match="No audio stream"):
        audio.probe_audio(video)


@pytest.mark.skipif(not HAS_FFMPEG, reason="FFmpeg and ffprobe are not installed")
def test_zero_frame_wave_is_rejected(tmp_path):
    path = tmp_path / "empty.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"")
    with pytest.raises(AudioError, match="empty|usable"):
        audio.probe_audio(path)


def test_missing_tools_have_install_instructions(monkeypatch):
    monkeypatch.setattr(audio.shutil, "which", lambda name: None)
    with pytest.raises(DependencyError, match="ffmpeg, ffprobe.*Install FFmpeg"):
        audio.check_ffmpeg()


@pytest.mark.parametrize("exists", [False, True])
def test_missing_and_zero_byte_files_are_rejected(tmp_path, exists):
    path = tmp_path / "empty.wav"
    if exists:
        path.touch()
    with pytest.raises(AudioError):
        audio.probe_audio(path)


def test_probe_passes_literal_paths_without_shell(monkeypatch, stereo_audio):
    commands = []
    monkeypatch.setattr(audio.shutil, "which", lambda name: name)

    def run(command, **kwargs):
        commands.append(command)
        assert "shell" not in kwargs
        return SimpleNamespace(returncode=0, stderr="", stdout=json.dumps({
            "streams": [{"sample_rate": "44100", "channels": 2, "codec_name": "pcm_s16le"}],
            "format": {"duration": "0.5"},
        }))

    monkeypatch.setattr(audio.subprocess, "run", run)
    assert audio.probe_audio(stereo_audio)["duration"] == 0.5
    assert commands[0][-1] == str(stereo_audio)


@pytest.mark.parametrize("stdout", ["bad json", '{"streams": []}', '{"streams": [{"sample_rate": "NaN"}]}'])
def test_invalid_probe_results_fail_cleanly(monkeypatch, stereo_audio, stdout):
    monkeypatch.setattr(audio.shutil, "which", lambda name: name)
    monkeypatch.setattr(audio.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout=stdout, stderr=""))
    with pytest.raises(AudioError):
        audio.probe_audio(stereo_audio)


def test_probe_timeout_is_clear(monkeypatch, stereo_audio):
    monkeypatch.setattr(audio.shutil, "which", lambda name: name)

    def run(*args, **kwargs):
        raise subprocess.TimeoutExpired("ffprobe", 120)

    monkeypatch.setattr(audio.subprocess, "run", run)
    with pytest.raises(AudioError, match="timed out"):
        audio.probe_audio(stereo_audio)


def test_conversion_failure_preserves_previous_audio(monkeypatch, stereo_audio, tmp_path):
    target = tmp_path / "audio.wav"
    target.write_bytes(b"previous successful output")
    monkeypatch.setattr(audio, "check_ffmpeg", lambda: None)
    monkeypatch.setattr(audio, "probe_audio", lambda path: {"duration": 1})

    class Process:
        stdout = io.StringIO("progress=end\n")

        def __init__(self, command, **kwargs):
            assert "shell" not in kwargs
            Path(command[-1]).write_bytes(b"partial")
            kwargs["stderr"].write(b"decoder rejected input")

        def wait(self, **kwargs):
            return 1

        def poll(self):
            return 1

    monkeypatch.setattr(audio.subprocess, "Popen", Process)
    with pytest.raises(AudioError, match="decoder rejected"):
        audio.extract_audio(stereo_audio, target, progress=lambda text: None)
    assert target.read_bytes() == b"previous successful output"
    assert not list(tmp_path.glob(".audio-*.wav"))


def test_interruption_stops_process_and_removes_partial(monkeypatch, stereo_audio, tmp_path):
    target = tmp_path / "audio.wav"
    monkeypatch.setattr(audio, "check_ffmpeg", lambda: None)
    monkeypatch.setattr(audio, "probe_audio", lambda path: {"duration": 1})
    state = {"stopped": False}

    class InterruptingStream:
        def __iter__(self):
            raise KeyboardInterrupt

        def close(self):
            pass

    class Process:
        stdout = InterruptingStream()

        def __init__(self, command, **kwargs):
            Path(command[-1]).write_bytes(b"partial")

        def poll(self):
            return None

        def terminate(self):
            state["stopped"] = True

        def wait(self, **kwargs):
            return -15

    monkeypatch.setattr(audio.subprocess, "Popen", Process)
    with pytest.raises(KeyboardInterrupt):
        audio.extract_audio(stereo_audio, target, progress=lambda text: None)
    assert state["stopped"]
    assert not target.exists()
    assert not list(tmp_path.glob(".audio-*.wav"))


def test_source_cannot_be_replaced(monkeypatch, stereo_audio):
    monkeypatch.setattr(audio, "check_ffmpeg", lambda: None)
    with pytest.raises(AudioError, match="different"):
        audio.extract_audio(stereo_audio, stereo_audio)

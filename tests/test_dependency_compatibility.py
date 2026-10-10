"""Exercise the actual decoder boundary without loading model weights."""

import wave

import pytest


def test_installed_whisper_can_decode_generated_wav(tmp_path):
    decoder = pytest.importorskip("faster_whisper.audio")
    path = tmp_path / "silence.wav"
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\x00\x00" * 1600)
    samples = decoder.decode_audio(str(path), sampling_rate=16000)
    assert samples.shape == (1600,)
    assert samples.dtype.name == "float32"
    assert not samples.any()

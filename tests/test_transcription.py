"""Model-independent transcription tests: never download models or call the network."""

import sys
from types import SimpleNamespace

import pytest

from video_transcriber import transcription
from video_transcriber.errors import DependencyError, TranscriptionError


@pytest.fixture
def audio_path(tmp_path):
    path = tmp_path / "audio.wav"
    path.write_bytes(b"mock audio; decoder is mocked in these unit tests")
    return path


@pytest.fixture
def model_stub(monkeypatch):
    state = {"segments": [
        SimpleNamespace(start=1.25, end=3.5, text="  నమస్కారం friends  "),
        SimpleNamespace(start=12.0, end=13.5, text="हिंदी and English."),
    ], "consumed": False}

    class Model:
        supported_languages = ["te", "en", "hi"]

        def __init__(self, name, **kwargs):
            state["model"] = name
            state["load_options"] = kwargs

        def transcribe(self, path, *, multilingual=False, **kwargs):
            state["options"] = {"multilingual": multilingual, **kwargs}
            state["path"] = path

            def generate():
                yield from state["segments"]
                if state.get("error"):
                    raise state["error"]
                state["consumed"] = True

            return generate(), SimpleNamespace(language="te", language_probability=0.93, duration=16.0, duration_after_vad=5.0)

    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=Model))
    state["class"] = Model
    return state


def test_original_languages_and_timestamps_survive(audio_path, model_stub):
    result = transcription.transcribe_audio(audio_path, progress=lambda text: None)
    assert model_stub["consumed"]
    assert result["segments"] == [
        {"start": 1.25, "end": 3.5, "text": "నమస్కారం friends"},
        {"start": 12.0, "end": 13.5, "text": "हिंदी and English."},
    ]
    assert result["language"] == "te"
    assert result["language_probability"] == 0.93
    assert result["duration"] == 16.0
    assert result["multilingual_detection"] is True
    assert model_stub["options"]["language"] is None
    assert model_stub["options"]["task"] == "transcribe"
    assert model_stub["options"]["vad_filter"] is True
    assert model_stub["load_options"]["device"] == "cpu"
    assert model_stub["load_options"]["compute_type"] == "int8"


@pytest.mark.parametrize(("name", "code"), [("Telugu", "te"), ("Hindi", "hi"), ("English", "en"), ("te", "te")])
def test_forced_language_disables_window_detection(audio_path, model_stub, name, code):
    transcription.transcribe_audio(audio_path, language=name, progress=lambda text: None)
    assert model_stub["options"]["language"] == code
    assert model_stub["options"]["multilingual"] is False


def test_explicit_translation_only(audio_path, model_stub):
    result = transcription.transcribe_audio(audio_path, task="translate", progress=lambda text: None)
    assert model_stub["options"]["task"] == "translate"
    assert result["task"] == "translate"


def test_model_cache_offline_and_device_options(audio_path, model_stub, tmp_path):
    cache = tmp_path / "cache"
    result = transcription.transcribe_audio(
        audio_path, model="medium", device="cuda", compute_type="float16",
        model_dir=cache, offline=True, cpu_threads=2, progress=lambda text: None,
    )
    assert cache.is_dir()
    assert model_stub["model"] == "medium"
    assert model_stub["load_options"] == {
        "device": "cuda", "compute_type": "float16", "cpu_threads": 2,
        "download_root": str(cache), "local_files_only": True,
    }
    assert result["model"] == "medium"


def test_lazy_generator_failure_does_not_return_partial(audio_path, model_stub):
    model_stub["error"] = RuntimeError("model decoder failure")
    with pytest.raises(TranscriptionError, match="before completion; no partial transcript"):
        transcription.transcribe_audio(audio_path, progress=lambda text: None)
    assert model_stub["consumed"] is False


def test_silence_does_not_invent_text(audio_path, model_stub):
    model_stub["segments"] = []
    with pytest.raises(TranscriptionError, match="No recognizable speech"):
        transcription.transcribe_audio(audio_path, progress=lambda text: None)


def test_blank_segments_are_ignored(audio_path, model_stub):
    model_stub["segments"].append(SimpleNamespace(start=15, end=16, text="  \n "))
    result = transcription.transcribe_audio(audio_path, progress=lambda text: None)
    assert len(result["segments"]) == 2


@pytest.mark.parametrize(("start", "end"), [(-1, 1), (2, 1), (float("nan"), 1), (1, float("inf"))])
def test_invalid_model_timestamps_fail(audio_path, model_stub, start, end):
    model_stub["segments"] = [SimpleNamespace(start=start, end=end, text="Words")]
    with pytest.raises(TranscriptionError, match="invalid timestamps"):
        transcription.transcribe_audio(audio_path, progress=lambda text: None)


def test_old_model_api_has_explicit_mixed_language_limitation(audio_path, model_stub, monkeypatch):
    original = model_stub["class"].transcribe

    def old_transcribe(self, path, **kwargs):
        assert "multilingual" not in kwargs
        return original(self, path, **kwargs)

    monkeypatch.setattr(model_stub["class"], "transcribe", old_transcribe)
    messages = []
    result = transcription.transcribe_audio(audio_path, progress=messages.append)
    assert result["multilingual_detection"] is False
    assert any("may miss later language switches" in text for text in messages)


def test_missing_package_has_actionable_error(audio_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    with pytest.raises(DependencyError, match="pip install -r requirements.txt"):
        transcription.transcribe_audio(audio_path)


@pytest.mark.parametrize(("kwargs", "message"), [
    ({"task": "summarize"}, "Task must"),
    ({"device": "mps"}, "Device must"),
    ({"cpu_threads": -1}, "CPU thread"),
    ({"model": "small.en", "language": "te"}, "English-only"),
    ({"model": ""}, "Choose a Whisper"),
])
def test_bad_configuration_is_rejected(audio_path, kwargs, message):
    with pytest.raises(TranscriptionError, match=message):
        transcription.transcribe_audio(audio_path, **kwargs)


def test_unsupported_language_is_actionable(audio_path, model_stub):
    with pytest.raises(TranscriptionError, match="unsupported"):
        transcription.transcribe_audio(audio_path, language="not-a-language", progress=lambda text: None)


def test_missing_audio_is_rejected(tmp_path):
    with pytest.raises(TranscriptionError, match="missing or empty"):
        transcription.transcribe_audio(tmp_path / "absent.wav")


@pytest.mark.parametrize(("error", "kwargs", "hint"), [
    (RuntimeError("CUDA library absent"), {"device": "cuda"}, "--device cpu"),
    (RuntimeError("model not cached"), {"offline": True}, "Offline mode requires"),
    (MemoryError("out of memory"), {}, "smaller model"),
    (RuntimeError("connection failed"), {}, "network"),
])
def test_load_errors_explain_recovery(audio_path, monkeypatch, error, kwargs, hint):
    def failing_model(*args, **kwargs):
        raise error

    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=failing_model))
    with pytest.raises(TranscriptionError, match=hint):
        transcription.transcribe_audio(audio_path, progress=lambda text: None, **kwargs)


def test_keyboard_interrupt_is_not_hidden(audio_path, model_stub):
    model_stub["error"] = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        transcription.transcribe_audio(audio_path, progress=lambda text: None)

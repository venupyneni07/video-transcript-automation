import json
import re
from pathlib import Path

import pytest

from video_transcriber.errors import TranscriptionError
from video_transcriber import exporter
from video_transcriber.exporter import export_transcript, sanitize_title, timestamp


def test_export_preserves_all_original_language_speech(tmp_path):
    result = {"language": "te", "duration": 3605.25, "task": "transcribe", "segments": [
        {"start": 1.2, "end": 5.004, "text": "నమస్కారం Hello **friends**."},
        {"start": 3601, "end": 3605.25, "text": "हिंदी మాట్లాడుతున్నాము."},
    ]}
    paths = export_transcript(tmp_path, {"title": "Test <video>", "source": "YouTube",
                                        "url": "https://www.youtube.com/watch?v=abcdefghijk"}, result)
    assert paths["text"].read_text() == "నమస్కారం Hello **friends**.\n\nहिंदी మాట్లాడుతున్నాము.\n"
    markdown = paths["markdown"].read_text()
    assert "**Language:** Telugu" in markdown
    assert "[01:00:01]" in markdown
    assert r"\*\*friends\*\*" in markdown
    subtitles = paths["subtitles"].read_text()
    assert "00:00:01,200 --> 00:00:05,004" in subtitles
    assert "01:00:01,000 --> 01:00:05,250" in subtitles
    metadata = json.loads(paths["metadata"].read_text())
    assert metadata["segments"] == result["segments"]
    assert all(path.is_absolute() for path in paths.values())


@pytest.mark.parametrize("title", ["../../danger:*?", "CON", "", "తెలుగు" * 200, "a\x00b"])
def test_safe_titles_are_single_bounded_components(title):
    safe = sanitize_title(title)
    assert safe and len(safe.encode()) <= 110
    assert not re.search(r'[<>:"/\\|?*\x00]', safe)
    assert safe.upper() != "CON"


def test_timestamp_carry_and_srt_precision():
    assert timestamp(59.9999, srt=True) == "00:01:00,000"
    assert timestamp(360000, srt=True) == "100:00:00,000"


def test_empty_recognition_does_not_replace_old_transcript(tmp_path):
    old = tmp_path / "transcript.txt"
    old.write_text("previous successful speech")
    with pytest.raises(TranscriptionError):
        export_transcript(tmp_path, {}, {"segments": []})
    assert old.read_text() == "previous successful speech"


def test_explicit_translation_label(tmp_path):
    result = {"language": "hi", "duration": 1, "task": "translate",
              "segments": [{"start": 0, "end": 1, "text": "Hello"}]}
    paths = export_transcript(tmp_path, {"title": "Test", "source": "Local file"}, result)
    assert "English (translated from Hindi)" in paths["markdown"].read_text()


@pytest.mark.parametrize("start,end", [(float("nan"), 1), (0, float("inf")), (-1, 0), (2, 1)])
def test_invalid_timing_is_rejected(tmp_path, start, end):
    with pytest.raises(TranscriptionError):
        export_transcript(tmp_path, {}, {"segments": [{"start": start, "end": end, "text": "Hello"}]})


def _result(text):
    return {"language": "en", "duration": 1, "task": "transcribe",
            "segments": [{"start": 0, "end": 1, "text": text}]}


@pytest.mark.parametrize("failure", [PermissionError("file locked"), KeyboardInterrupt()])
def test_commit_failure_restores_complete_previous_export(tmp_path, monkeypatch, failure):
    source = {"title": "Test", "source": "Local file"}
    paths = export_transcript(tmp_path, source, _result("previous speech"))
    original_contents = {name: path.read_bytes() for name, path in paths.items()}
    original_replace = Path.replace
    failed = False

    def fail_second_file(path, target):
        nonlocal failed
        if Path(target) == paths["text"] and not failed:
            failed = True
            assert not paths["metadata"].exists(), "Completion marker must be invalid during commit"
            raise failure
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_second_file)
    with pytest.raises(type(failure)):
        export_transcript(tmp_path, source, _result("new speech"))
    assert {name: path.read_bytes() for name, path in paths.items()} == original_contents
    assert not list(tmp_path.glob(".exports-*"))


def test_failed_first_export_removes_newly_committed_files(tmp_path, monkeypatch):
    original_replace = Path.replace
    root = tmp_path.resolve()

    def fail_subtitles(path, target):
        if Path(target) == root / "transcript.srt":
            assert not (root / "metadata.json").exists()
            raise PermissionError("subtitle output locked")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_subtitles)
    with pytest.raises(PermissionError):
        export_transcript(tmp_path, {"title": "Test", "source": "Local file"}, _result("new speech"))
    assert list(tmp_path.iterdir()) == []


def test_interrupt_after_final_rename_also_restores_previous_set(tmp_path, monkeypatch):
    source = {"title": "Test", "source": "Local file"}
    paths = export_transcript(tmp_path, source, _result("previous speech"))
    original_contents = {name: path.read_bytes() for name, path in paths.items()}
    original_replace = Path.replace
    failed = False

    def interrupt_after_metadata(path, target):
        nonlocal failed
        result = original_replace(path, target)
        if Path(target) == paths["metadata"] and not failed:
            failed = True
            raise KeyboardInterrupt
        return result

    monkeypatch.setattr(Path, "replace", interrupt_after_metadata)
    with pytest.raises(KeyboardInterrupt):
        export_transcript(tmp_path, source, _result("new speech"))
    assert {name: path.read_bytes() for name, path in paths.items()} == original_contents
    assert not list(tmp_path.glob(".exports-*"))


def test_backup_failure_never_changes_previous_output(tmp_path, monkeypatch):
    source = {"title": "Test", "source": "Local file"}
    paths = export_transcript(tmp_path, source, _result("previous speech"))
    original_contents = {name: path.read_bytes() for name, path in paths.items()}
    original_copy = exporter.shutil.copy2

    def fail_backup(source, destination):
        if Path(source) == paths["text"]:
            raise OSError("backup device is full")
        return original_copy(source, destination)

    monkeypatch.setattr(exporter.shutil, "copy2", fail_backup)
    with pytest.raises(OSError, match="device is full"):
        export_transcript(tmp_path, source, _result("new speech"))
    assert {name: path.read_bytes() for name, path in paths.items()} == original_contents
    assert not list(tmp_path.glob(".exports-*"))


def test_rollback_failure_keeps_backups_and_invalidates_marker(tmp_path, monkeypatch):
    source = {"title": "Test", "source": "Local file"}
    paths = export_transcript(tmp_path, source, _result("previous speech"))
    original_contents = {name: path.read_bytes() for name, path in paths.items()}
    original_replace = Path.replace
    first_commit_finished = False

    def fail_commit_and_restore(path, target):
        nonlocal first_commit_finished
        if Path(target) == paths["text"]:
            raise PermissionError("text output locked")
        if Path(target) == paths["markdown"]:
            if first_commit_finished:
                raise PermissionError("markdown restore blocked")
            first_commit_finished = True
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_commit_and_restore)
    with pytest.raises(TranscriptionError, match="preserved for recovery"):
        export_transcript(tmp_path, source, _result("new speech"))
    assert not paths["metadata"].exists()
    recovery = next(tmp_path.glob(".exports-*")) / "previous"
    assert {name: (recovery / path.name).read_bytes() for name, path in paths.items()} == original_contents

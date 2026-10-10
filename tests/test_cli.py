import json
from pathlib import Path

import pytest

from video_transcriber import cli
from video_transcriber.errors import TranscriptionError

URL = "https://www.youtube.com/watch?v=abcdefghijk"
INFO = {"duration": 2.0, "sample_rate": 16000, "channels": 1}
RESULT = {"language": "te", "duration": 2.0, "language_probability": .98,
          "task": "transcribe", "segments": [{"start": 0, "end": 2, "text": "నమస్కారం Hello."}]}


@pytest.fixture
def pipeline(monkeypatch):
    calls = {"inspect": 0, "download": 0, "extract": 0, "recognize": []}
    monkeypatch.setattr(cli, "_require_package", lambda name: None)
    monkeypatch.setattr(cli.audio, "check_ffmpeg", lambda: None)
    monkeypatch.setattr(cli.audio, "probe_audio", lambda path: INFO.copy())

    def inspect(url):
        calls["inspect"] += 1
        return {**cli.downloader.identify_url(url), "title": "Demo", "duration": 2}

    def download(url, output_dir, progress):
        calls["download"] += 1
        path = output_dir / "media.mp4"
        path.write_bytes(b"synthetic media")
        return path

    def extract(source, target, progress):
        calls["extract"] += 1
        target.write_bytes(b"converted PCM")
        return INFO.copy()

    def recognize(path, **kwargs):
        calls["recognize"].append(kwargs)
        return {**RESULT, "task": kwargs["task"]}

    monkeypatch.setattr(cli.downloader, "inspect_video", inspect)
    monkeypatch.setattr(cli.downloader, "download_video", download)
    monkeypatch.setattr(cli.downloader, "cached_video", lambda url, folder:
                        folder / "media.mp4" if (folder / "media.mp4").exists() else None)
    monkeypatch.setattr(cli.audio, "extract_audio", extract)
    monkeypatch.setattr(cli.transcription, "transcribe_audio", recognize)
    return calls


def test_complete_url_run_and_offline_repeat_avoid_network(tmp_path, pipeline, capsys):
    argv = [URL, "--output-dir", str(tmp_path)]
    assert cli.main(argv) == 0
    assert pipeline["download"] == pipeline["extract"] == pipeline["inspect"] == 1
    folder = next(tmp_path.glob("Demo--*"))
    assert not (folder / "media.mp4").exists()
    assert (folder / "transcript.txt").read_text() == "నమస్కారం Hello.\n"
    assert str(folder / "transcript.md") in capsys.readouterr().out
    assert cli.main(argv + ["--offline"]) == 0
    assert pipeline["download"] == pipeline["extract"] == pipeline["inspect"] == 1
    assert len(pipeline["recognize"]) == 2
    assert not list((tmp_path / ".locks").iterdir())


def test_keep_video_then_default_cleanup(tmp_path, pipeline):
    argv = [URL, "--output-dir", str(tmp_path)]
    assert cli.main(argv + ["--keep-video"]) == 0
    folder = next(tmp_path.glob("Demo--*"))
    assert (folder / "media.mp4").exists()
    assert cli.main(argv) == 0
    assert pipeline["download"] == 1
    assert not (folder / "media.mp4").exists()


def test_keep_video_fetches_source_even_when_audio_cached(tmp_path, pipeline):
    argv = [URL, "--output-dir", str(tmp_path)]
    assert cli.main(argv) == 0
    assert cli.main(argv + ["--keep-video"]) == 0
    assert pipeline["download"] == 2
    assert pipeline["extract"] == 1


def test_local_input_preserved_and_new_content_has_new_folder(tmp_path, pipeline):
    source = tmp_path / "video.mp4"
    source.write_bytes(b"original user media")
    output = tmp_path / "results"
    argv = ["--file", str(source), "--output-dir", str(output), "--language", "Telugu"]
    assert cli.main(argv) == 0
    assert source.read_bytes() == b"original user media"
    assert pipeline["download"] == pipeline["inspect"] == 0
    assert pipeline["recognize"][0]["language"] == "te"
    source.write_bytes(b"changed user media")
    assert cli.main(argv) == 0
    assert len(list(output.glob("video--local-*"))) == 2


def test_positional_video_file_supports_spaces_and_preserves_original(tmp_path, pipeline):
    source = tmp_path / "phone video.mp4"
    source.write_bytes(b"a directly supplied video attachment")
    output = tmp_path / "results"
    assert cli.main([str(source), "--output-dir", str(output)]) == 0
    assert source.read_bytes() == b"a directly supplied video attachment"
    assert pipeline["download"] == pipeline["inspect"] == 0
    assert next(output.glob("phone_video--local-*/transcript.txt")).read_text() == "నమస్కారం Hello.\n"


def test_missing_positional_file_has_actionable_error(tmp_path, capsys):
    missing = tmp_path / "missing-video.mp4"
    assert cli.main([str(missing)]) == 1
    assert "Local media does not exist or is empty" in capsys.readouterr().err


@pytest.mark.parametrize("url", ["https://vimeo.com/123456789",
                                "https://cdn.example.org/video.mp4?token=abc%2F123&expires=99"])
def test_other_video_url_completes_pipeline(tmp_path, pipeline, url):
    assert cli.main([url, "--output-dir", str(tmp_path)]) == 0
    folder = next(tmp_path.glob("Demo--*"))
    metadata = json.loads((folder / "metadata.json").read_text())
    assert metadata["source"]["url"] == url
    assert pipeline["inspect"] == pipeline["download"] == pipeline["extract"] == 1


def test_damaged_audio_cache_rebuilt(tmp_path, pipeline):
    argv = [URL, "--output-dir", str(tmp_path), "--keep-video"]
    assert cli.main(argv) == 0
    (next(tmp_path.glob("Demo--*")) / "audio.wav").write_bytes(b"damaged")
    assert cli.main(argv) == 0
    assert pipeline["extract"] == 2
    assert pipeline["download"] == 1


def test_failure_and_interrupt_preserve_media_and_release_lock(tmp_path, pipeline, monkeypatch):
    argv = [URL, "--output-dir", str(tmp_path)]

    def failed(*args, **kwargs):
        raise TranscriptionError("No speech")

    monkeypatch.setattr(cli.transcription, "transcribe_audio", failed)
    assert cli.main(argv) == 1
    folder = next(tmp_path.glob("Demo--*"))
    assert (folder / "media.mp4").exists()
    assert not (folder / "transcript.txt").exists()

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli.transcription, "transcribe_audio", interrupted)
    assert cli.main(argv) == 130
    assert not list((tmp_path / ".locks").iterdir())
    assert pipeline["download"] == 1


def test_offline_uncached_source_fails_without_network(tmp_path, pipeline):
    assert cli.main([URL, "--output-dir", str(tmp_path), "--offline"]) == 1
    assert pipeline["inspect"] == pipeline["download"] == 0


def test_concurrent_source_lock_does_not_remove_other_job_lock(tmp_path, pipeline):
    key = cli.downloader.identify_url(URL)["stable_key"]
    with cli._source_lock(tmp_path, key):
        assert cli.main([URL, "--output-dir", str(tmp_path)]) == 1
        assert (tmp_path / ".locks" / f"{key}.lock").exists()
    assert pipeline["download"] == 0


@pytest.mark.parametrize("argv", [[], [URL, "--file", "video.mp4"], [URL, "--cpu-threads", "-1"]])
def test_invalid_input_arguments(argv):
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == 2

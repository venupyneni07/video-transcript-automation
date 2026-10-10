"""Downloader tests do not contact the network or require yt-dlp/FFmpeg."""

from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from video_transcriber import downloader
from video_transcriber.errors import DependencyError, DownloadError

VIDEO_ID = "dQw4w9WgXcQ"
URL = f"https://www.youtube.com/watch?v={VIDEO_ID}"


@pytest.mark.parametrize("url", [
    f"https://youtu.be/{VIDEO_ID}?si=share123",
    f"http://m.youtube.com/watch?v={VIDEO_ID}&list=PL123&t=42",
    f"https://www.youtube.com/shorts/{VIDEO_ID}",
    f"https://youtube.com/embed/{VIDEO_ID}/",
    f"https://www.youtube.com/live/{VIDEO_ID}",
])
def test_youtube_urls_are_canonical_and_ignore_playlist_context(url):
    assert downloader.identify_url(url) == {
        "source": "YouTube", "id": VIDEO_ID, "url": URL,
        "stable_key": f"youtube-{VIDEO_ID}",
    }


@pytest.mark.parametrize("kind", ["reel", "reels", "p", "tv"])
def test_instagram_video_urls(kind):
    result = downloader.identify_url(f"https://instagram.com/{kind}/ABC_123-xyz/?igsh=tracking")
    assert result["source"] == "Instagram"
    assert result["stable_key"] == "instagram-ABC_123-xyz"
    assert "igsh" not in result["url"]


@pytest.mark.parametrize("url", [
    "", "youtube.com/watch?v=dQw4w9WgXcQ", "file:///tmp/video.mp4",
    "https://youtube.com@evil.example/watch?v=dQw4w9WgXcQ",
    "https://user:pass@youtube.com/watch?v=dQw4w9WgXcQ",
    "https://youtube.com:443/watch?v=dQw4w9WgXcQ",
    "https://youtube.com:/watch?v=dQw4w9WgXcQ",
    "https://youtube.com/watch?v=short",
    "https://youtube.com/watch?v=dQw4w9WgXcQ&v=abcdefghijk",
    "https://youtube.com/playlist?list=PL123",
    "https://youtube.com/channel/ABC", "https://instagram.com/someuser/",
    "https://instagram.com/stories/abc/1234", "https://youtu.be/dQw4w9WgXcQ/more",
    "https://youtu.be/dQw4w9WgXcQ\n", "https://youtube.com\\evil/watch?v=dQw4w9WgXcQ",
    "https://youtube.com/watch?v=dQw4w9WgXcQ&x=%xx",
    "ftp://example.com/video.mp4", "javascript:alert(1)",
    "https://exa_mple.com/video.mp4", "https://-example.com/video.mp4",
    "https://example..com/video.mp4", "https://example.com:99999/video.mp4",
    "https://example.com:notaport/video.mp4", "https://256.999.0.1/video.mp4",
    "https://[invalid]/video.mp4", "https://[::1]malformed/video.mp4",
    "https://example.com/video\u0085.mp4",
])
def test_reject_unsafe_or_nonvideo_urls(url):
    with pytest.raises(DownloadError):
        downloader.identify_url(url)


@pytest.mark.parametrize("url,source", [
    ("https://vimeo.com/123456789", "vimeo.com"),
    ("https://cdn.example.org/video.mp4?token=abc%2F123&expires=99", "cdn.example.org"),
    ("https://example.com/video-page", "example.com"),
    ("https://cdn.example.com/movie.webm", "cdn.example.com"),
    ("http://127.0.0.1:8123/clip.mov", "127.0.0.1"),
    ("http://[::1]:8123/video", "::1"),
    ("http://localhost:8123/video?signature=one", "localhost"),
])
def test_generic_http_urls_retain_content_and_use_stable_hash(url, source):
    result = downloader.identify_url(url)
    assert result["url"] == url
    assert result["source"] == source
    assert len(result["id"]) == 20
    assert result["stable_key"] == f"url-{result['id']}"
    assert downloader.identify_url(url) == result


def test_generic_identity_preserves_signed_query_order_encoding_and_fragment():
    url = "HTTPS://CDN.EXAMPLE.ORG/media/clip%2Emp4?x=first&x=second&sig=a%2Fb%2Bc#part=2"
    result = downloader.identify_url(url)
    assert result["url"] == "https://cdn.example.org/media/clip%2Emp4?x=first&x=second&sig=a%2Fb%2Bc#part=2"
    assert result["stable_key"] == downloader.identify_url(result["url"])["stable_key"]
    changed = downloader.identify_url(result["url"].replace("sig=a", "sig=b"))
    assert changed["stable_key"] != result["stable_key"]


def test_similarly_named_host_is_generic_never_a_youtube_identity():
    result = downloader.identify_url("https://youtube.com.evil.example/watch?v=dQw4w9WgXcQ")
    assert result["source"] == "youtube.com.evil.example"
    assert result["stable_key"].startswith("url-")


def test_international_domain_is_normalized_without_modifying_path():
    result = downloader.identify_url("https://bücher.example/vidéo.mp4?sig=a%2B1")
    assert result["url"] == "https://xn--bcher-kva.example/vidéo.mp4?sig=a%2B1"


def install_fake_ydl(monkeypatch, tmp_path=None, info=None, failure=None, write_media=True):
    seen = {}
    result = {"title": "Example", "duration": 10.0, "availability": "public", **(info or {})}

    class FakeYDL:
        def extract_info(self, url, download, process=False, ie_key=None):
            seen["url"] = url
            seen.setdefault("extract_calls", []).append({"url": url, "download": download, "process": process})
            if failure:
                raise failure
            return result

        def process_ie_result(self, resolved, download):
            seen["download"] = download
            seen.setdefault("processed", []).append(resolved)
            seen["options"]["match_filter"](resolved)
            if download and tmp_path and write_media:
                (tmp_path / "media.mkv").write_bytes(b"fake-video-media")
            return resolved

    @contextmanager
    def factory(options):
        seen["options"] = options
        yield FakeYDL()

    monkeypatch.setattr(downloader, "_youtube_dl", factory)
    monkeypatch.setattr(downloader.shutil, "which", lambda name: f"/fake/bin/{name}")
    return seen


def test_inspection_metadata_and_safe_options(monkeypatch):
    seen = install_fake_ydl(monkeypatch)
    result = downloader.inspect_video(f"https://youtu.be/{VIDEO_ID}?t=5")
    assert result["title"] == "Example"
    assert result["duration"] == 10.0
    assert result["url"] == URL == seen["url"]
    assert seen["download"] is False
    options = seen["options"]
    assert options["cookiefile"] is None
    assert options["cookiesfrombrowser"] is None
    assert options["usenetrc"] is False
    assert options["geo_bypass"] is False
    assert options["allow_unplayable_formats"] is False
    assert options["remote_components"] == []
    assert options["skip_unavailable_fragments"] is False
    assert options["retries"] == options["fragment_retries"] == 3
    assert options["socket_timeout"] == 30
    assert options["allowed_extractors"] == ["default"]


@pytest.mark.parametrize("container", [
    {"_type": "playlist", "extractor_key": "YoutubeTab", "entries": [{"title": "one video"}]},
    {"_type": "playlist", "extractor_key": "VimeoAlbum", "entries": [{"title": "one video"}]},
    {"_type": "playlist", "extractor_key": "Generic", "entries": [{"title": "one"}, {"title": "two"}]},
    {"_type": "multi_video", "extractor_key": "Generic", "entries": [{"title": "one"}, {"title": "two"}]},
    {"_type": "playlist", "extractor_key": "Generic", "entries": []},
])
def test_containers_are_rejected_before_processing_or_downloading(monkeypatch, tmp_path, container):
    seen = install_fake_ydl(monkeypatch, tmp_path, info=container)
    with pytest.raises(DownloadError, match="video"):
        downloader.download_video("https://example.com/collection", tmp_path, lambda _: None)
    assert "processed" not in seen
    assert all(call["download"] is False and call["process"] is False for call in seen["extract_calls"])
    assert not (tmp_path / "media.mkv").exists()


@pytest.mark.parametrize("kind", ["playlist", "multi_video"])
def test_generic_single_embedded_video_wrapper_is_unwrapped(monkeypatch, tmp_path, kind):
    entry = {"_type": "video", "title": "Embedded clip", "duration": 10, "url": "https://cdn.example.com/clip.mp4"}
    seen = install_fake_ydl(monkeypatch, tmp_path, info={
        "_type": kind, "extractor_key": "Generic", "entries": [entry],
    })
    monkeypatch.setattr(downloader, "_probe_media", lambda *args: True)
    metadata = downloader.inspect_video("https://example.com/watch")
    assert metadata["title"] == "Embedded clip"
    assert metadata["source"] == "example.com"
    path = downloader.download_video("https://example.com/watch", tmp_path, lambda _: None)
    assert path.is_file()
    assert all("entries" not in item for item in seen["processed"])


def test_generic_lazy_wrapper_looks_at_only_two_entries():
    consumed = []

    def entries():
        for index in range(10_000):
            consumed.append(index)
            yield {"title": f"Video {index}"}

    with pytest.raises(DownloadError, match="multiple videos"):
        downloader._resolve_single_video(None, {"_type": "playlist", "extractor": "generic", "entries": entries()})
    assert consumed == [0, 1]


def test_embedded_reference_resolves_metadata_before_one_download():
    calls = []
    processed = []
    signed = "https://cdn.example.com/video?sig=a%2Fb&x=1&x=2"

    class FakeYDL:
        def extract_info(self, url, *, download, process, ie_key=None):
            calls.append((url, download, process, ie_key))
            if len(calls) == 1:
                return {"_type": "url_transparent", "url": signed, "ie_key": "Generic", "title": "Page title"}
            return {"_type": "video", "url": signed, "id": "clip", "duration": 10}

        def process_ie_result(self, info, *, download):
            processed.append((info, download))
            return info

    result = downloader._extract_video(FakeYDL(), "https://example.com/embed", download=True)
    assert calls == [("https://example.com/embed", False, False, None), (signed, False, False, "Generic")]
    assert len(processed) == 1
    assert processed[0][1] is True
    assert result["title"] == "Page title"
    assert result["url"] == signed


def test_redirect_to_playlist_is_rejected_before_processing():
    class FakeYDL:
        def extract_info(self, *args, **kwargs):
            assert kwargs["download"] is False
            assert kwargs["process"] is False
            return {"_type": "playlist", "extractor_key": "YoutubeTab", "entries": [{"id": "clip"}]}

        def process_ie_result(self, *args, **kwargs):
            pytest.fail("A playlist must never reach the processing/download step")

    with pytest.raises(DownloadError, match="Playlists"):
        downloader._resolve_single_video(FakeYDL(), {"_type": "url", "url": "https://example.com/playlist"})


def test_recursive_embedded_references_are_bounded():
    class FakeYDL:
        def extract_info(self, *args, **kwargs):
            return {"_type": "url", "url": "https://example.com/loop"}

    with pytest.raises(DownloadError, match="Too many"):
        downloader._extract_video(FakeYDL(), "https://example.com/loop", download=True)


def test_additional_video_urls_are_rejected_before_processing():
    with pytest.raises(DownloadError, match="additional videos"):
        downloader._resolve_single_video(None, {
            "_type": "video", "id": "clip", "additional_urls": ["https://example.com/other.mp4"],
        })


def test_signed_generic_url_reaches_extractor_unchanged(monkeypatch, tmp_path):
    url = "https://cdn.example.org/video.mp4?token=abc%2F123&expires=99"
    seen = install_fake_ydl(monkeypatch, tmp_path)
    monkeypatch.setattr(downloader, "_probe_media", lambda *args: True)
    assert downloader.inspect_video(url)["url"] == url
    downloader.download_video(url, tmp_path, lambda _: None)
    assert all(call["url"] == url for call in seen["extract_calls"])


@pytest.mark.parametrize("metadata,reason", [
    ({"availability": "private"}, "private"),
    ({"availability": "needs_auth"}, "authentication"),
    ({"availability": "subscriber_only"}, "restricted"),
    ({"is_live": True}, "Live"),
    ({"live_status": "is_upcoming"}, "upcoming"),
    ({"live_status": "post_live"}, "processing"),
    ({"has_drm": True}, "DRM"),
    ({"formats": [{"has_drm": True}]}, "DRM"),
    ({"age_limit": 18}, "Age-restricted"),
    ({"_type": "playlist", "entries": []}, "multiple videos"),
])
def test_restrictions_are_rejected_with_local_fallback(monkeypatch, metadata, reason):
    install_fake_ydl(monkeypatch, info=metadata)
    with pytest.raises(DownloadError, match=reason) as error:
        downloader.inspect_video(URL)
    assert "--file" in str(error.value)


def test_archived_livestream_is_supported(monkeypatch):
    install_fake_ydl(monkeypatch, info={"live_status": "was_live"})
    assert downloader.inspect_video(URL)["title"] == "Example"


def test_extractor_errors_are_actionable(monkeypatch):
    install_fake_ydl(monkeypatch, failure=RuntimeError("HTTP Error 403: Forbidden"))
    with pytest.raises(DownloadError, match="403") as error:
        downloader.inspect_video(URL)
    assert "--file" in str(error.value)
    assert "Deno/Node" in str(error.value)


def test_missing_package_is_actionable_and_environment_is_restored(monkeypatch):
    monkeypatch.delenv("YTDLP_NO_PLUGINS", raising=False)

    def missing(name):
        raise ImportError("not installed")

    monkeypatch.setattr(downloader.importlib, "import_module", missing)
    with pytest.raises(DependencyError, match="requirements.txt"):
        downloader.inspect_video(URL)
    assert "YTDLP_NO_PLUGINS" not in downloader.os.environ


def test_download_writes_completion_manifest_and_reuses_verified_copy(monkeypatch, tmp_path):
    seen = install_fake_ydl(monkeypatch, tmp_path)
    monkeypatch.setattr(downloader, "_probe_media", lambda *args: True)
    messages = []
    path = downloader.download_video(URL, tmp_path, messages.append)
    assert path == tmp_path / "media.mkv"
    manifest = json.loads((tmp_path / "media.download.json").read_text())
    assert manifest["state"] == "complete"
    assert manifest["stable_key"] == f"youtube-{VIDEO_ID}"
    assert seen["download"] is True

    def unexpected(*args):
        pytest.fail("Valid cached media must not trigger a second network request")

    monkeypatch.setattr(downloader, "_youtube_dl", unexpected)
    assert downloader.cached_video(URL, tmp_path) == path
    assert downloader.download_video(URL, tmp_path, messages.append) == path
    assert any("Reusing verified" in message for message in messages)


def test_partial_media_never_becomes_successful_cache(monkeypatch, tmp_path):
    install_fake_ydl(monkeypatch, tmp_path)
    monkeypatch.setattr(downloader, "_probe_media", lambda *args: True)
    (tmp_path / "media.mkv.part").write_bytes(b"partial")
    with pytest.raises(DownloadError, match="incomplete"):
        downloader.download_video(URL, tmp_path, lambda _: None)
    assert json.loads((tmp_path / "media.download.json").read_text())["state"] == "downloading"
    assert (tmp_path / "media.mkv.part").exists()


def test_retry_recovers_unchanged_project_owned_partial_after_verified_success(monkeypatch, tmp_path):
    install_fake_ydl(monkeypatch, tmp_path)
    monkeypatch.setattr(downloader, "_probe_media", lambda *args: True)
    (tmp_path / "media.download.json").write_text(json.dumps({
        "stable_key": f"youtube-{VIDEO_ID}", "state": "downloading",
    }))
    partial = tmp_path / "media.f140.m4a.part"
    partial.write_bytes(b"interrupted earlier attempt")
    unrelated = tmp_path / "user-notes.part"
    unrelated.write_text("keep this")
    path = downloader.download_video(URL, tmp_path, lambda _: None)
    assert path.is_file()
    assert not partial.exists()
    assert unrelated.read_text() == "keep this"
    assert downloader.cached_video(URL, tmp_path) == path


def test_failed_retry_preserves_original_partial(monkeypatch, tmp_path):
    install_fake_ydl(monkeypatch, tmp_path, failure=RuntimeError("temporary error"))
    (tmp_path / "media.download.json").write_text(json.dumps({
        "stable_key": f"youtube-{VIDEO_ID}", "state": "downloading",
    }))
    partial = tmp_path / "media.mp4.part"
    partial.write_bytes(b"partial")
    with pytest.raises(DownloadError, match="temporary error"):
        downloader.download_video(URL, tmp_path, lambda _: None)
    assert partial.read_bytes() == b"partial"


def test_cached_video_never_uses_an_untracked_or_different_video(monkeypatch, tmp_path):
    install_fake_ydl(monkeypatch, tmp_path)
    (tmp_path / "media.mp4").write_bytes(b"video")
    monkeypatch.setattr(downloader, "_probe_media", lambda *args: True)
    assert downloader.cached_video(URL, tmp_path) is None
    (tmp_path / "media.download.json").write_text(json.dumps({
        "stable_key": "youtube-abcdefghijk", "state": "complete", "filename": "media.mp4",
    }))
    assert downloader.cached_video(URL, tmp_path) is None


def test_preexisting_untracked_file_cannot_be_claimed_as_new_download(monkeypatch, tmp_path):
    install_fake_ydl(monkeypatch, tmp_path, write_media=False)
    monkeypatch.setattr(downloader, "_probe_media", lambda *args: True)
    (tmp_path / "media.mkv").write_bytes(b"old unrelated media")
    with pytest.raises(DownloadError, match="No complete video"):
        downloader.download_video(URL, tmp_path, lambda _: None)


def test_cache_requires_same_video(monkeypatch, tmp_path):
    install_fake_ydl(monkeypatch, tmp_path)
    (tmp_path / "media.download.json").write_text(json.dumps({"stable_key": "youtube-abcdefghijk"}))
    with pytest.raises(DownloadError, match="different video"):
        downloader.download_video(URL, tmp_path, lambda _: None)


def test_cache_rejects_path_traversal_and_modified_file(monkeypatch, tmp_path):
    candidate = tmp_path / "media.mp4"
    candidate.write_bytes(b"video")
    manifest = {"state": "complete", "filename": "../media.mp4"}
    assert downloader._cached_media(tmp_path, manifest, "ffprobe") is None
    manifest.update(filename="media.mp4", size=1, mtime_ns=candidate.stat().st_mtime_ns)
    monkeypatch.setattr(downloader, "_probe_media", lambda *args: True)
    assert downloader._cached_media(tmp_path, manifest, "ffprobe") is None


def test_download_rejects_symlink_before_writing(monkeypatch, tmp_path):
    install_fake_ydl(monkeypatch, tmp_path)
    target = tmp_path / "user-data.txt"
    target.write_text("keep this")
    try:
        (tmp_path / "media.mp4").symlink_to(target)
    except OSError:
        pytest.skip("Platform does not permit creating symbolic links")
    with pytest.raises(DownloadError, match="symbolic link"):
        downloader.download_video(URL, tmp_path, lambda _: None)
    assert target.read_text() == "keep this"


@pytest.mark.parametrize("streams,duration,valid", [
    ([{"codec_type": "video"}, {"codec_type": "audio"}], "10.0", True),
    ([{"codec_type": "video"}], "10.0", False),
    ([{"codec_type": "audio"}], "10.0", False),
    ([{"codec_type": "video"}, {"codec_type": "audio"}], "0", False),
    ([{"codec_type": "video"}, {"codec_type": "audio"}], "3.0", False),
])
def test_probe_requires_complete_audio_and_video(monkeypatch, tmp_path, streams, duration, valid):
    media = tmp_path / "media.mp4"
    media.write_bytes(b"video")

    def fake_run(command, **kwargs):
        assert isinstance(command, list)
        assert command[-1] == str(media)
        assert command[command.index("-protocol_whitelist") + 1] == "file,pipe"
        assert kwargs["timeout"] == 30
        return SimpleNamespace(stdout=json.dumps({"streams": streams, "format": {"duration": duration}}))

    monkeypatch.setattr(downloader.subprocess, "run", fake_run)
    assert downloader._probe_media(media, "ffprobe", 10) is valid


def test_probe_rejects_corrupt_media(monkeypatch, tmp_path):
    media = tmp_path / "media.mp4"
    media.write_bytes(b"corrupt")

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "ffprobe")

    monkeypatch.setattr(downloader.subprocess, "run", fail)
    assert downloader._probe_media(media, "ffprobe") is False


def test_interruption_is_not_wrapped_or_swallowed(monkeypatch):
    install_fake_ydl(monkeypatch, failure=KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        downloader.inspect_video(URL)

"""Real yt-dlp downloads against generated media on a loopback-only HTTP server."""

import errno
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import shutil
import subprocess
import threading

import pytest

from video_transcriber import downloader


@pytest.fixture
def generated_video(tmp_path):
    """A short synthetic picture and tone; no external or personal media."""
    pytest.importorskip("yt_dlp", reason="The real HTTP integration test requires yt-dlp.")
    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing:
        pytest.skip(f"The real HTTP integration test requires installed {', '.join(missing)}.")
    directory = tmp_path / "served"
    directory.mkdir()
    video = directory / "sample.mp4"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
            "-f", "lavfi", "-i", "color=c=black:s=48x32:r=10",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000",
            "-t", "0.6", "-c:v", "mpeg4", "-q:v", "5", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "24k", "-movflags", "+faststart", str(video),
        ],
        check=True, capture_output=True, text=True, timeout=30,
    )
    return video


def test_direct_mp4_query_download_and_offline_cache(generated_video, tmp_path, monkeypatch):
    # Keep even environments with a configured HTTP proxy entirely on loopback.
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    # A signed URL can depend on exact query bytes. The server refuses missing,
    # reordered, or decoded query strings, so accidental canonicalization fails.
    resource = "/sample.mp4?token=fixture%2Bkey&expires=123"
    requests = []

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(generated_video.parent), **kwargs)

        def send_head(self):
            requests.append(self.path)
            if self.path != resource:
                self.send_error(403, "Fixture URL query was not preserved")
                return None
            return super().send_head()

        def log_message(self, format, *args):
            pass

    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    except OSError as exc:
        if exc.errno in {errno.EACCES, errno.EPERM}:
            pytest.skip(f"Sandbox prohibits binding a localhost HTTP server: {exc}")
        raise
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    url = f"http://127.0.0.1:{server.server_port}{resource}"
    destination = tmp_path / "downloaded"
    progress = []
    try:
        info = downloader.inspect_video(url)
        assert info["url"] == url
        assert info["title"]
        media = downloader.download_video(url, destination, progress=progress.append)
        assert media.is_file()
        assert media.parent == destination.resolve()
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type:format=duration",
             "-of", "json", str(media)],
            check=True, capture_output=True, text=True, timeout=30,
        )
        details = json.loads(probe.stdout)
        assert {stream["codec_type"] for stream in details["streams"]} == {"audio", "video"}
        assert float(details["format"]["duration"]) == pytest.approx(0.6, abs=0.1)
        assert len(requests) >= 2 and all(path == resource for path in requests)
        assert hashlib.sha256(media.read_bytes()).digest() == hashlib.sha256(generated_video.read_bytes()).digest()
        manifest = json.loads((destination / "media.download.json").read_text(encoding="utf-8"))
        assert manifest["state"] == "complete"
        assert manifest["url"] == url
        assert not list(destination.glob("media*.part*"))
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)
    assert not worker.is_alive()

    # The server is stopped. Make an accidental extractor/network attempt fail
    # immediately; cache validation still uses the real on-disk file and ffprobe.
    def unexpected_network(*args, **kwargs):
        pytest.fail("A verified downloaded video must be reusable without a server or extractor")

    monkeypatch.setattr(downloader, "_youtube_dl", unexpected_network)
    assert downloader.cached_video(url, destination) == media
    assert downloader.download_video(url, destination, progress=progress.append) == media
    assert any("Reusing verified" in message for message in progress)
    assert downloader.cached_video(url.replace("expires=123", "expires=456"), destination) is None

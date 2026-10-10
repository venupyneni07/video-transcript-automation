"""Download supported, anonymously accessible video with bounded yt-dlp retries."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import contextmanager
import hashlib
import importlib
import ipaddress
from itertools import islice
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any
import unicodedata
from urllib.parse import parse_qs, urlsplit, urlunsplit

from .errors import DependencyError, DownloadError, TranscriberError

_YOUTUBE_ID = re.compile(r"[A-Za-z0-9_-]{11}\Z")
_INSTAGRAM_ID = re.compile(r"[A-Za-z0-9_-]{5,64}\Z")
_VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".mov", ".m4v", ".flv", ".avi", ".3gp", ".ts"}
_PARTIAL_NAME = re.compile(
    r"media(?:\.f[A-Za-z0-9_-]+)?\.(?:mp4|mkv|webm|mov|m4v|flv|avi|3gp|ts|m4a|opus|ogg|aac|mp3)\.part(?:-Frag\d+)?\Z"
)
_MANIFEST = "media.download.json"
_FALLBACK = 'Use --file "/path/to/authorized-media.mp4" with a local copy you may transcribe.'
_YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com", "youtu.be", "www.youtu.be"}
_INSTAGRAM_HOSTS = {"instagram.com", "www.instagram.com", "m.instagram.com"}


def _normalized_host(host: str) -> str:
    """Validate a DNS name, IPv4 address, or IPv6 literal without resolving it."""
    if "%" in host:
        raise ValueError("escaped hostnames and scoped addresses are unsupported")
    try:
        return ipaddress.ip_address(host).compressed
    except ValueError:
        if ":" in host or re.fullmatch(r"[0-9.]+", host):
            raise ValueError("invalid IP address") from None
    normalized = host.encode("idna").decode("ascii").lower()
    labels = normalized.removesuffix(".").split(".")
    if len(normalized) > 253 or not all(
        re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label) for label in labels
    ):
        raise ValueError("invalid hostname")
    return normalized


def identify_url(url: str) -> dict[str, str]:
    """Identify supported platform links or a normal HTTP(S) video/page URL.

    Known platforms retain canonical IDs. Other hosts use a hash of the complete
    normalized URL: path encoding, query parameters/order, and fragments are kept
    intact, so signed URLs are not damaged or conflated in the download cache.
    Acceptance here does not guarantee the site is supported by yt-dlp.
    """
    if not isinstance(url, str) or not url or len(url) > 8192:
        raise DownloadError("Provide one HTTP(S) video or video-page URL.")
    if ("\\" in url or any(char.isspace() or unicodedata.category(char) == "Cc" for char in url)
            or re.search(r"%(?![0-9A-Fa-f]{2})", url)):
        raise DownloadError("The video URL contains whitespace, control characters, or invalid escaping.")
    try:
        parsed = urlsplit(url)
        if (parsed.scheme.lower() not in {"https", "http"} or not parsed.hostname
                or parsed.username is not None or parsed.password is not None
                or parsed.netloc.endswith(":")):
            raise ValueError("unsupported URL authority")
        if parsed.netloc.startswith("[") and not re.fullmatch(r"\[[^\]]+\](?::[0-9]+)?", parsed.netloc):
            raise ValueError("malformed bracketed host")
        host = _normalized_host(parsed.hostname)
        port = parsed.port  # Access validates malformed/out-of-range port text.
    except ValueError as exc:
        raise DownloadError("Use a valid HTTP(S) video URL without embedded credentials.") from exc
    if host in _YOUTUBE_HOSTS | _INSTAGRAM_HOSTS and port is not None:
        raise DownloadError("YouTube and Instagram URLs must not include an explicit port.")
    video_id: str | None = None
    if host in {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com"}:
        if parsed.path in {"/watch", "/watch/"}:
            values = parse_qs(parsed.query, keep_blank_values=True).get("v", [])
            if len(values) == 1:
                video_id = values[0]
        else:
            match = re.fullmatch(r"/(?:shorts|embed|live)/([A-Za-z0-9_-]+)/?", parsed.path)
            if match:
                video_id = match.group(1)
    elif host in {"youtu.be", "www.youtu.be"}:
        match = re.fullmatch(r"/([A-Za-z0-9_-]+)/?", parsed.path)
        if match:
            video_id = match.group(1)
    elif host in {"instagram.com", "www.instagram.com", "m.instagram.com"}:
        match = re.fullmatch(r"/(reel|reels|p|tv)/([A-Za-z0-9_-]+)/?", parsed.path)
        if match and _INSTAGRAM_ID.fullmatch(match.group(2)):
            shortcode = match.group(2)
            kind = "reel" if match.group(1) in {"reel", "reels"} else match.group(1)
            return {"source": "Instagram", "id": shortcode,
                    "url": f"https://www.instagram.com/{kind}/{shortcode}/",
                    "stable_key": f"instagram-{shortcode}"}
    else:
        authority = f"[{host}]" if ":" in host else host
        if port is not None:
            authority += f":{port}"
        normalized = urlunsplit((parsed.scheme.lower(), authority, parsed.path, parsed.query, parsed.fragment))
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20]
        return {"source": host, "id": digest, "url": normalized, "stable_key": f"url-{digest}"}
    if video_id and _YOUTUBE_ID.fullmatch(video_id):
        return {"source": "YouTube", "id": video_id,
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "stable_key": f"youtube-{video_id}"}
    raise DownloadError("Provide a single valid video URL; profiles, channels, stories, and playlist-only URLs are unsupported. " + _FALLBACK)


def _restriction(info: Mapping[str, Any], *, allow_container: bool = False) -> str | None:
    if not allow_container and (info.get("_type", "video") in {"playlist", "multi_video", "compat_list"} or "entries" in info):
        return "This URL resolved to multiple items; provide one video."
    if info.get("additional_urls"):
        return "This URL requests additional videos; provide one video."
    if info.get("is_live") or info.get("live_status") in {"is_live", "is_upcoming", "post_live"}:
        return "Live, upcoming, and still-processing broadcasts are unsupported; wait for the complete recording."
    availability = info.get("availability")
    if info.get("is_private") or availability not in {None, "public", "unlisted"}:
        return f"The video is private, restricted, or requires authentication (availability: {availability or 'private'})."
    age_limit = info.get("age_limit")
    if isinstance(age_limit, (int, float)) and age_limit >= 18:
        return "Age-restricted videos are unsupported."
    formats = info.get("formats") or []
    if info.get("has_drm") or (formats and all(fmt.get("has_drm") for fmt in formats)):
        return "DRM-protected media is unsupported."
    return None


def _public_filter(info: Mapping[str, Any], *, incomplete: bool = False) -> None:
    # yt-dlp invokes this before transferring media, including on partial metadata.
    reason = _restriction(info)
    if reason:
        raise DownloadError(reason + " " + _FALLBACK)


def _options(progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    options: dict[str, Any] = {
        "quiet": True, "noprogress": True, "noplaylist": True,
        "ignoreerrors": False, "cachedir": False, "socket_timeout": 30,
        "retries": 3, "fragment_retries": 3, "extractor_retries": 3,
        "file_access_retries": 1, "skip_unavailable_fragments": False,
        "retry_sleep_functions": {name: lambda attempt: min(attempt, 3)
                                  for name in ("http", "fragment", "extractor", "file_access")},
        "cookiefile": None, "cookiesfrombrowser": None, "usenetrc": False,
        "username": None, "password": None, "videopassword": None,
        "geo_bypass": False, "allow_unplayable_formats": False,
        "enable_file_urls": False, "remote_components": [],
        "allowed_extractors": ["default"],
        "match_filter": _public_filter,
        "format": "bv[height<=480]+ba/b[height<=480]/wv+ba/w",
        "merge_output_format": "mkv", "overwrites": True,
        "continuedl": False, "nopart": False, "updatetime": False,
    }
    # Newer YouTube extraction needs yt-dlp-ejs and a JS runtime. No remote
    # components are fetched; use the ejs bundled by yt-dlp[default].
    runtimes = {name: {"path": path} for name in ("deno", "node")
                if (path := shutil.which(name))}
    options["js_runtimes"] = runtimes
    if progress:
        last_bucket = [-1]

        def hook(state: dict[str, Any]) -> None:
            if state.get("status") == "finished":
                progress("Media stream downloaded; finalizing video and audio…")
                last_bucket[0] = -1
            elif state.get("status") == "downloading":
                total = state.get("total_bytes") or state.get("total_bytes_estimate")
                if total:
                    percent = min(100, int(100 * (state.get("downloaded_bytes") or 0) / total))
                    bucket = percent // 10
                    if bucket > last_bucket[0]:
                        progress(f"Downloading media: {percent}%")
                        last_bucket[0] = bucket

        options["progress_hooks"] = [hook]
    return options


@contextmanager
def _youtube_dl(options: dict[str, Any]):
    # The embedding API does not read CLI configuration. Disable automatic
    # third-party plugin discovery too, without altering the user's environment.
    previous = os.environ.get("YTDLP_NO_PLUGINS")
    os.environ["YTDLP_NO_PLUGINS"] = "1"
    try:
        try:
            module = importlib.import_module("yt_dlp")
        except ImportError as exc:
            raise DependencyError('yt-dlp is missing. Install dependencies with python -m pip install -r requirements.txt. ' + _FALLBACK) from exc
        with module.YoutubeDL(options) as ydl:
            yield ydl
    finally:
        if previous is None:
            os.environ.pop("YTDLP_NO_PLUGINS", None)
        else:
            os.environ["YTDLP_NO_PLUGINS"] = previous


def _checked_info(info: Any) -> dict[str, Any]:
    if not isinstance(info, Mapping) or not info:
        raise DownloadError("The extractor returned no accessible video. " + _FALLBACK)
    _public_filter(info)
    return dict(info)


def _resolve_single_video(ydl: Any, info: Any, *, depth: int = 0) -> dict[str, Any]:
    """Resolve extraction references without permitting playlist media transfers.

    ``process=False`` keeps yt-dlp from descending into containers. Generic HTML
    pages sometimes represent a single embedded video as a playlist; unwrap only
    those single-entry wrappers, inspecting at most two entries of a lazy result.
    Named site playlists and all multi-video results are refused before download.
    """
    if depth >= 12:
        raise DownloadError("Too many nested video references or redirects. " + _FALLBACK)
    if not isinstance(info, Mapping) or not info:
        raise DownloadError("The extractor returned no accessible video. " + _FALLBACK)
    info = dict(info)
    reason = _restriction(info, allow_container=True)
    if reason:
        raise DownloadError(reason + " " + _FALLBACK)
    kind = info.get("_type", "video")
    if kind in {"playlist", "multi_video", "compat_list"} or "entries" in info:
        generic = str(info.get("extractor_key") or info.get("extractor") or "").lower() == "generic"
        if not generic or kind == "compat_list":
            raise DownloadError("Playlists and multiple videos are unsupported; provide one video URL. " + _FALLBACK)
        entries = list(islice(iter(info.get("entries") or []), 2))
        if len(entries) != 1 or not isinstance(entries[0], Mapping):
            raise DownloadError("This page contains multiple videos or no accessible video; provide one video URL. " + _FALLBACK)
        entry = dict(entries[0])
        entry.setdefault("extractor", "generic")
        entry.setdefault("extractor_key", "Generic")
        return _resolve_single_video(ydl, entry, depth=depth + 1)
    if kind in {"url", "url_transparent"}:
        reference = info.get("url")
        if not isinstance(reference, str) or not reference:
            raise DownloadError("The page did not provide a usable video reference. " + _FALLBACK)
        resolved = ydl.extract_info(reference, download=False, process=False, ie_key=info.get("ie_key"))
        result = _resolve_single_video(ydl, resolved, depth=depth + 1)
        if kind == "url_transparent":
            # Retain useful embedding-page labels only when the video lacks them.
            for key in ("title", "description", "duration"):
                if result.get(key) is None and info.get(key) is not None:
                    result[key] = info[key]
        return result
    if kind != "video":
        raise DownloadError("The URL did not resolve to a supported single video. " + _FALLBACK)
    return _checked_info(info)


def _extract_video(ydl: Any, url: str, *, download: bool) -> dict[str, Any]:
    raw = ydl.extract_info(url, download=False, process=False)
    single = _resolve_single_video(ydl, raw)
    return _checked_info(ydl.process_ie_result(single, download=download))


def _failure(exc: Exception) -> DownloadError:
    detail = re.sub(r"\x1b\[[0-9;]*[A-Za-z]|[\x00-\x1f\x7f]", " ", str(exc))[:1200]
    return DownloadError(
        "Unable to access or download this video anonymously. It may be private, deleted, "
        "restricted, rate-limited, or unsupported by the current extractor. "
        "For YouTube, check yt-dlp[default] and a supported Deno/Node runtime. "
        f"Extractor detail: {detail}. {_FALLBACK}"
    )


def _duration(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result > 0 else None


def inspect_video(url: str) -> dict[str, Any]:
    """Check anonymous accessibility and return minimal, stable metadata."""
    identity = identify_url(url)
    try:
        with _youtube_dl(_options()) as ydl:
            info = _extract_video(ydl, identity["url"], download=False)
    except TranscriberError:
        raise
    except Exception as exc:
        raise _failure(exc) from exc
    return {**identity, "title": str(info.get("title") or identity["id"]),
            "duration": _duration(info.get("duration"))}


def _probe_media(path: Path, ffprobe: str, expected_duration: float | None = None) -> bool:
    if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-protocol_whitelist", "file,pipe",
             "-show_entries", "format=duration:stream=codec_type",
             "-of", "json", str(path)], capture_output=True, text=True, check=True, timeout=30,
        )
        info = json.loads(result.stdout)
        streams = {stream.get("codec_type") for stream in info.get("streams", [])}
        duration = _duration(info.get("format", {}).get("duration"))
        if not {"audio", "video"}.issubset(streams) or duration is None:
            return False
        return expected_duration is None or duration >= expected_duration - max(2, expected_duration * 0.02)
    except (OSError, subprocess.SubprocessError, ValueError, TypeError, AttributeError):
        return False


def _read_manifest(directory: Path) -> dict[str, Any]:
    path = directory / _MANIFEST
    if path.is_symlink():
        raise DownloadError("Refusing to use a symbolic link for the download manifest.")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (FileNotFoundError, ValueError):
        return {}


def _write_manifest(directory: Path, value: dict[str, Any]) -> None:
    if (directory / _MANIFEST).is_symlink():
        raise DownloadError("Refusing to replace a symbolic link for the download manifest.")
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory,
                                     prefix=".download-", suffix=".json", delete=False) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        temporary = Path(handle.name)
    temporary.replace(directory / _MANIFEST)


def _cached_media(directory: Path, manifest: dict[str, Any], ffprobe: str) -> Path | None:
    if manifest.get("state") != "complete" or any(directory.glob("media*.part*")):
        return None
    name = manifest.get("filename")
    if not isinstance(name, str) or name != Path(name).name:
        return None
    candidate = directory / name
    if candidate.stem != "media" or candidate.suffix not in _VIDEO_EXTENSIONS:
        return None
    if not candidate.is_file() or candidate.is_symlink():
        return None
    stat = candidate.stat()
    if stat.st_size != manifest.get("size") or stat.st_mtime_ns != manifest.get("mtime_ns"):
        return None
    return candidate if _probe_media(candidate, ffprobe, _duration(manifest.get("duration"))) else None


def cached_video(url: str, output_dir: Path) -> Path | None:
    """Return only a provenance-checked complete download, without using the network."""
    identity = identify_url(url)
    directory = Path(output_dir).expanduser().resolve()
    if not directory.is_dir():
        return None
    manifest = _read_manifest(directory)
    if manifest.get("stable_key") != identity["stable_key"]:
        return None
    ffprobe = shutil.which("ffprobe")
    return _cached_media(directory, manifest, ffprobe) if ffprobe else None


def download_video(url: str, output_dir: Path, progress: Callable[[str], None] = print) -> Path:
    """Download one complete video or reuse a provenance-checked, probed copy.

    ``output_dir`` must belong to this video. A completion manifest records the
    stable identity, exact filename, size, and mtime; incomplete files are never
    returned. No arbitrary files are deleted by this module.
    """
    identity = identify_url(url)
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise DependencyError("ffprobe is required to validate downloaded media. Install FFmpeg (which includes ffprobe).")
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    manifest = _read_manifest(directory)
    if manifest.get("stable_key") not in {None, identity["stable_key"]}:
        raise DownloadError("This output folder belongs to a different video; choose a new output folder.")
    cached = _cached_media(directory, manifest, ffprobe)
    if cached and manifest.get("stable_key") == identity["stable_key"]:
        progress(f"Reusing verified downloaded media: {cached}")
        return cached
    if not shutil.which("ffmpeg"):
        raise DependencyError("FFmpeg is required to merge downloaded video and audio. Install the FFmpeg executable.")
    if any(path.is_symlink() for path in directory.glob("media*")):
        raise DownloadError("Refusing to download over a media symbolic link in the output folder.")
    previous_files = {path.name: (path.stat().st_size, path.stat().st_mtime_ns)
                      for path in directory.glob("media.*") if path.is_file()}
    # A prior manifest binds interrupted files to this video. Only unchanged,
    # strictly named partials from that prior attempt may be removed on success.
    previous_partials = {
        path.name: (path.stat().st_size, path.stat().st_mtime_ns)
        for path in directory.glob("media*.part*")
        if manifest.get("stable_key") == identity["stable_key"]
        and path.is_file() and _PARTIAL_NAME.fullmatch(path.name)
    }
    _write_manifest(directory, {**identity, "state": "downloading"})
    progress(f"Downloading {identity['source']} video…")
    options = _options(progress)
    # Escape literal '%' in the directory, because outtmpl has its own template syntax.
    options["outtmpl"] = str(directory / "media").replace("%", "%%") + ".%(ext)s"
    try:
        with _youtube_dl(options) as ydl:
            info = _extract_video(ydl, identity["url"], download=True)
        expected_duration = _duration(info.get("duration"))
        candidates = [path for path in directory.glob("media.*")
                      if path.stem == "media" and path.suffix in _VIDEO_EXTENSIONS
                      and previous_files.get(path.name) != (path.stat().st_size, path.stat().st_mtime_ns)]
        # Merged output can change extension; yt-dlp's raw extractor ext is not
        # authoritative. Prefer the newest completed file and validate its streams.
        candidates.sort(key=lambda path: path.stat().st_mtime_ns, reverse=True)
        media = next((path for path in candidates if _probe_media(path, ffprobe, expected_duration)), None)
        if media is None:
            raise DownloadError("No complete video with a readable audio stream was downloaded. " + _FALLBACK)
        stale_partials = []
        for partial in directory.glob("media*.part*"):
            if (partial.is_symlink() or not partial.is_file()
                    or previous_partials.get(partial.name) != (partial.stat().st_size, partial.stat().st_mtime_ns)):
                raise DownloadError("The download is incomplete (new or untracked partial media remains). Retry the URL or " + _FALLBACK)
            stale_partials.append(partial)
        for partial in stale_partials:
            partial.unlink()
        stat = media.stat()
        _write_manifest(directory, {**identity, "state": "complete", "filename": media.name,
                                   "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                                   "duration": expected_duration})
        return media
    except TranscriberError:
        raise
    except Exception as exc:
        raise _failure(exc) from exc

"""CLI orchestration, durable audio caching, and source-scoped locking."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path

from . import audio, downloader, transcription
from .errors import DependencyError, TranscriberError
from .exporter import export_transcript, sanitize_title, write_json

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Transcribe a supported video URL, direct video link, or local media file.")
    result.add_argument("url", nargs="?", metavar="INPUT",
                        help="HTTP(S) video/page URL or a local video/audio path")
    result.add_argument("--file", type=Path, help="Local video or audio (the original is never deleted)")
    result.add_argument("--model", default="small", choices=["tiny", "base", "small", "medium", "large-v3"])
    result.add_argument("--language", default="auto", help="auto, te, en, hi, or another Whisper language code")
    result.add_argument("--keep-video", action="store_true", help="Retain the downloaded source media")
    result.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    result.add_argument("--compute-type", default="int8", choices=["int8", "float16", "int8_float16", "float32", "auto"])
    result.add_argument("--cpu-threads", type=int, default=0, help="CPU threads (0 uses library default)")
    result.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "output")
    result.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / ".models")
    result.add_argument("--offline", action="store_true", help="Use cached models and media only; no URL access")
    result.add_argument("--translate", action="store_true", help="Explicitly translate speech into English")
    result.add_argument("--check", action="store_true", help="Check installed dependencies and exit")
    return result


def check_dependencies() -> bool:
    okay = True
    print(f"Python: {sys.version.split()[0]}")
    for executable in ("ffmpeg", "ffprobe"):
        path = shutil.which(executable)
        print(f"{executable}: {path or 'MISSING (install FFmpeg; see README.md)'}")
        okay = okay and bool(path)
    for name, distribution in (("yt_dlp", "yt-dlp"), ("faster_whisper", "faster-whisper")):
        try:
            importlib.import_module(name)
            print(f"{distribution}: {importlib.metadata.version(distribution)}")
        except (ImportError, OSError, RuntimeError, importlib.metadata.PackageNotFoundError) as exc:
            print(f"{distribution}: UNAVAILABLE ({exc}); install requirements.txt in this interpreter")
            okay = False
    print("YouTube JavaScript runtime: " + (shutil.which("deno") or shutil.which("node") or
          "none found; install Deno or a supported Node version for full YouTube support"))
    return okay


def _require_package(name: str) -> None:
    if importlib.util.find_spec(name) is None:
        raise DependencyError(f"Missing {name}. Run: python -m pip install -r requirements.txt")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        return result if isinstance(result, dict) else {}
    except (OSError, ValueError):
        return {}


@contextmanager
def _source_lock(root: Path, key: str):
    locks = root / ".locks"
    locks.mkdir(parents=True, exist_ok=True)
    lock = locks / f"{key}.lock"
    try:
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise TranscriberError(f"Another job may be processing this source. Lock: {lock}. "
                               "If no such job is running, remove the stale lock and retry.") from exc
    try:
        with os.fdopen(descriptor, "w") as file:
            file.write(f"pid={os.getpid()}\n")
        yield
    finally:
        lock.unlink(missing_ok=True)


def _cached_folder(root: Path, key: str) -> Path | None:
    for folder in sorted(root.glob(f"*--{key}")):
        if folder.is_dir() and not folder.is_symlink() and _read_json(folder / "source.json").get("stable_key") == key:
            return folder
    return None


def _cached_audio(folder: Path, key: str) -> dict | None:
    target = folder / "audio.wav"
    cache = _read_json(folder / "audio-cache.json")
    if cache.get("stable_key") != key or not target.is_file() or target.is_symlink():
        return None
    if cache.get("sha256") != _sha256(target):
        return None
    try:
        info = audio.probe_audio(target)
        if int(info["sample_rate"]) == 16000 and int(info["channels"]) == 1:
            return info
    except (TranscriberError, KeyError, ValueError, TypeError):
        pass
    return None


def run(args: argparse.Namespace) -> dict[str, Path]:
    local = args.file.expanduser().resolve() if args.file is not None else None
    if local is not None:
        if not local.is_file() or local.stat().st_size == 0:
            raise TranscriberError(f"Local media does not exist or is empty: {local}")
        progress("Identifying local media…")
        identifier = _sha256(local)[:24]
        source = {"source": "Local file", "id": identifier, "stable_key": f"local-{identifier}",
                  "title": local.stem, "file": str(local), "url": None}
    else:
        source = downloader.identify_url(args.url)
    audio.check_ffmpeg()
    _require_package("faster_whisper")
    root = args.output_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    key = source["stable_key"]
    with _source_lock(root, key):
        folder = _cached_folder(root, key)
        if folder is None:
            if local is None:
                if args.offline:
                    raise TranscriberError("No cached media for this URL. Run online first or use --file.")
                _require_package("yt_dlp")
                progress("Checking public video availability…")
                source = downloader.inspect_video(source["url"])
            folder = root / f"{sanitize_title(source['title'])}--{key}"
            folder.mkdir(parents=True, exist_ok=True)
            write_json(folder / "source.json", source)
        else:
            source = _read_json(folder / "source.json")
            if local is not None:
                source["file"] = str(local)
        target = folder / "audio.wav"
        audio_info = _cached_audio(folder, key)
        media = local if local is not None else downloader.cached_video(source["url"], folder)
        needs_media = audio_info is None or (args.keep_video and local is None and media is None)
        if needs_media and media is None:
            if args.offline:
                raise TranscriberError("No usable cached source media. Run online or provide --file.")
            _require_package("yt_dlp")
            progress("Downloading video…")
            media = downloader.download_video(source["url"], folder, progress=progress)
        if audio_info is None:
            audio_info = audio.extract_audio(media, target, progress=progress)
            write_json(folder / "audio-cache.json", {"stable_key": key, "sha256": _sha256(target),
                                                     "sample_rate": 16000, "channels": 1})
        else:
            progress(f"Using verified cached audio: {target}")
        language = args.language.lower().strip()
        language = {"telugu": "te", "english": "en", "hindi": "hi"}.get(language, language)
        result = transcription.transcribe_audio(
            target, model=args.model, language=None if language == "auto" else language,
            device=args.device, compute_type=args.compute_type,
            task="translate" if args.translate else "transcribe",
            model_dir=args.model_dir.expanduser().resolve(), offline=args.offline,
            cpu_threads=args.cpu_threads, progress=progress,
        )
        source["duration"] = audio_info["duration"]
        result["duration"] = audio_info["duration"]
        paths = export_transcript(folder, source, result)
        paths["audio"] = target.resolve()
        # Never delete a local input. Only our named download after a complete export.
        if local is None and media is not None and not args.keep_video:
            if media.parent.resolve() == folder.resolve() and media.name.startswith("media."):
                try:
                    media.unlink()
                except OSError as exc:
                    progress(f"Transcript saved, but could not remove downloaded video: {exc}")
        return paths


def main(argv: list[str] | None = None) -> int:
    arguments = parser()
    args = arguments.parse_args(argv)
    if args.check:
        return 0 if check_dependencies() else 1
    if bool(args.url) == (args.file is not None):
        arguments.error("Provide exactly one video URL, local file path, or --file PATH.")
    if args.cpu_threads < 0:
        arguments.error("--cpu-threads must be zero or positive.")
    # Attachments and local media can use either the simple positional form or
    # --file. Keep URL-shaped inputs on the URL validator's path, even if invalid.
    if args.file is None and "://" not in args.url:
        args.file = Path(args.url)
        args.url = None
    try:
        paths = run(args)
    except KeyboardInterrupt:
        progress("Interrupted. Completed audio and partial downloads are retained; rerun the same command.")
        return 130
    except (TranscriberError, OSError) as exc:
        progress(f"Error: {exc}")
        return 1
    print("Transcript saved:")
    for kind, path in paths.items():
        print(f"  {kind}: {path}")
    return 0

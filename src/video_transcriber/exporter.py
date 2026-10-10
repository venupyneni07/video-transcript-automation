"""UTF-8 exports. Transcript text is never summarized or translated here."""

from __future__ import annotations

import filecmp
import json
import math
import os
import re
import shutil
import tempfile
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from .errors import TranscriptionError

LANGUAGES = {"te": "Telugu", "en": "English", "hi": "Hindi"}


def sanitize_title(title: str) -> str:
    """A portable, bounded filename component, including Windows device names."""
    title = unicodedata.normalize("NFKC", str(title))
    title = re.sub(r"[^\w\-]+", "_", title, flags=re.UNICODE).strip("_-")
    title = title.encode("utf-8")[:100].decode("utf-8", errors="ignore")
    title = title.rstrip("_-") or "video"
    if re.fullmatch(r"(?i)(con|prn|aux|nul|com[1-9]|lpt[1-9])", title):
        title = "video_" + title
    return title


def timestamp(seconds: float, *, srt: bool = False) -> str:
    milliseconds = max(0, round(float(seconds) * 1000))
    whole, millis = divmod(milliseconds, 1000)
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    base = f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{base},{millis:03d}" if srt else base


def _markdown(value: object) -> str:
    value = " ".join(str(value).split())
    return re.sub(r"([\\`*_{}\[\]<>#!|])", r"\\\1", value)


def write_json(path: Path, value: dict) -> None:
    _atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         dir=path.parent, prefix=".export-", delete=False) as file:
            temp_path = Path(file.name)
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        temp_path.replace(path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def export_transcript(output_dir: Path, source: dict, result: dict) -> dict[str, Path]:
    """Write complete exports, rolling back normal errors and interruptions.

    The metadata marker is absent while the set is being committed. Backups
    protect previous exports on caught failures; a forced process kill or a
    storage failure can still require recovery from the staging directory.
    """
    segments = []
    for item in result["segments"]:
        text = str(item["text"]).strip()
        if not text:
            continue
        start, end = float(item["start"]), float(item["end"])
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start:
            raise TranscriptionError("The recognizer returned invalid segment timestamps.")
        segments.append({"start": start, "end": end, "text": text})
    if not segments:
        raise TranscriptionError("No recognizable speech was returned; no transcript was created.")
    language = LANGUAGES.get(result["language"], result["language"])
    if result.get("task") == "translate":
        language = f"English (translated from {language})"
    markdown = ["# Video Transcript", "", f"**Title:** {_markdown(source['title'])}",
                f"**Source:** {_markdown(source['source'])}"]
    if source.get("url"):
        markdown.append(f"**URL:** {source['url']}")
    if source.get("file"):
        markdown.append(f"**File:** {_markdown(source['file'])}")
    markdown.extend([f"**Language:** {_markdown(language)}",
                     f"**Duration:** {timestamp(result['duration'])}", "", "## Full Transcript", ""])
    for segment in segments:
        markdown.extend([f"[{timestamp(segment['start'])}] {_markdown(segment['text'])}", ""])
    plain = "\n\n".join(segment["text"] for segment in segments) + "\n"
    subtitles = []
    for index, segment in enumerate(segments, 1):
        start_ms = round(segment["start"] * 1000)
        end_ms = max(start_ms + 1, round(segment["end"] * 1000))
        subtitles.append(f"{index}\n{timestamp(start_ms / 1000, srt=True)} --> "
                         f"{timestamp(end_ms / 1000, srt=True)}\n{segment['text']}\n")
    paths = {name: (output_dir / filename).resolve() for name, filename in
             (("markdown", "transcript.md"), ("text", "transcript.txt"),
              ("subtitles", "transcript.srt"), ("metadata", "metadata.json"))}
    metadata = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "transcription": {key: value for key, value in result.items() if key != "segments"},
        "segments": segments,
        "files": {key: str(path) for key, path in paths.items()},
        "quality_note": "Automatic best-effort transcription; review unclear speech against audio. "
                        "Language detection, code-switching, words, and timestamps may be inaccurate.",
    }
    # Prepare every file and back up every old destination before changing any.
    # Keep staging on disk if rollback itself fails so old output is recoverable.
    staged = Path(tempfile.mkdtemp(prefix=".exports-", dir=output_dir))
    remove_staging = True
    try:
        (staged / "transcript.md").write_text("\n".join(markdown), encoding="utf-8")
        (staged / "transcript.txt").write_text(plain, encoding="utf-8")
        (staged / "transcript.srt").write_text("\n".join(subtitles), encoding="utf-8")
        write_json(staged / "metadata.json", metadata)
        backup = staged / "previous"
        backup.mkdir()
        for path in paths.values():
            if path.exists():
                shutil.copy2(path, backup / path.name)
        marker = paths["metadata"]
        try:
            marker.unlink(missing_ok=True)
            # Dict order places metadata last, after every user-facing export.
            for path in paths.values():
                (staged / path.name).replace(path)
        except BaseException as original:
            failures = []
            try:
                marker.unlink(missing_ok=True)
            except OSError as exc:
                failures.append(f"{marker.name}: {exc}")
            for path in paths.values():
                if path == marker:
                    continue
                previous = backup / path.name
                try:
                    if previous.exists():
                        # An unchanged destination may be locked against replacement.
                        # Comparing also handles interrupts just after an atomic rename.
                        if not path.is_file() or not filecmp.cmp(path, previous, shallow=False):
                            shutil.copy2(previous, staged / path.name)
                            (staged / path.name).replace(path)
                    else:
                        path.unlink(missing_ok=True)
                except OSError as exc:
                    failures.append(f"{path.name}: {exc}")
            if not failures and (backup / marker.name).exists():
                try:
                    shutil.copy2(backup / marker.name, staged / marker.name)
                    (staged / marker.name).replace(marker)
                except OSError as exc:
                    failures.append(f"{marker.name}: {exc}")
            if failures:
                remove_staging = False
                raise TranscriptionError(
                    f"Export failed and could not fully restore previous files: {'; '.join(failures)}. "
                    f"Previous exports are preserved for recovery in {backup}."
                ) from original
            raise
    finally:
        if remove_staging:
            shutil.rmtree(staged)
    return paths

---
name: video-to-transcript
description: Transcribe accessible video links from YouTube, Instagram, other yt-dlp-supported sites, direct media URLs, or attached/local video and audio files into full original-language Markdown, TXT, and SRT transcripts. Use when the user supplies a video link or file and asks to transcribe it into text.
---

# Video to transcript

Use the project's existing `transcribe.py` pipeline. Produce the full recognized transcript and file links; do not replace it with a summary or infer speech from a title, description, or captions.

## Locate the configured project

Resolve the project root before running commands:

1. If `VIDEO_TRANSCRIBER_HOME` is set, expand it and verify that it contains `transcribe.py`, `requirements.txt`, and `src/video_transcriber/`.
2. Otherwise resolve this `SKILL.md` path, following symlinks, and walk its ancestors to find those project files. For the bundled location, the project is the directory containing `.agents/`, three levels above the skill folder.
3. A copied global skill has no project among its ancestors. Use an explicit project path already supplied by the user, or ask for the project's location. Do not search unrelated personal directories or create a second installation.

Use the absolute `<project>/.venv/bin/python` executable on macOS/Linux or `<project>/.venv/Scripts/python.exe` on Windows, with the working directory set to `<project>`. Resolve a user-provided relative media path against the directory in which the user referred to it before changing the working directory. Read `<project>/README.md` when setup or troubleshooting is needed.

## Execute the request

- Resolve one supplied HTTP(S) video link, existing local media file, or attached/uploaded video. For an attachment, inspect the file information already provided by the environment and use its real accessible local path. Do not guess paths, mistake a display filename for a file, or ask for another upload when the attachment is already available. If no usable input is available, ask for a link, accessible local path, or upload in the normal chat response.
- Accept supported links beyond YouTube and Instagram: the CLI uses yt-dlp's built-in site and generic extractors, including direct MP4/WebM/MOV links. Do not promise every URL will work. Preserve complete query strings when passing download links; treat signed URL query parameters as sensitive source data and avoid echoing them unnecessarily in chat. Quote shell arguments correctly; with process tools that accept argument arrays, pass each argument separately.
- Run `<python> <project>/transcribe.py --check` before the first transcription in an unfamiliar environment. If dependencies are missing, use the project README's setup for the current OS within the user's authorized environment. Install Python dependencies into the project virtual environment; do not change global Python packages. Report environment permission or network limitations accurately.
- For a URL, run `<python> <project>/transcribe.py <URL>`. For a local or attached file, use `<python> <project>/transcribe.py --file <absolute-media-path>`. The CLI also accepts a local path as its positional input; the explicit `--file` form makes attachment handling clear.
- Defaults are `--model small --device cpu --compute-type int8 --language auto`. Preserve these unless the user requests another setting or troubleshooting calls for a clearly explained change. Use `--language te`, `en`, or `hi` (also Telugu, English, or Hindi) for an explicit language hint; leave mixed-language recordings on auto initially.
- Honor requested `--model` (`tiny`, `base`, `small`, `medium`, `large-v3`), `--keep-video`, output directory, thread count, and device choices. Optional NVIDIA execution uses `--device cuda --compute-type float16` after compatible CUDA libraries are available. Apple Silicon uses CPU in this implementation.
- Translate only when explicitly requested, with `--translate` for English translation. If the user wants both the original and English version, use separate output roots so the two exports are retained. Successful reruns for the same source overwrite that source's transcript files.
- The first run of each model needs a weight download. Reuse the existing `.models/` cache; `--offline` prohibits media and model downloads. For operation without a network request, use `--offline` with a local file or previously cached URL/audio and an already cached model. No paid transcription API or API key is involved.
- Let long jobs finish and relay meaningful progress. After interruption, rerun with the same settings to reuse completed source/audio; inference restarts from the beginning. Do not run simultaneous jobs for the same source or remove a lock owned by a running job.

The CLI validates the source, downloads or reuses permitted media, produces 16 kHz mono audio, transcribes locally, and exports UTF-8 Markdown/TXT/SRT plus metadata. Keep the recognized text intact, including Telugu and Devanagari characters. The model can still make errors; do not claim that uncertainty filters guarantee hallucination-free or complete recognition. The single language in metadata is the initial detected or forced language, not a complete list of languages in mixed speech. Long recordings require memory for the full decoded waveform and accumulated segments.

## Handle failure and return results

For inaccessible, private, deleted, restricted, login-required, or unsupported media, explain the actual download error and offer `--file` for an authorized local copy. Do not add browser cookies, sign in, bypass access controls, or fetch credentials. A dependency or extractor update may resolve a technical failure; avoid repeated unchanged retries. The README documents current YouTube Deno/EJS setup.

After a successful exit, verify that `transcript.md`, `transcript.txt`, `transcript.srt`, `metadata.json`, and `audio.wav` exist at the printed output paths. Inspect metadata and a short transcript excerpt to confirm that recognized text was exported. Do not represent silent/empty output or old files left by a failed run as a new successful transcription.

Return clickable **absolute paths** to Markdown and TXT, with SRT/metadata when useful, and state the detected or requested language and any material warning. Distinguish a completed transcription from a dependency check, mock test, or interrupted run. Do not paste a long transcript into the chat unless requested, and do not send or upload it elsewhere without user instruction.

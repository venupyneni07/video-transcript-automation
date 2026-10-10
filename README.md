# Video Transcript Automation

Transcribe an accessible video link or a directly supplied video/audio file into timestamped Markdown, clean TXT, SRT subtitles, and JSON metadata. Links can come from YouTube, Instagram, other sites supported by yt-dlp, or direct media downloads such as MP4, WebM, and MOV. Codex can also process an attached video when its local file is available to the execution environment. Speech recognition runs locally with `faster-whisper`; no transcription API key or paid transcription service is required.

The defaults are the multilingual `small` model, CPU execution, `int8` compute, and automatic language detection. Telugu, English, Hindi, and mixed-language speech are supported on a best-effort basis. The workflow requests transcription in the spoken language and keeps Unicode text intact. It does not summarize or deliberately translate the recording. Speech recognition can still miss words, misidentify languages, transliterate, or hallucinate; review important passages against the audio. Larger models may improve recognition but require more memory and time.

## Setup

Use Python 3.11 or newer. Python 3.12 or 3.13 is a practical choice when binary dependencies are not yet available for a newer Python release. Install both `ffmpeg` and `ffprobe` on `PATH`; this project uses their command-line tools even though faster-whisper also includes an audio decoder.

Clone the repository, then run the setup commands for your operating system from its root directory:

```bash
git clone https://github.com/venupyneni07/video-transcript-automation.git
cd video-transcript-automation
```

### macOS

With [Homebrew](https://brew.sh/) installed:

```bash
brew install python@3.13 ffmpeg deno
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python transcribe.py --check
```

Apple Silicon uses the CPU in this implementation. The CTranslate2 backend used here does not expose an Apple Metal/MPS device; `--device cuda` is for compatible NVIDIA hardware.

### Windows PowerShell

Install Python, FFmpeg, and the JavaScript runtime used for YouTube:

```powershell
winget install --exact --id Python.Python.3.13
winget install --exact --id Gyan.FFmpeg
winget install --exact --id DenoLand.Deno
```

Open a new PowerShell window so the updated `PATH` is available, then enter your copy of `video-transcript-automation` and run:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe transcribe.py --check
.\.venv\Scripts\python.exe transcribe.py "VIDEO_URL"
```

Using the virtual environment's Python directly avoids PowerShell activation-policy changes. The Windows CTranslate2 wheel may also need the [Microsoft Visual C++ runtime](https://opennmt.net/CTranslate2/installation.html).

### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install python3 python3-venv ffmpeg
python3 --version
```

Ensure that Python is at least 3.11; on an older distribution, install a supported Python release before continuing. Install a current Deno binary using the [official Deno installation instructions](https://docs.deno.com/runtime/getting_started/installation/), and put it on `PATH` for YouTube downloads. Then:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python transcribe.py --check
```

Install from the complete `requirements.txt`: it caps PyAV below version 19 because faster-whisper 1.2.1 still calls an argument removed by PyAV 19. This compatibility issue was reproduced during the real speech test and is tracked [upstream](https://github.com/SYSTRAN/faster-whisper/issues/1589). The decoder compatibility test exercises this boundary without downloading a model.

### YouTube JavaScript runtime

Recent yt-dlp releases require a supported JavaScript runtime and the matching EJS challenge-solver package for full YouTube support. The requirement `yt-dlp[default]` installs the EJS package. Deno is recommended; this project's downloader explicitly enables Deno and Node when found on `PATH`. Their versions must meet yt-dlp's requirements. Follow the [official yt-dlp EJS guide](https://github.com/yt-dlp/yt-dlp/wiki/EJS) if runtime requirements change.

`--check` reports Python, required packages, FFmpeg/ffprobe, and available JavaScript runtimes. It does not download media or a model, and it cannot prove that a particular URL is accessible or that a GPU is configured correctly. Missing required dependencies produce a nonzero exit status; an optional runtime warning does not prevent local-file transcription.

## Run

After activating the virtual environment on macOS/Linux:

```bash
python transcribe.py "VIDEO_URL"
python transcribe.py "VIDEO_URL" --model medium
python transcribe.py "VIDEO_URL" --language te --keep-video
python transcribe.py "https://example.com/videos/myvideo.mp4"
python transcribe.py "myvideo.mp4"
python transcribe.py --file "myvideo.mp4"
python transcribe.py --file "myrecording.wav" --language Hindi
```

Replace `VIDEO_URL` or the example direct-download URL with your complete link, enclosed in quotes. The CLI accepts standard HTTP(S) links and uses yt-dlp's built-in site and generic extractors. YouTube watch/Shorts links and Instagram Reel links remain supported, alongside other extractable video pages and direct media links. Preserve the entire query string for download links that need it.

This does not guarantee that every link works. Sites change, some pages do not expose downloadable media, and some content requires access the downloader does not have. Even an entry in [yt-dlp's supported-sites list](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md) is not a guarantee of current availability. One video is processed per command; playlists and live streams are outside this workflow. If extraction fails, supply an authorized local copy with `--file`.

For a local video or audio file, both `python transcribe.py "myvideo.mp4"` and `python transcribe.py --file "myvideo.mp4"` work. Use the actual downloaded or uploaded file path; a chat attachment must be accessible as a file before the Python CLI can read it.

On Windows replace `python` with `.\.venv\Scripts\python.exe`. Without activation on macOS/Linux, use `.venv/bin/python`.

| Option | Behavior |
| --- | --- |
| `URL`, `PATH`, or `--file PATH` | Supply one supported HTTP(S) video link or one existing local video/audio file. |
| `--model small` | Default; also accepts `tiny`, `base`, `medium`, and `large-v3`. All are multilingual variants. |
| `--language auto` | Default automatic detection. Use `te`, `en`, `hi` or `Telugu`, `English`, `Hindi` to give a language hint. |
| `--keep-video` | Retain downloaded source media after a successful run. Local originals are always kept. |
| `--device cpu` | Default; also accepts `cuda` or `auto`. |
| `--compute-type int8` | Default; also accepts `float16`, `int8_float16`, `float32`, or `auto`. Hardware support varies. |
| `--cpu-threads N` | Set the CPU thread count. |
| `--output-dir PATH` | Write under a different output root. The default is this project's `output/`. |
| `--model-dir PATH` | Use a different model cache. The default is this project's `.models/`. |
| `--offline` | Require cached Whisper weights and local/cached media; prohibit URL access. |
| `--translate` | Explicitly translate speech into English instead of preserving the spoken language. |
| `--check` | Check dependencies without requiring a URL or file. |
| `--help` | Show the complete CLI help. |

The first use of each model downloads its weights from Hugging Face and requires network access and sufficient disk space. Later runs reuse the model cache. To transcribe without network access, use a local file and a model that was previously downloaded (or rerun a URL with valid cached audio):

```bash
python transcribe.py --file "myvideo.mp4" --model small --offline
```

For mixed-language audio, leave `--language auto` initially. The implementation enables multilingual decoding when the installed faster-whisper version provides it. Automatic detection and switching are imperfect, especially for short clips, names, background music, overlapping speakers, and quiet speech. A manual language hint can help a predominantly Telugu or Hindi recording; it does not supply multiple language labels or guarantee code-switching accuracy. The language shown in the Markdown header and metadata is the initial detected language or the forced language selection, not an inventory of every language spoken. A forced language's reported probability is not an independently measured confidence score. No speaker diarization is provided.

### Optional NVIDIA GPU

Install a compatible NVIDIA driver and the libraries required by your installed faster-whisper/CTranslate2 version. Current faster-whisper guidance calls for CUDA 12 cuBLAS and cuDNN 9; consult its [GPU setup documentation](https://github.com/SYSTRAN/faster-whisper#gpu) before changing GPU libraries. Then run:

```bash
python transcribe.py "VIDEO_URL" --device cuda --compute-type float16 --model medium
```

Use `--device cpu --compute-type int8` if CUDA is unavailable. `--device auto` selects an available backend but cannot repair missing CUDA libraries. The default CPU mode requires no GPU setup.

## Outputs and reuse

The command prints absolute paths after successful export. Each source gets a folder based on a sanitized title and stable identifier:

```text
output/
└── example_title--youtube-VIDEO_ID/
    ├── audio.wav
    ├── transcript.md
    ├── transcript.txt
    ├── transcript.srt
    ├── metadata.json
    ├── source.json
    ├── audio-cache.json
    └── media.download.json
```

The exact title and identifier depend on the source. Local files use a content hash so changed file contents get a different identity. YouTube and Instagram use their video identifiers; other links use a stable URL-derived identity. Download query strings are preserved when needed. `source.json`, `audio-cache.json`, and the URL downloader's `media.download.json` support cache validation; they are not transcripts.

| File | Contents |
| --- | --- |
| `transcript.md` | Title, source, language, duration, and complete recognized segments with timestamps. |
| `transcript.txt` | Recognized speech text only, without mandatory timestamps or metadata. |
| `transcript.srt` | Numbered subtitles with start/end timestamps. |
| `metadata.json` | Source and transcription settings, timing, and recognition metadata. |
| `audio.wav` | The 16 kHz mono PCM audio used for transcription and later reuse. |

All text exports use UTF-8, preserving Telugu and Devanagari characters. TXT is the easiest file to paste into ChatGPT. No transcript is replaced with a summary.

A valid cached WAV is reused on subsequent runs, avoiding another download or conversion. Usable downloaded source media is also reused when available. Downloaded source media is removed only after successful export unless `--keep-video` is set; the user's original local file is never removed. WAV audio remains available for reviewing the transcript. If you add `--keep-video` on a later run and the source video was removed, the CLI downloads it again; this requires network access and cannot run with `--offline` unless that media is still cached.

Successful reruns with a new model, language, or `--translate` replace the existing transcript files for that source. Use a separate output root to retain both versions:

```bash
python transcribe.py --file "myvideo.mp4" --language te --output-dir output-original
python transcribe.py --file "myvideo.mp4" --translate --output-dir output-english
```

Failures during download, conversion, or recognition preserve the last successful transcript. Exports are staged and backed up before replacement; caught write errors and interruptions roll back the prior set. Metadata is committed last. If storage problems prevent rollback, the error identifies preserved recovery backups. A forced process kill or hardware failure can still leave an incomplete export set, so the command must finish successfully before you treat the new files as complete.

The workflow processes long recordings with segment progress; recognition resumes from the beginning after interruption while reusing completed media/audio. Faster-whisper decodes the full waveform into memory, and the application collects recognized segments before export, so memory use grows with recording length; this is not a constant-memory streaming pipeline. Download retries are bounded. Same-source locks prevent simultaneous jobs from writing the same result directory.

## Use through Codex

The project contains `.agents/skills/video-to-transcript/SKILL.md`. Open **the cloned repository's root directory** as the Codex working directory. Codex searches `.agents/skills` from the working directory upward; it does not discover arbitrary nested child projects. This discovery behavior and explicit invocation are documented in [official OpenAI skills documentation](https://learn.chatgpt.com/docs/build-skills).

After setup, ask:

```text
Transcribe this video: VIDEO_URL
```

Or attach a video and ask:

```text
Transcribe this video.
```

Codex uses the attachment's actual available local path with `--file`. If the execution environment cannot access the attachment, it will ask for an accessible local path or upload; you do not need to provide the file again when it is already available.

Or explicitly invoke the skill:

```text
$video-to-transcript Transcribe VIDEO_URL in the original spoken language.
$video-to-transcript Transcribe the attached video into text.
$video-to-transcript Transcribe the local file /absolute/path/myvideo.mp4 using medium.
```

The skill locates the project, runs its virtual environment's interpreter, keeps the recognized transcript intact, verifies the outputs, and returns absolute file links. If the skill does not appear, restart Codex and confirm the selected working directory. No global skill installation is required.

### Optional use from other projects

To make a personal copy available across projects, copy the skill folder into the user skill directory documented by Codex, currently `~/.agents/skills`. Run these commands from the cloned repository's root directory only if you want that global installation; the project setup does not modify your global skills:

```bash
mkdir -p "$HOME/.agents/skills"
cp -R .agents/skills/video-to-transcript "$HOME/.agents/skills/"
export VIDEO_TRANSCRIBER_HOME="$(pwd -P)"
```

On Windows PowerShell, from your project folder:

```powershell
$transcriberRoot = (Get-Location).Path
New-Item -ItemType Directory -Force "$env:USERPROFILE\.agents\skills"
Copy-Item -Recurse ".agents\skills\video-to-transcript" "$env:USERPROFILE\.agents\skills\"
$env:VIDEO_TRANSCRIBER_HOME = $transcriberRoot
```

`VIDEO_TRANSCRIBER_HOME` is the skill's project-location hint, not a transcription API setting. It must be visible to the environment in which Codex runs shell commands. A desktop app launched separately may not inherit an export from your terminal; in that case include the absolute project path in your transcription request, or configure that variable in its execution environment. Keep the original project and virtual environment in place. Updating the project-local `SKILL.md` does not automatically update a copied global version.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| `ffmpeg` or `ffprobe` is missing | Install FFmpeg, reopen the terminal, and rerun `python transcribe.py --check`. Both binaries must be on the command's `PATH`. |
| A Python package is missing | Use the project's virtual environment and run `python -m pip install -r requirements.txt`. |
| No compatible wheel or native build failure | Try a supported 64-bit Python 3.12/3.13 interpreter and recreate the virtual environment; check upstream platform support. |
| YouTube challenge/runtime warning | Verify `deno --version`; update the paired packages with `python -m pip install --upgrade "yt-dlp[default]"`. Follow the EJS guide linked above. |
| Private, deleted, login-required, region-restricted, or rate-limited media | The downloader cannot make the content public. Use an accessible public link or an authorized local copy with `--file`. |
| Instagram download fails despite a public page | Instagram may deny anonymous requests or change extraction behavior. Update yt-dlp once, then use an authorized local file if the failure persists. |
| Another website or direct-download link fails | Check that the complete HTTP(S) link is current and contains any required query string. The installed extractor must support the site's media. If it cannot access the video, use an authorized local copy. |
| An attached video is not accessible to Codex | Supply its real local file path or an upload the execution environment can access. A filename alone does not make the file available. |
| Model download fails | Check network access and disk space. `--offline` works only after that model has been cached. |
| CUDA/library error | Use CPU/int8, or install the GPU libraries matching the current backend. |
| No audio stream, corrupt media, or empty/silent audio | Supply a media file with a decodable audio track. No recognized speech is reported as a failure, not fabricated transcript text. |
| Incorrect language or unclear transcription | Retry with the appropriate language hint or a larger model, then compare the result with `audio.wav`. |
| Long recording appears slow | CPU inference may take substantial time, particularly with larger models. Watch progress and leave sufficient RAM and disk space. |
| Source reported as already running after an abrupt process kill | Check that no process is still working on that source before removing its stale lock in the output root's `.locks/` directory. |

Use media that you have permission to process and respect platform terms. The application does not import browser cookies, sign into accounts, or bypass DRM, paywalls, privacy controls, or authentication restrictions. The Python CLI stores audio and transcripts locally and does not upload them to a transcription service. Requests to download media or model weights still contact their respective hosts.

## Development and tests

```bash
python -m pytest -q
python transcribe.py --help
python transcribe.py --check
```

Unit tests use mocked network/model dependencies and locally generated audio. Tests using actual FFmpeg require it on `PATH`. These checks exercise implementation behavior; they do not establish speech-recognition accuracy across languages or guarantee ongoing support for remote platforms.

### Validation status

Validated on **10 October 2026**, macOS Apple Silicon, Python **3.13.12**. The validation environment used a project virtual environment with yt-dlp **2026.8.19**, faster-whisper **1.2.1**, CTranslate2 **4.8.2**, and PyAV **18.1.0**. FFmpeg/ffprobe and Node **25.7.0** were available in that environment.

- `python -m pytest -q`: **168 passed**, covering mocked downloads and inference, URL validation, signed download queries, playlist rejection before media transfers, embedded-video resolution, retry/cache behavior, positional/explicit local inputs, interruption handling, export rollback, Unicode/timestamps, real generated-audio conversion, and the actual Whisper/PyAV decoder boundary. The loopback HTTP integration test needs permission to bind localhost; restricted environments report an explicit skip.
- `python transcribe.py --check`, `python -m pip check`, and compilation checks passed. The official skill-creator validator reported **Skill is valid!** The skill's project/path resolution and command construction were also checked independently.
- Real Whisper inference passed on a locally authored English sentence synthesized with macOS `say`, both as AIFF audio and as an MP4 with a generated black video track. Both runs used multilingual `tiny`, CPU/int8, automatic language detection, and `--offline` after the model download. The sentence was recognized in full, with punctuation and `nine` normalized to `9`; three segments were exported. Completed audio reuse was exercised by the retry.
- Broader input support was verified with a real direct HTTP MP4 download from a temporary localhost server, query-string preservation, ffprobe stream validation, and cache reuse after server shutdown. A separate full direct-link-to-transcript run on the authored speech MP4 succeeded with cached `tiny` weights and `HF_HUB_OFFLINE=1`. The positional local-file command also passed real inference. The temporary test server has been shut down.
- Real public YouTube/Instagram/other-site transfers, Telugu/Hindi/mixed-language recognition accuracy, GPU execution, other OS installations, very long recordings, and larger model inference **remain untested**. Local HTTP tests do not establish external platform availability. Only `tiny` was cached in the validation environment; a fresh clone includes no model weights. Each model, including the default `small`, needs an initial download before it can be used offline.

The manually generated speech samples, transcripts, virtual environment, and model cache are excluded from the repository. To repeat the automated checks in a fresh clone, complete setup above and run:

```bash
python -m pytest -q
```

The pytest fixtures generate their own media and mock model inference, so the automated suite does not require the original validation artifacts or external model downloads. Real transcription can be checked separately with an authorized local recording using the commands in the Run section.

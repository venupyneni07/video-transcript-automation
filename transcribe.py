#!/usr/bin/env python3
"""Run the local video transcription CLI without an editable package install."""

import sys
from pathlib import Path

if sys.version_info < (3, 11):
    print("Python 3.11 or newer is required. Python 3.13 is recommended.", file=sys.stderr)
    sys.exit(2)

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from video_transcriber.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())

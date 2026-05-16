"""
audio_to_srt.py — Automatic Speech Recognition → SRT Subtitle Generator
=========================================================================
Converts audio files (MP3, WAV, M4A, FLAC) to timestamped .srt subtitle
files using OpenAI Whisper via the HuggingFace Transformers pipeline.

Dependencies (install before running):
    pip install transformers torch torchaudio accelerate tqdm
    pip install ffmpeg-python librosa soundfile

    # Also requires ffmpeg binary on your system PATH:
    #   Ubuntu/Debian : sudo apt install ffmpeg
    #   macOS         : brew install ffmpeg
    #   Windows       : https://ffmpeg.org/download.html

Supported models (--model flag):
    openai/whisper-tiny        ~39M  params  fastest, lowest accuracy
    openai/whisper-base        ~74M  params  good balance (default)
    openai/whisper-small       ~244M params
    openai/whisper-medium      ~769M params
    openai/whisper-large-v3    ~1.5B params  highest accuracy, slowest

Usage examples:
    # Basic — uses whisper-base, auto-detects GPU
    python audio_to_srt.py podcast.mp3

    # Choose model size
    python audio_to_srt.py lecture.wav --model openai/whisper-large-v3

    # Specify output path and language
    python audio_to_srt.py interview.flac --output interview.srt --language id

    # Force CPU even if GPU is available
    python audio_to_srt.py clip.m4a --device cpu

    # Batch: multiple files at once
    python audio_to_srt.py ep1.mp3 ep2.mp3 ep3.mp3 --model openai/whisper-small
"""

from __future__ import annotations

import argparse
import sys
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import torch
from tqdm import tqdm
from transformers import pipeline, Pipeline

import librosa
import soundfile as sf

# ---------------------------------------------------------------------------
# Dependency checks — fail early with a helpful message
# ---------------------------------------------------------------------------

def _require(package: str, install_hint: str) -> None:
    """Import-check a package and print an actionable error if missing."""
    import importlib
    if importlib.util.find_spec(package) is None:
        print(f"[ERROR] Missing package '{package}'. Install with:\n  {install_hint}")
        sys.exit(1)


_require("transformers", "pip install transformers")
_require("torch",        "pip install torch torchaudio")
_require("tqdm",         "pip install tqdm")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".webm"}

DEFAULT_MODEL        = "openai/whisper-base"
CHUNK_LENGTH_S       = 30          # seconds per chunk fed to Whisper
BATCH_SIZE           = 8           # parallel chunks (reduce if OOM on GPU)
STRIDE_LENGTH_S      = 5           # overlap between chunks for better continuity
MAX_WORDS_PER_BLOCK  = 16          # max words per SRT subtitle block

# ---------------------------------------------------------------------------
# Progress callback types
# ---------------------------------------------------------------------------

@dataclass
class TranscriptionProgress:
    """
    Immutable snapshot of transcription progress, passed to callbacks.

    Attributes:
        stage:                  Current processing stage.
                                One of: 'validating', 'loading_audio',
                                'transcribing', 'formatting', 'saving',
                                'complete', 'error'.
        percent:                Estimated completion percentage (0.0 – 100.0).
        elapsed_seconds:        Wall-clock seconds since transcription started.
        audio_duration_seconds: Total audio duration in seconds (0.0 if unknown).
        total_segments:         Number of segments produced so far.
        file_name:              Basename of the audio file being processed.
        message:                Human-readable status message.
    """
    stage:                  str   = "validating"
    percent:                float = 0.0
    elapsed_seconds:        float = 0.0
    audio_duration_seconds: float = 0.0
    total_segments:         int   = 0
    file_name:              str   = ""
    message:                str   = ""


# Callable that receives a TranscriptionProgress snapshot.
ProgressCallback = Callable[[TranscriptionProgress], None]


def _default_progress_callback(progress: TranscriptionProgress) -> None:
    """Built-in callback that prints a one-line progress update to stdout."""
    bar_width = 30
    filled    = int(bar_width * progress.percent / 100)
    bar       = "█" * filled + "░" * (bar_width - filled)
    print(
        f"\r[{bar}] {progress.percent:5.1f}% │ "
        f"{progress.stage:<14} │ {progress.message}",
        end="", flush=True,
    )
    if progress.stage in ("complete", "error"):
        print()  # newline after final update


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _get_audio_duration(file_path: Path) -> float:
    """
    Return the duration of an audio file in seconds.

    Tries librosa first (accurate), falls back to a rough estimate
    based on file size if librosa is unavailable.
    """
    info = sf.info(str(file_path))
    return float(info.duration)

# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def detect_device() -> str:
    """Return 'cuda', 'mps', or 'cpu' based on hardware availability."""
    if torch.cuda.is_available():
        return "cuda"
    # Apple Silicon MPS support (PyTorch ≥ 2.0)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_pipeline(model_name: str, device: str) -> Pipeline:
    """
    Load the Whisper ASR pipeline from HuggingFace.

    Args:
        model_name: HuggingFace model identifier, e.g. 'openai/whisper-base'
        device:     'cuda', 'mps', or 'cpu'

    Returns:
        A HuggingFace transformers ASR Pipeline object.

    Raises:
        SystemExit: if the model cannot be loaded.
    """
    print(f"[INFO] Loading model '{model_name}' on device '{device}' …")
    try:
        torch_dtype = torch.float16 if device == "cuda" else torch.float32

        asr = pipeline(
            task="automatic-speech-recognition",
            model=model_name,
            torch_dtype=torch_dtype,
            device=device,
            model_kwargs={"use_safetensors": True},
        )
        print("[INFO] Model loaded successfully.")
        return asr

    except OSError as exc:
        print(f"[ERROR] Could not load model '{model_name}':\n  {exc}")
        print("  Check your internet connection or try a smaller model with --model.")
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Unexpected error while loading model:\n  {exc}")
        sys.exit(1)


def transcribe_audio(
    file_path: str | Path,
    asr_pipeline: Pipeline,
    language: Optional[str] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> List[dict]:
    """
    Transcribe an audio file and return a list of timed segments.

    Args:
        file_path:         Path to the audio file.
        asr_pipeline:      Pre-loaded HuggingFace ASR pipeline.
        language:          ISO-639-1 language code hint, e.g. 'en', 'id', 'fr'.
                           None = auto-detect.
        progress_callback: Optional callable invoked with a
                           ``TranscriptionProgress`` snapshot at each stage.
                           If *None*, a built-in console progress bar is used.

    Returns:
        List of segment dicts: [{"timestamp": (start, end), "text": "…"}, …]

    Raises:
        FileNotFoundError: if the audio file does not exist.
        ValueError:        if the file extension is unsupported.
    """
    file_path = Path(file_path)
    cb = progress_callback or _default_progress_callback
    start_time = time.time()
    file_name  = file_path.name
    audio_dur  = 0.0

    # --- Helper to emit progress ---
    def _emit(
        stage: str,
        percent: float,
        message: str,
        segments: int = 0,
    ) -> None:
        cb(TranscriptionProgress(
            stage=stage,
            percent=min(percent, 100.0),
            elapsed_seconds=time.time() - start_time,
            audio_duration_seconds=audio_dur,
            total_segments=segments,
            file_name=file_name,
            message=message,
        ))

    # --- Validation ---
    _emit("validating", 0.0, f"Validating '{file_name}' …")

    if not file_path.exists():
        _emit("error", 0.0, f"File not found: {file_path}")
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        _emit("error", 0.0, f"Unsupported type: {file_path.suffix}")
        raise ValueError(
            f"Unsupported file type '{file_path.suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    # --- Get audio duration for progress estimation ---
    _emit("loading_audio", 5.0, "Reading audio duration …")
    audio_dur = _get_audio_duration(file_path)
    _emit("loading_audio", 10.0,
          f"Audio duration: {audio_dur:.1f}s ({audio_dur / 60:.1f} min)")

    # --- Build generation kwargs ---
    generate_kwargs: dict = {"task": "transcribe"}
    if language:
        generate_kwargs["language"] = language

    # --- Background thread: estimate progress while pipeline runs ---
    stop_event       = threading.Event()
    transcribe_start = time.time()

    def _progress_updater() -> None:
        """Periodically fire the callback with an estimated percent."""
        # Conservative speed assumption: ~3× real-time on CPU, ~15× on GPU.
        # We cap estimated progress at 89% so the bar never "completes" early.
        while not stop_event.is_set():
            elapsed = time.time() - transcribe_start
            # Rough estimate — most models process ≥2× real-time even on CPU
            est_total = max(audio_dur / 3.0, 5.0)
            pct = 10.0 + (elapsed / est_total) * 79.0  # map to 10%–89%
            pct = min(pct, 89.0)
            _emit("transcribing", pct,
                  f"Transcribing … {elapsed:.0f}s elapsed")
            stop_event.wait(2.0)  # update every 2 seconds

    updater = threading.Thread(target=_progress_updater, daemon=True)
    updater.start()

    # --- Run pipeline ---
    try:
        result = asr_pipeline(
            str(file_path),
            chunk_length_s=CHUNK_LENGTH_S,
            batch_size=BATCH_SIZE,
            stride_length_s=STRIDE_LENGTH_S,
            return_timestamps=True,        # required for SRT timing
            generate_kwargs=generate_kwargs,
        )
    finally:
        stop_event.set()
        updater.join(timeout=2.0)

    elapsed = time.time() - start_time
    chunks: list = result.get("chunks", [])

    _emit("transcribing", 90.0,
          f"Transcription done in {elapsed:.1f}s — {len(chunks)} segments",
          segments=len(chunks))

    _emit("complete", 100.0,
          f"Finished '{file_name}' — {len(chunks)} segments in {elapsed:.1f}s",
          segments=len(chunks))

    return chunks


def _seconds_to_srt_timestamp(seconds: float) -> str:
    """
    Convert a float seconds value to SRT timestamp format.

    Example:
        3723.456  →  '01:02:03,456'
    """
    if seconds is None or seconds < 0:
        seconds = 0.0
    hours   = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs    = int(seconds % 60)
    millis  = int(round((seconds - int(seconds)) * 1000))
    # Guard against millisecond rounding to 1000
    if millis >= 1000:
        millis = 999
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _split_long_segments(
    segments: List[dict],
    max_words: int = MAX_WORDS_PER_BLOCK,
) -> List[dict]:
    """
    Split segments whose text exceeds *max_words* into smaller sub-segments.

    Timestamps are distributed proportionally based on word count so that
    each resulting block covers a fair share of the original time span.

    Args:
        segments:  Original segment list from Whisper.
        max_words: Maximum number of words allowed per subtitle block.

    Returns:
        A new list of segments, each containing at most *max_words* words.
    """
    result: List[dict] = []

    for seg in segments:
        timestamp = seg.get("timestamp", (0.0, 0.0))
        text = seg.get("text", "").strip()
        words = text.split()

        if len(words) <= max_words:
            result.append(seg)
            continue

        # --- Proportional time splitting ---
        start_s = timestamp[0] if timestamp[0] is not None else 0.0
        end_s   = timestamp[1] if timestamp[1] is not None else start_s + 2.0
        if end_s <= start_s:
            end_s = start_s + 0.5

        total_words = len(words)
        duration    = end_s - start_s

        for chunk_start_idx in range(0, total_words, max_words):
            chunk_words = words[chunk_start_idx : chunk_start_idx + max_words]
            chunk_end_idx = min(chunk_start_idx + len(chunk_words), total_words)

            # Proportional timestamps
            sub_start = start_s + duration * (chunk_start_idx / total_words)
            sub_end   = start_s + duration * (chunk_end_idx   / total_words)

            result.append({
                "timestamp": (sub_start, sub_end),
                "text": " ".join(chunk_words),
            })

    return result


def _wrap_text(text: str, max_chars_per_line: int = 42) -> str:
    """
    Wrap subtitle text into at most 2 lines for comfortable on-screen reading.

    Standard subtitle best-practice is ≤42 characters per line and a maximum
    of 2 lines per block.  This function splits roughly in the middle at a
    word boundary when the text exceeds *max_chars_per_line*.

    Args:
        text:               The subtitle text (single paragraph).
        max_chars_per_line:  Soft character limit that triggers wrapping.

    Returns:
        The text, potentially with a single newline inserted.
    """
    if len(text) <= max_chars_per_line:
        return text

    words = text.split()
    if len(words) <= 1:
        return text

    # Find the split point closest to the middle
    mid = len(text) // 2
    best_pos = None
    best_dist = len(text)

    pos = 0
    for i, word in enumerate(words[:-1]):
        pos += len(word)
        dist = abs(pos + i - mid)       # +i accounts for spaces
        if dist < best_dist:
            best_dist = dist
            best_pos  = i
        pos += 0  # space counted via index offset

    if best_pos is not None:
        line1 = " ".join(words[: best_pos + 1])
        line2 = " ".join(words[best_pos + 1 :])
        return f"{line1}\n{line2}"

    return text


def segments_to_srt(segments: List[dict]) -> str:
    """
    Convert a list of Whisper segments into a valid SRT-formatted string.

    Long segments are automatically split so that each subtitle block
    contains at most MAX_WORDS_PER_BLOCK words, and text is wrapped into
    two lines for comfortable reading.

    Args:
        segments: List of dicts with 'timestamp' (tuple) and 'text' (str).
                  timestamp = (start_seconds, end_seconds)

    Returns:
        A string in SRT format ready to be saved as a .srt file.

    Example output block:
        1
        00:00:00,000 --> 00:00:04,320
        Hello and welcome
        to the podcast.

    """
    if not segments:
        return ""

    # Split segments that exceed the word limit
    segments = _split_long_segments(segments)

    srt_blocks: List[str] = []

    for index, segment in enumerate(segments, start=1):
        timestamp = segment.get("timestamp", (0.0, 0.0))
        text      = segment.get("text", "").strip()

        # Whisper occasionally returns None for end timestamp on the last chunk
        start_s: float = timestamp[0] if timestamp[0] is not None else 0.0
        end_s:   float = timestamp[1] if timestamp[1] is not None else start_s + 2.0

        # Ensure start < end (safety clamp)
        if end_s <= start_s:
            end_s = start_s + 0.5

        start_ts = _seconds_to_srt_timestamp(start_s)
        end_ts   = _seconds_to_srt_timestamp(end_s)

        # Wrap text into 2 lines for readability
        wrapped_text = _wrap_text(text)

        block = f"{index}\n{start_ts} --> {end_ts}\n{wrapped_text}"
        srt_blocks.append(block)

    # SRT blocks are separated by a blank line
    return "\n\n".join(srt_blocks) + "\n"


def save_srt(srt_content: str, output_path: str | Path) -> Path:
    """
    Write SRT content to a file.

    Args:
        srt_content: The formatted SRT string.
        output_path: Destination file path (will be created/overwritten).

    Returns:
        Resolved Path of the saved file.

    Raises:
        IOError: if the file cannot be written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        output_path.write_text(srt_content, encoding="utf-8")
        print(f"[INFO] SRT saved → {output_path.resolve()}")
        return output_path.resolve()
    except IOError as exc:
        print(f"[ERROR] Could not write SRT file:\n  {exc}")
        raise


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def build_output_path(audio_path: Path, output_arg: Optional[str]) -> Path:
    """Derive the .srt output path from the audio filename if not specified."""
    if output_arg:
        p = Path(output_arg)
        if p.suffix.lower() != ".srt":
            p = p.with_suffix(".srt")
        return p
    return audio_path.with_suffix(".srt")


def process_file(
    audio_path: Path,
    asr_pipeline: Pipeline,
    output_arg: Optional[str],
    language: Optional[str],
    progress_callback: Optional[ProgressCallback] = None,
) -> None:
    """Transcribe a single file and save its .srt output."""
    try:
        segments = transcribe_audio(
            audio_path, asr_pipeline, language,
            progress_callback=progress_callback,
        )
        srt_content = segments_to_srt(segments)

        if not srt_content.strip():
            print(f"[WARN] No speech detected in '{audio_path.name}'. SRT not saved.")
            return

        out_path = build_output_path(audio_path, output_arg)
        save_srt(srt_content, out_path)

    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
    except ValueError as exc:
        print(f"[ERROR] {exc}")
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Failed to process '{audio_path.name}':\n  {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="audio_to_srt",
        description="Convert audio files to SRT subtitles using HuggingFace Whisper.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "audio_files",
        nargs="+",
        metavar="AUDIO_FILE",
        help="Path(s) to audio file(s). Supported: mp3 wav m4a flac ogg opus webm",
    )
    parser.add_argument(
        "--model", "-m",
        default=DEFAULT_MODEL,
        metavar="MODEL_ID",
        help=(
            f"HuggingFace Whisper model ID (default: {DEFAULT_MODEL}).\n"
            "Options: whisper-tiny, whisper-base, whisper-small, "
            "whisper-medium, whisper-large-v3"
        ),
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        metavar="OUTPUT.srt",
        help=(
            "Output .srt file path. Only valid when a single input file is given. "
            "For multiple files, .srt is saved alongside each audio file."
        ),
    )
    parser.add_argument(
        "--language", "-l",
        default=None,
        metavar="LANG",
        help=(
            "ISO-639-1 language code to force (e.g. 'en', 'id', 'fr', 'ja'). "
            "Omit for auto-detection."
        ),
    )
    parser.add_argument(
        "--device", "-d",
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Compute device. Defaults to auto-detect (cuda > mps > cpu).",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="audio_to_srt 1.0.0",
    )

    args = parser.parse_args()

    # --- Resolve device ---
    device = args.device or detect_device()
    print(f"[INFO] Using device: {device}")

    # --- Guard: --output only makes sense with one input file ---
    if args.output and len(args.audio_files) > 1:
        print("[WARN] --output is ignored when multiple input files are given.")
        args.output = None

    # --- Load model once, reuse across all files ---
    asr_pipeline = load_pipeline(args.model, device)

    # --- Process files ---
    audio_paths = [Path(f) for f in args.audio_files]

    if len(audio_paths) == 1:
        process_file(audio_paths[0], asr_pipeline, args.output, args.language)
    else:
        print(f"[INFO] Batch mode: processing {len(audio_paths)} files …")
        for audio_path in tqdm(audio_paths, desc="Files", unit="file"):
            process_file(audio_path, asr_pipeline, None, args.language)

    print("[INFO] All done.")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
asr_to_srt.py

Convert audio files to timestamped SRT subtitle files using
OpenAI Whisper from Hugging Face Transformers.

Usage examples:
    python asr_to_srt.py audio.mp3
    python asr_to_srt.py audio.wav -o subtitle.srt
    python asr_to_srt.py audio.m4a --model openai/whisper-large-v3
    python asr_to_srt.py audio.flac --model-size base --chunk-length 30 --batch-size 8
    python asr_to_srt.py audio.mp3 --timestamp-mode word

Supported input:
    MP3, WAV, M4A, FLAC

Dependencies:
    pip install -U transformers torch accelerate soundfile librosa torchcodec ffmpeg-python

System dependency:
    ffmpeg must be installed and available in PATH.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import textwrap
from pathlib import Path
from typing import Any


DEFAULT_AUDIO_FILE: str | None = None

SUPPORTED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac"}

MODEL_SIZE_MAP = {
    "tiny": "openai/whisper-tiny",
    "base": "openai/whisper-base",
    "small": "openai/whisper-small",
    "medium": "openai/whisper-medium",
    "large-v3": "openai/whisper-large-v3",
}


def load_runtime_dependencies():
    """
    Import heavy dependencies only when needed.
    This gives cleaner error messages if packages are missing.
    """
    try:
        import torch
        from transformers import pipeline
    except ModuleNotFoundError as error:
        missing_package = error.name or "unknown"

        raise RuntimeError(
            f"Missing Python dependency: {missing_package}\n\n"
            "Install dependencies with:\n"
            "    pip install -U transformers torch accelerate soundfile librosa torchcodec ffmpeg-python\n"
        ) from error

    return torch, pipeline


def validate_audio_file(file_path: str | Path) -> Path:
    path = Path(file_path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"Input audio file not found: {path}")

    if not path.is_file():
        raise ValueError(f"Input path is not a file: {path}")

    if path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_AUDIO_EXTENSIONS))
        raise ValueError(
            f"Unsupported audio format: {path.suffix}\n"
            f"Supported formats: {supported}"
        )

    return path


def ensure_ffmpeg_available() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg was not found in PATH.\n\n"
            "Install it first:\n"
            "    Ubuntu/Debian: sudo apt-get install -y ffmpeg\n"
            "    macOS: brew install ffmpeg\n"
        )


def get_device_config(torch_module: Any) -> tuple[int, Any]:
    """
    Return device and dtype for Hugging Face pipeline.

    device:
        0  = first CUDA GPU
        -1 = CPU
    """
    if torch_module.cuda.is_available():
        return 0, torch_module.float16

    return -1, torch_module.float32


def build_asr_pipeline(
    model_name: str,
    chunk_length_s: float,
):
    torch_module, pipeline = load_runtime_dependencies()
    device, torch_dtype = get_device_config(torch_module)

    try:
        return pipeline(
            task="automatic-speech-recognition",
            model=model_name,
            device=device,
            torch_dtype=torch_dtype,
            chunk_length_s=chunk_length_s,
        )
    except TypeError:
        return pipeline(
            task="automatic-speech-recognition",
            model=model_name,
            device=device,
            dtype=torch_dtype,
            chunk_length_s=chunk_length_s,
        )
    except Exception as error:
        raise RuntimeError(
            f"Failed to load ASR model: {model_name}\n"
            "Check your internet connection, model name, and available RAM/VRAM."
        ) from error


def clean_text(text: str) -> str:
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def safe_float(value: Any) -> float | None:
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_timestamp(segment: dict[str, Any]) -> tuple[float | None, float | None]:
    timestamp = segment.get("timestamp")

    if isinstance(timestamp, (list, tuple)) and len(timestamp) >= 2:
        return safe_float(timestamp[0]), safe_float(timestamp[1])

    start = safe_float(segment.get("start"))
    end = safe_float(segment.get("end"))

    return start, end


def group_word_chunks(
    word_chunks: list[dict[str, Any]],
    max_duration_s: float = 6.0,
    max_chars: int = 84,
) -> list[dict[str, Any]]:
    """
    Convert word-level chunks into subtitle-sized segments.

    Whisper can return word-level timestamps with return_timestamps="word".
    SRT usually needs readable sentence-like blocks, not one block per word.
    """
    segments: list[dict[str, Any]] = []

    current_words: list[str] = []
    current_start: float | None = None
    current_end: float | None = None

    def flush_current() -> None:
        nonlocal current_words, current_start, current_end

        if not current_words or current_start is None or current_end is None:
            current_words = []
            current_start = None
            current_end = None
            return

        text = clean_text(" ".join(current_words))

        if text:
            segments.append(
                {
                    "text": text,
                    "timestamp": (current_start, current_end),
                }
            )

        current_words = []
        current_start = None
        current_end = None

    for chunk in word_chunks:
        word = clean_text(str(chunk.get("text", "")))

        if not word:
            continue

        word_start, word_end = extract_timestamp(chunk)

        if word_start is None or word_end is None:
            continue

        if current_start is None:
            current_start = word_start

        proposed_text = clean_text(" ".join([*current_words, word]))
        proposed_duration = word_end - current_start

        should_flush_before_add = (
            current_words
            and (
                proposed_duration > max_duration_s
                or len(proposed_text) > max_chars
            )
        )

        if should_flush_before_add:
            flush_current()
            current_start = word_start

        current_words.append(word)
        current_end = word_end

        ends_sentence = bool(re.search(r"[.!?。！？]$", word))
        long_enough = current_start is not None and (word_end - current_start) >= 1.0

        if ends_sentence and long_enough:
            flush_current()

    flush_current()

    return segments


def transcribe_audio(
    file_path: str | Path,
    model_name: str = "openai/whisper-base",
    chunk_length_s: float = 30.0,
    batch_size: int = 8,
    timestamp_mode: str = "chunk",
    language: str | None = None,
    task: str = "transcribe",
) -> list[dict[str, Any]]:
    """
    Transcribe audio and return timestamped segments.

    Args:
        file_path:
            Path to MP3, WAV, M4A, or FLAC file.
        model_name:
            Hugging Face model name, for example:
            openai/whisper-base or openai/whisper-large-v3.
        chunk_length_s:
            Audio chunk length for long-form transcription.
        batch_size:
            Inference batch size. Higher is faster on GPU but uses more memory.
        timestamp_mode:
            "chunk" uses return_timestamps=True.
            "word" uses return_timestamps="word" then groups words into SRT segments.
        language:
            Optional Whisper language hint, for example "indonesian" or "english".
        task:
            Whisper task. Usually "transcribe". Can also be "translate".

    Returns:
        List of segment dictionaries:
            [
                {
                    "text": "...",
                    "timestamp": (start_seconds, end_seconds)
                }
            ]
    """
    audio_path = validate_audio_file(file_path)

    if timestamp_mode not in {"chunk", "word"}:
        raise ValueError("timestamp_mode must be either 'chunk' or 'word'")

    asr_pipeline = build_asr_pipeline(
        model_name=model_name,
        chunk_length_s=chunk_length_s,
    )

    return_timestamps: bool | str

    if timestamp_mode == "word":
        return_timestamps = "word"
    else:
        return_timestamps = True

    generate_kwargs: dict[str, str] = {
        "task": task,
    }

    if language:
        generate_kwargs["language"] = language

    call_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "return_timestamps": return_timestamps,
        "generate_kwargs": generate_kwargs,
    }

    try:
        result = asr_pipeline(str(audio_path), **call_kwargs)
    except Exception as error:
        raise RuntimeError(
            "Failed to transcribe audio.\n"
            "Check that the file is readable, ffmpeg is installed, "
            "and the selected model fits your RAM/VRAM."
        ) from error

    chunks = result.get("chunks")

    if not chunks:
        text = clean_text(result.get("text", ""))

        if not text:
            raise RuntimeError("No transcription result was returned by the model.")

        return [
            {
                "text": text,
                "timestamp": (0.0, 1.0),
            }
        ]

    if timestamp_mode == "word":
        return group_word_chunks(chunks)

    return chunks


def format_srt_timestamp(seconds: float | int | None) -> str:
    """
    Convert seconds to SRT timestamp format:
        HH:MM:SS,mmm
    """
    if seconds is None:
        seconds = 0.0

    seconds = max(float(seconds), 0.0)
    total_milliseconds = int(round(seconds * 1000))

    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, milliseconds = divmod(remainder, 1_000)

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def segments_to_srt(segments: list[dict[str, Any]]) -> str:
    """
    Convert timestamped ASR segments to valid SRT content.
    """
    srt_blocks: list[str] = []
    subtitle_number = 1
    last_end = 0.0

    for segment in segments:
        text = clean_text(str(segment.get("text", "")))

        if not text:
            continue

        start, end = extract_timestamp(segment)

        if start is None:
            start = last_end

        if end is None:
            end = start + 1.0

        if end <= start:
            end = start + 0.5

        wrapped_text = "\n".join(
            textwrap.wrap(
                text,
                width=84,
                break_long_words=False,
                break_on_hyphens=False,
            )
        )

        block = (
            f"{subtitle_number}\n"
            f"{format_srt_timestamp(start)} --> {format_srt_timestamp(end)}\n"
            f"{wrapped_text}\n"
        )

        srt_blocks.append(block)

        subtitle_number += 1
        last_end = end

    if not srt_blocks:
        raise RuntimeError("No valid subtitle segments were generated.")

    return "\n".join(srt_blocks).strip() + "\n"


def save_srt(srt_content: str, output_path: str | Path) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(srt_content, encoding="utf-8")
    return path


def build_default_output_path(input_path: Path) -> Path:
    return input_path.with_suffix(".srt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert an audio file to a timestamped SRT subtitle file "
            "using OpenAI Whisper from Hugging Face."
        )
    )

    parser.add_argument(
        "input",
        nargs="?",
        help="Path to input audio file: MP3, WAV, M4A, or FLAC.",
    )

    parser.add_argument(
        "-o",
        "--output",
        help="Path to output .srt file. Defaults to input filename with .srt extension.",
    )

    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Full Hugging Face model name. "
            "Example: openai/whisper-base or openai/whisper-large-v3. "
            "Overrides --model-size."
        ),
    )

    parser.add_argument(
        "--model-size",
        default="base",
        choices=sorted(MODEL_SIZE_MAP.keys()),
        help="Whisper model size shortcut. Default: base.",
    )

    parser.add_argument(
        "--chunk-length",
        type=float,
        default=30.0,
        help="Chunk length in seconds for long audio. Default: 30.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Inference batch size. Reduce this if you run out of memory. Default: 8.",
    )

    parser.add_argument(
        "--timestamp-mode",
        choices=["chunk", "word"],
        default="chunk",
        help=(
            "Timestamp mode. "
            "'chunk' uses return_timestamps=True. "
            "'word' uses return_timestamps='word' then groups words into SRT blocks. "
            "Default: chunk."
        ),
    )

    parser.add_argument(
        "--language",
        default=None,
        help=(
            "Optional Whisper language hint, e.g. 'indonesian', 'english', 'japanese'. "
            "Leave empty for auto-detection."
        ),
    )

    parser.add_argument(
        "--task",
        choices=["transcribe", "translate"],
        default="transcribe",
        help="Whisper task. Default: transcribe.",
    )

    parser.add_argument(
        "--skip-ffmpeg-check",
        action="store_true",
        help="Skip ffmpeg availability check.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_value = args.input or DEFAULT_AUDIO_FILE

    if not input_value:
        print(
            "Error: no input audio file was provided.\n\n"
            "Example:\n"
            "    python asr_to_srt.py audio.mp3\n",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        input_path = validate_audio_file(input_value)

        if not args.skip_ffmpeg_check:
            ensure_ffmpeg_available()

        model_name = args.model or MODEL_SIZE_MAP[args.model_size]
        output_path = Path(args.output).expanduser().resolve() if args.output else build_default_output_path(input_path)

        print(f"Input audio : {input_path}")
        print(f"Output SRT  : {output_path}")
        print(f"Model       : {model_name}")
        print(f"Timestamp   : {args.timestamp_mode}")
        print("Processing...")

        segments = transcribe_audio(
            file_path=input_path,
            model_name=model_name,
            chunk_length_s=args.chunk_length,
            batch_size=args.batch_size,
            timestamp_mode=args.timestamp_mode,
            language=args.language,
            task=args.task,
        )

        srt_content = segments_to_srt(segments)
        saved_path = save_srt(srt_content, output_path)

        print(f"Done. SRT saved to: {saved_path}")

    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
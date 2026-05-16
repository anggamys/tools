from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

SubtitleFormat = Literal["srt", "vtt", "ass", "ssa"]

SRT_TIMESTAMP_RE = re.compile(
    r"^\s*\d{2}:\d{2}:\d{2},\d{3}\s+-->\s+\d{2}:\d{2}:\d{2},\d{3}.*$"
)

VTT_TIMESTAMP_RE = re.compile(
    r"^\s*\d{2}:\d{2}:\d{2}\.\d{3}\s+-->\s+\d{2}:\d{2}:\d{2}\.\d{3}.*$|"
    r"^\s*\d{2}:\d{2}\.\d{3}\s+-->\s+\d{2}:\d{2}\.\d{3}.*$"
)


@dataclass(slots=True)
class SubtitleSegment:
    """A translatable subtitle unit."""

    text: str
    start_line: int | None = None
    end_line: int | None = None
    ass_line_index: int | None = None
    ass_prefix: str | None = None
    ass_fields: list[str] | None = None
    ass_text_index: int | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class ParsedSubtitle:
    """Parsed subtitle document with original lines and translatable segments."""

    path: Path
    format: SubtitleFormat
    lines: list[str]
    segments: list[SubtitleSegment]
    original_ended_with_newline: bool = True


def read_text_with_fallback(path: Path) -> tuple[str, str]:
    """Read text using UTF-8 first, then common fallbacks."""
    encodings = ("utf-8-sig", "utf-8", "cp1252", "latin-1")

    for encoding in encodings:
        try:
            return path.read_text(encoding=encoding), encoding
        except UnicodeDecodeError:
            continue

    raise UnicodeDecodeError(
        "unknown",
        b"",
        0,
        1,
        f"Unable to decode subtitle file: {path}",
    )


def detect_subtitle_format(path: Path) -> SubtitleFormat:
    suffix = path.suffix.lower()

    if suffix == ".srt":
        return "srt"
    if suffix == ".vtt":
        return "vtt"
    if suffix == ".ass":
        return "ass"
    if suffix == ".ssa":
        return "ssa"

    raise ValueError(
        f"Unsupported subtitle format: {suffix}. "
        "Supported formats: .srt, .vtt, .ass, .ssa"
    )


def parse_subtitle_file(path: str | Path) -> ParsedSubtitle:
    subtitle_path = Path(path).expanduser().resolve()

    if not subtitle_path.exists():
        raise FileNotFoundError(f"Input file not found: {subtitle_path}")

    if not subtitle_path.is_file():
        raise ValueError(f"Input path is not a file: {subtitle_path}")

    subtitle_format = detect_subtitle_format(subtitle_path)
    text, encoding = read_text_with_fallback(subtitle_path)
    logger.info("Read input file using encoding=%s", encoding)

    original_ended_with_newline = text.endswith(("\n", "\r"))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")

    if lines and lines[-1] == "":
        lines.pop()

    if subtitle_format == "srt":
        segments = parse_srt_segments(lines)
    elif subtitle_format == "vtt":
        segments = parse_vtt_segments(lines)
    elif subtitle_format in {"ass", "ssa"}:
        segments = parse_ass_segments(lines)
    else:
        raise ValueError(f"Unsupported subtitle format: {subtitle_format}")

    return ParsedSubtitle(
        path=subtitle_path,
        format=subtitle_format,
        lines=lines,
        segments=segments,
        original_ended_with_newline=original_ended_with_newline,
    )


def parse_srt_segments(lines: list[str]) -> list[SubtitleSegment]:
    segments: list[SubtitleSegment] = []
    index = 0

    while index < len(lines):
        if not lines[index].strip():
            index += 1
            continue

        sequence_line_index = index
        sequence_line = lines[index].strip()

        if not sequence_line.isdigit():
            logger.warning(
                "Expected SRT sequence number at line %d, got: %s",
                index + 1,
                lines[index],
            )
            index += 1
            continue

        index += 1

        if index >= len(lines):
            logger.warning("Missing timestamp after SRT sequence at line %d", sequence_line_index + 1)
            break

        timestamp_line = lines[index]

        if "-->" not in timestamp_line:
            logger.warning("Missing SRT timestamp arrow at line %d", index + 1)
            index += 1
            continue

        if not SRT_TIMESTAMP_RE.match(timestamp_line):
            logger.warning("Malformed SRT timestamp at line %d: %s", index + 1, timestamp_line)

        index += 1
        text_start = index

        while index < len(lines) and lines[index].strip():
            index += 1

        text_end = index
        text_lines = lines[text_start:text_end]
        text = "\n".join(text_lines).strip()

        if text:
            segments.append(
                SubtitleSegment(
                    text=text,
                    start_line=text_start,
                    end_line=text_end,
                    metadata={"sequence": sequence_line},
                )
            )

    return segments


def parse_vtt_segments(lines: list[str]) -> list[SubtitleSegment]:
    segments: list[SubtitleSegment] = []
    index = 0

    while index < len(lines):
        line = lines[index]

        if not line.strip():
            index += 1
            continue

        stripped = line.strip()

        if stripped.startswith(("WEBVTT", "NOTE", "STYLE", "REGION")):
            index += 1
            while index < len(lines) and lines[index].strip():
                index += 1
            continue

        timestamp_index: int | None = None

        if "-->" in line:
            timestamp_index = index
        elif index + 1 < len(lines) and "-->" in lines[index + 1]:
            timestamp_index = index + 1

        if timestamp_index is None:
            index += 1
            continue

        if not VTT_TIMESTAMP_RE.match(lines[timestamp_index]):
            logger.warning("Malformed VTT timestamp at line %d: %s", timestamp_index + 1, lines[timestamp_index])

        text_start = timestamp_index + 1
        text_end = text_start

        while text_end < len(lines) and lines[text_end].strip():
            text_end += 1

        text = "\n".join(lines[text_start:text_end]).strip()

        if text:
            segments.append(
                SubtitleSegment(
                    text=text,
                    start_line=text_start,
                    end_line=text_end,
                )
            )

        index = text_end + 1

    return segments


def parse_ass_segments(lines: list[str]) -> list[SubtitleSegment]:
    segments: list[SubtitleSegment] = []
    in_events_section = False
    format_fields: list[str] | None = None

    for line_index, line in enumerate(lines):
        stripped = line.strip()

        if stripped.lower() == "[events]":
            in_events_section = True
            continue

        if stripped.startswith("[") and stripped.endswith("]") and stripped.lower() != "[events]":
            in_events_section = False
            continue

        if not in_events_section:
            continue

        if stripped.lower().startswith("format:"):
            raw_format = line.split(":", 1)[1]
            format_fields = [field.strip() for field in raw_format.split(",")]
            continue

        if not stripped.lower().startswith("dialogue:"):
            continue

        if not format_fields:
            logger.warning("ASS/SSA Dialogue line found before Format line at line %d", line_index + 1)
            continue

        try:
            text_index = next(
                idx for idx, field_name in enumerate(format_fields)
                if field_name.lower() == "text"
            )
        except StopIteration:
            logger.warning("ASS/SSA Format line has no Text field; skipping line %d", line_index + 1)
            continue

        prefix, payload = line.split(":", 1)
        payload = payload.lstrip()
        fields = payload.split(",", maxsplit=len(format_fields) - 1)

        if len(fields) <= text_index:
            logger.warning("Malformed ASS/SSA Dialogue line at line %d", line_index + 1)
            continue

        text = fields[text_index].replace("\\N", "\n").replace("\\n", "\n")
        text = text.strip()

        if text:
            segments.append(
                SubtitleSegment(
                    text=text,
                    ass_line_index=line_index,
                    ass_prefix=prefix,
                    ass_fields=fields,
                    ass_text_index=text_index,
                )
            )

    return segments


def split_translated_text(text: str, max_line_length: int = 84) -> list[str]:
    """Split translated text into subtitle lines without damaging short subtitles."""
    import textwrap

    normalized = " ".join(text.replace("\r", " ").replace("\n", " ").split())

    if not normalized:
        return [""]

    return textwrap.wrap(
        normalized,
        width=max_line_length,
        break_long_words=False,
        break_on_hyphens=False,
    ) or [normalized]


def render_subtitle(parsed: ParsedSubtitle, translations: list[str]) -> str:
    if len(translations) != len(parsed.segments):
        raise ValueError(
            f"Translation count mismatch: got {len(translations)}, "
            f"expected {len(parsed.segments)}"
        )

    lines = list(parsed.lines)
    paired_segments = list(zip(parsed.segments, translations))

    ass_pairs = [pair for pair in paired_segments if pair[0].ass_line_index is not None]
    range_pairs = [pair for pair in paired_segments if pair[0].start_line is not None and pair[0].end_line is not None]

    for segment, translated_text in ass_pairs:
        assert segment.ass_line_index is not None
        assert segment.ass_prefix is not None
        assert segment.ass_fields is not None
        assert segment.ass_text_index is not None

        fields = list(segment.ass_fields)
        fields[segment.ass_text_index] = translated_text.replace("\n", r"\N")
        lines[segment.ass_line_index] = f"{segment.ass_prefix}: " + ",".join(fields)

    for segment, translated_text in sorted(
        range_pairs,
        key=lambda item: item[0].start_line or 0,
        reverse=True,
    ):
        assert segment.start_line is not None
        assert segment.end_line is not None
        lines[segment.start_line:segment.end_line] = split_translated_text(translated_text)

    rendered = "\n".join(lines)

    if parsed.original_ended_with_newline:
        rendered += "\n"

    return rendered
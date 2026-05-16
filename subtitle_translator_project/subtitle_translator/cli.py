from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from .subtitle_parser import parse_subtitle_file, render_subtitle
from .translator import DEFAULT_MODEL, SubtitleTranslator, TranslationConfig, detect_source_language

logger = logging.getLogger(__name__)


def build_default_output_path(input_path: Path, target_lang: str) -> Path:
    safe_target = target_lang.replace("/", "-").replace(" ", "_")
    return input_path.with_name(f"{input_path.stem}_{safe_target}{input_path.suffix}")


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(levelname)s: %(message)s",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate SRT, VTT, ASS, and SSA subtitle files using Hugging Face models."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input subtitle file path: .srt, .vtt, .ass, or .ssa.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output subtitle file path. Defaults to {original_name}_{target_lang}{same_extension}.",
    )
    parser.add_argument(
        "--source-lang",
        default=None,
        help=(
            "Source language code. Examples: en, id, ja, eng_Latn, ind_Latn. "
            "If omitted, the script auto-detects it using langdetect."
        ),
    )
    parser.add_argument(
        "--target-lang",
        required=True,
        help="Target language code. Examples: id, en, ja, ind_Latn, eng_Latn.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "Hugging Face translation model. Default: facebook/nllb-200-distilled-600M. "
            "Example: Helsinki-NLP/opus-mt-en-id"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of subtitle entries translated per batch. Default: 16.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device: auto, cpu, cuda, -1, 0, 1, ... Default: auto.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum generated token length per subtitle entry. Default: 512.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level. Default: INFO.",
    )

    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)

    start_time = time.perf_counter()

    try:
        input_path = Path(args.input).expanduser().resolve()
        parsed = parse_subtitle_file(input_path)

        if not parsed.segments:
            raise RuntimeError("No translatable subtitle entries were found.")

        source_lang = args.source_lang or detect_source_language(
            [segment.text for segment in parsed.segments]
        )

        output_path = (
            Path(args.output).expanduser().resolve()
            if args.output
            else build_default_output_path(input_path, args.target_lang)
        )

        config = TranslationConfig(
            model_name=args.model,
            source_lang=source_lang,
            target_lang=args.target_lang,
            batch_size=args.batch_size,
            device=args.device,
            max_length=args.max_length,
        )

        translator = SubtitleTranslator(config)
        translations = translator.translate_texts([segment.text for segment in parsed.segments])
        output_content = render_subtitle(parsed, translations)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_content, encoding="utf-8")

        elapsed = time.perf_counter() - start_time

        print()
        print("Translation summary")
        print("-------------------")
        print(f"Input file          : {input_path}")
        print(f"Output file         : {output_path}")
        print(f"Subtitle format     : {parsed.format}")
        print(f"Entries translated  : {len(parsed.segments)}")
        print(f"Source language     : {source_lang}")
        print(f"Target language     : {args.target_lang}")
        print(f"Model used          : {args.model}")
        print(f"Device              : {args.device}")
        print(f"Elapsed time        : {elapsed:.2f}s")

        return 0

    except KeyboardInterrupt:
        logger.error("Cancelled by user.")
        return 130
    except Exception as error:
        logger.error("%s", error)
        return 1


def main() -> None:
    sys.exit(run())


if __name__ == "__main__":
    main()
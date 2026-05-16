#!/usr/bin/env python3
"""
Translate subtitle files using Hugging Face transformer models.

Usage examples:
    python translate_srt.py --input movie.srt --target-lang id
    python translate_srt.py --input movie.srt --target-lang id --model Helsinki-NLP/opus-mt-en-id
    python translate_srt.py --input movie.vtt --source-lang en --target-lang ind_Latn
    python translate_srt.py --input movie.ass --source-lang eng_Latn --target-lang ind_Latn --batch-size 8 --device cuda

Default model:
    facebook/nllb-200-distilled-600M

For Helsinki OPUS-MT models, choose a model that matches the language pair:
    Helsinki-NLP/opus-mt-en-id
    Helsinki-NLP/opus-mt-id-en

Output default:
    {original_name}_{target_lang}{same_extension}
"""

from subtitle_translator.cli import main


if __name__ == "__main__":
    main()
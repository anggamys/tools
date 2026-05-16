from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from tqdm import tqdm
# pyre-ignore [untyped-import]
from langdetect import DetectorFactory, detect
# pyre-ignore [untyped-import]
from langdetect.lang_detect_exception import LangDetectException

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "facebook/nllb-200-distilled-600M"

NLLB_LANGUAGE_MAP: dict[str, str] = {
    "af": "afr_Latn",
    "am": "amh_Ethi",
    "ar": "arb_Arab",
    "az": "azj_Latn",
    "be": "bel_Cyrl",
    "bg": "bul_Cyrl",
    "bn": "ben_Beng",
    "bs": "bos_Latn",
    "ca": "cat_Latn",
    "ceb": "ceb_Latn",
    "cs": "ces_Latn",
    "cy": "cym_Latn",
    "da": "dan_Latn",
    "de": "deu_Latn",
    "el": "ell_Grek",
    "en": "eng_Latn",
    "es": "spa_Latn",
    "et": "est_Latn",
    "eu": "eus_Latn",
    "fa": "pes_Arab",
    "fi": "fin_Latn",
    "fr": "fra_Latn",
    "ga": "gle_Latn",
    "gl": "glg_Latn",
    "gu": "guj_Gujr",
    "he": "heb_Hebr",
    "hi": "hin_Deva",
    "hr": "hrv_Latn",
    "hu": "hun_Latn",
    "hy": "hye_Armn",
    "id": "ind_Latn",
    "is": "isl_Latn",
    "it": "ita_Latn",
    "ja": "jpn_Jpan",
    "jv": "jav_Latn",
    "ka": "kat_Geor",
    "kk": "kaz_Cyrl",
    "km": "khm_Khmr",
    "kn": "kan_Knda",
    "ko": "kor_Hang",
    "lo": "lao_Laoo",
    "lt": "lit_Latn",
    "lv": "lvs_Latn",
    "mk": "mkd_Cyrl",
    "ml": "mal_Mlym",
    "mn": "khk_Cyrl",
    "mr": "mar_Deva",
    "ms": "zsm_Latn",
    "my": "mya_Mymr",
    "ne": "npi_Deva",
    "nl": "nld_Latn",
    "no": "nob_Latn",
    "pl": "pol_Latn",
    "ps": "pbt_Arab",
    "pt": "por_Latn",
    "ro": "ron_Latn",
    "ru": "rus_Cyrl",
    "si": "sin_Sinh",
    "sk": "slk_Latn",
    "sl": "slv_Latn",
    "sq": "als_Latn",
    "sr": "srp_Cyrl",
    "su": "sun_Latn",
    "sv": "swe_Latn",
    "sw": "swh_Latn",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "th": "tha_Thai",
    "tl": "tgl_Latn",
    "tr": "tur_Latn",
    "uk": "ukr_Cyrl",
    "ur": "urd_Arab",
    "vi": "vie_Latn",
    "zh": "zho_Hans",
    "zh-cn": "zho_Hans",
    "zh-tw": "zho_Hant",
}

PROTECTED_RE = re.compile(r"(\{[^}]*\}|<[^>]+>)")


@dataclass(frozen=True, slots=True)
class TranslationConfig:
    model_name: str
    source_lang: str
    target_lang: str
    batch_size: int = 16
    device: str = "auto"
    max_length: int = 512


def detect_source_language(texts: list[str]) -> str:
    sample = " ".join(text.strip() for text in texts if text.strip())
    sample = sample[:5000]

    if not sample:
        raise ValueError("Cannot auto-detect source language because subtitle text is empty.")

    DetectorFactory.seed = 42

    try:
        detected = detect(sample)
    except LangDetectException as error:
        raise RuntimeError("Failed to auto-detect source language.") from error

    logger.info("Auto-detected source language: %s", detected)
    return detected


def to_nllb_language_code(language: str) -> str:
    normalized = language.strip().replace("-", "_")

    if re.match(r"^[a-z]{3}_[A-Za-z]{4}$", normalized):
        return normalized

    lower = language.strip().lower()
    mapped = NLLB_LANGUAGE_MAP.get(lower)

    if mapped:
        return mapped

    raise ValueError(
        f"Unsupported or unknown NLLB language code: {language}. "
        "Use an ISO code supported by this script, or pass the exact NLLB/FLORES code "
        "such as eng_Latn, ind_Latn, jpn_Jpan, or zho_Hans."
    )


def is_nllb_model(model_name: str) -> bool:
    return "nllb" in model_name.lower()


def resolve_device(device: str) -> int:
    if device == "cpu":
        return -1

    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("Missing dependency: torch. Install it with: pip install torch") from error

    if device == "auto":
        return 0 if torch.cuda.is_available() else -1

    if device == "cuda":
        if torch.cuda.is_available():
            return 0

        logger.warning("CUDA was requested but is not available. Falling back to CPU.")
        return -1

    if re.fullmatch(r"-?\d+", device):
        return int(device)

    raise ValueError("Invalid device. Use: auto, cpu, cuda, -1, 0, 1, ...")


def mask_protected_markup(text: str) -> tuple[str, dict[str, str]]:
    replacements: dict[str, str] = {}

    def replace(match: re.Match[str]) -> str:
        token = f"__PH_{len(replacements)}__"
        replacements[token] = match.group(0)
        return token

    return PROTECTED_RE.sub(replace, text), replacements


def restore_protected_markup(text: str, replacements: dict[str, str]) -> str:
    restored = text

    for token, original in replacements.items():
        if token not in restored:
            logger.warning("Protected markup token was not preserved by the model: %s", token)
            continue
        restored = restored.replace(token, original)

    return restored


def should_translate(text: str) -> bool:
    compact = re.sub(PROTECTED_RE, "", text)
    return bool(re.search(
        r"[A-Za-zÀ-ÖØ-öø-ÿĀ-žА-Яа-я\u0370-\u03FF\u0590-\u05FF"
        r"\u0600-\u06FF\u0900-\u097F\u3040-\u30FF\u4E00-\u9FFF]",
        compact,
    ))


class SubtitleTranslator:
    def __init__(self, config: TranslationConfig) -> None:
        self.config = config
        self.device_id = resolve_device(config.device)
        self.pipeline = self._build_pipeline()

    def _build_pipeline(self):
        try:
            from transformers import pipeline
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "Missing dependency: transformers. Install dependencies from requirements.txt."
            ) from error

        logger.info("Loading translation model: %s", self.config.model_name)

        try:
            if is_nllb_model(self.config.model_name):
                src_code = to_nllb_language_code(self.config.source_lang)
                tgt_code = to_nllb_language_code(self.config.target_lang)

                logger.info("Using NLLB language codes: %s -> %s", src_code, tgt_code)

                return pipeline(
                    task="translation",
                    model=self.config.model_name,
                    src_lang=src_code,
                    tgt_lang=tgt_code,
                    device=self.device_id,
                )

            return pipeline(
                task="translation",
                model=self.config.model_name,
                device=self.device_id,
            )
        except Exception as error:
            raise RuntimeError(
                f"Failed to load translation model: {self.config.model_name}. "
                "Check the model name, internet connection, and available RAM/VRAM. "
                "If you use transformers v5, install transformers<5.0.0 because this script uses pipeline('translation')."
            ) from error

    def translate_texts(self, texts: list[str]) -> list[str]:
        if not texts:
            return []

        translations: list[str] = []
        total_batches = (len(texts) + self.config.batch_size - 1) // self.config.batch_size

        for start in tqdm(
            range(0, len(texts), self.config.batch_size),
            total=total_batches,
            desc="Translating",
            unit="batch",
        ):
            original_batch = texts[start:start + self.config.batch_size]
            translatable_indexes: list[int] = []
            masked_batch: list[str] = []
            markup_maps: list[dict[str, str]] = []
            batch_result = list(original_batch)

            for local_index, text in enumerate(original_batch):
                if not should_translate(text):
                    continue

                masked_text, replacements = mask_protected_markup(text)
                translatable_indexes.append(local_index)
                masked_batch.append(masked_text)
                markup_maps.append(replacements)

            if masked_batch:
                translated_batch = self._translate_batch(masked_batch)

                for local_index, translated_text, replacements in zip(
                    translatable_indexes,
                    translated_batch,
                    markup_maps,
                ):
                    batch_result[local_index] = restore_protected_markup(
                        translated_text,
                        replacements,
                    )

            translations.extend(batch_result)

        return translations

    def _translate_batch(self, texts: list[str]) -> list[str]:
        try:
            outputs = self.pipeline(
                texts,
                batch_size=len(texts),
                truncation=True,
                max_length=self.config.max_length,
                clean_up_tokenization_spaces=True,
            )
        except TypeError:
            outputs = self.pipeline(
                texts,
                batch_size=len(texts),
                truncation=True,
                max_length=self.config.max_length,
            )
        except Exception as error:
            raise RuntimeError("Translation inference failed.") from error

        if isinstance(outputs, dict):
            outputs = [outputs]

        normalized_outputs: list[dict[str, Any]] = []

        for item in outputs:
            if isinstance(item, list) and item:
                normalized_outputs.append(item[0])
            elif isinstance(item, dict):
                normalized_outputs.append(item)
            else:
                raise RuntimeError(f"Unexpected translation output format: {item!r}")

        translations: list[str] = []

        for output in normalized_outputs:
            translated_text = output.get("translation_text") or output.get("generated_text")

            if not isinstance(translated_text, str):
                raise RuntimeError(f"Missing translated text in output: {output!r}")

            translations.append(translated_text.strip())

        return translations
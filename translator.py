from deep_translator import GoogleTranslator

# Supported languages mapping (name -> ISO 639-1 code)
LANGUAGES = {
    "afrikaans": "af", "albanian": "sq", "amharic": "am", "arabic": "ar",
    "armenian": "hy", "assamese": "as", "aymara": "ay", "azerbaijani": "az",
    "bambara": "bm", "basque": "eu", "belarusian": "be", "bengali": "bn",
    "bhojpuri": "bho", "bosnian": "bs", "bulgarian": "bg", "catalan": "ca",
    "cebuano": "ceb", "chichewa": "ny", "chinese (simplified)": "zh-CN",
    "chinese (traditional)": "zh-TW", "corsican": "co", "croatian": "hr",
    "czech": "cs", "danish": "da", "dhivehi": "dv", "dogri": "doi", "dutch": "nl",
    "english": "en", "esperanto": "eo", "estonian": "et", "ewe": "ee",
    "filipino": "tl", "finnish": "fi", "french": "fr", "frisian": "fy",
    "galician": "gl", "georgian": "ka", "german": "de", "greek": "el",
    "guarani": "gn", "gujarati": "gu", "haitian creole": "ht", "hausa": "ha",
    "hawaiian": "haw", "hebrew": "iw", "hindi": "hi", "hmong": "hmn",
    "hungarian": "hu", "icelandic": "is", "igbo": "ig", "ilocano": "ilo",
    "indonesian": "id", "irish": "ga", "italian": "it", "japanese": "ja",
    "javanese": "jw", "kannada": "kn", "kazakh": "kk", "khmer": "km",
    "kinyarwanda": "rw", "konkani": "gom", "korean": "ko", "krio": "kri",
    "kurdish (kurmanji)": "ku", "kurdish (sorani)": "ckb", "kyrgyz": "ky",
    "lao": "lo", "latin": "la", "latvian": "lv", "lingala": "ln",
    "lithuanian": "lt", "luganda": "lg", "luxembourgish": "lb", "macedonian": "mk",
    "maithili": "mai", "malagasy": "mg", "malay": "ms", "malayalam": "ml",
    "maltese": "mt", "maori": "mi", "marathi": "mr", "meiteilon (manipuri)": "mni-Mtei",
    "mizo": "lus", "mongolian": "mn", "myanmar": "my", "nepali": "ne",
    "norwegian": "no", "odia (oriya)": "or", "oromo": "om", "pashto": "ps",
    "persian": "fa", "polish": "pl", "portuguese": "pt", "punjabi": "pa",
    "quechua": "qu", "romanian": "ro", "russian": "ru", "samoan": "sm",
    "sanskrit": "sa", "scots gaelic": "gd", "sepedi": "nso", "serbian": "sr",
    "sesotho": "st", "shona": "sn", "sindhi": "sd", "sinhala": "si",
    "slovak": "sk", "slovenian": "sl", "somali": "so", "spanish": "es",
    "sundanese": "su", "swahili": "sw", "swedish": "sv", "tajik": "tg",
    "tamil": "ta", "tatar": "tt", "telugu": "te", "thai": "th", "tigrinya": "ti",
    "tsonga": "ts", "turkish": "tr", "turkmen": "tk", "twi": "ak",
    "ukrainian": "uk", "urdu": "ur", "uyghur": "ug", "uzbek": "uz",
    "vietnamese": "vi", "welsh": "cy", "xhosa": "xh", "yiddish": "yi",
    "yoruba": "yo", "zulu": "zu"
}

def translate_text(text: str, target_lang: str) -> str:
    """
    Translate `text` to target language.
    Accepts either ISO code or full language name.
    Returns original text if 'none', empty, or unsupported.
    """
    if not target_lang or target_lang.lower() in ("none", ""):
        return text

    # Normalize target language: try name -> ISO code
    lang_code = LANGUAGES.get(target_lang.lower(), target_lang)

    try:
        return GoogleTranslator(source="auto", target=lang_code).translate(text)
    except Exception as e:
        return f"[Translation Error: {e}] {text}"




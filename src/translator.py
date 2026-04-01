import os
import re

import ollama

# Ollama client host: full URL or host:port (OLLAMA_HOST is the env var Ollama itself uses).
_raw_host = os.getenv("OLLAMA_HOST", "127.0.0.1:11434").strip()
OLLAMA_URL = _raw_host if _raw_host.startswith("http") else f"http://{_raw_host}"

MODEL_NAME = os.getenv("OLLAMA_MODEL", "deepseek-r1:1.5b")

client = ollama.Client(host=OLLAMA_URL)

classification_context = """
You are a language classifier. Detect the language of the input text and reply
only with the English name of that language.

Example:
INPUT: Bonjour, je m'appelle Bob
OUTPUT: French

INPUT: Können Sie mir bitte helfen?
OUTPUT: German
"""

translation_context = """
You are a highly accurate translator. Translate the input text into English and
reply only with the translated text. Do not include any extra commentary.

Example:
INPUT: Bonjour, je m'appelle Bob
OUTPUT: Hello, my name is Bob

INPUT: Können Sie mir bitte helfen?
OUTPUT: Can you please help me?
"""

# If the model returns more than this many whitespace-separated tokens, treat
# classification as unreliable and keep the original text (robustness).
_MAX_LANGUAGE_LABEL_WORDS = 4


def _strip_reasoning(text: str) -> str:
    """Remove chain-of-thought blocks common in reasoning models (e.g. deepseek-r1)."""
    open_t = "<" + "think" + ">"
    close_t = "<" + "/" + "think" + ">"
    pattern = re.escape(open_t) + r".*?" + re.escape(close_t)
    return re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE).strip()


def _normalize_language_label(text: str) -> str:
    """Alphabetic normalization for comparing the classifier output to 'English'."""
    return re.sub(r"[^a-zA-Z\s-]", "", text).strip().lower()


def _first_line(text: str) -> str:
    return text.strip().split("\n", 1)[0].strip()


def _message_content(response) -> str:
    if response is None:
        return ""
    msg = getattr(response, "message", None)
    if msg is not None and getattr(msg, "content", None) is not None:
        return str(msg.content)
    if isinstance(response, dict):
        m = response.get("message")
        if isinstance(m, dict) and m.get("content") is not None:
            return str(m["content"])
        if m is not None and getattr(m, "content", None) is not None:
            return str(m.content)
    return ""


def translate_content(content: str) -> tuple[bool, str]:
    """
    Classify language, then translate non-English text to English.

    Returns:
        (is_english, text):
        - If the post is English (or we keep the original for safety), is_english is True
          and text is the original content.
        - If translated, is_english is False and text is the translation (possibly empty
          if the model returned nothing usable).
    """
    if not content:
        return True, ""

    try:
        cls_response = client.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": classification_context},
                {"role": "user", "content": content},
            ],
        )
    except Exception:
        return True, content

    raw_lang = _strip_reasoning(_message_content(cls_response))
    lang_line = _first_line(raw_lang)
    normalized = _normalize_language_label(lang_line)
    word_count = len(lang_line.split()) if lang_line else 0

    if not normalized or word_count > _MAX_LANGUAGE_LABEL_WORDS:
        return True, content

    if normalized == "english":
        return True, content

    try:
        trans_response = client.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": translation_context},
                {"role": "user", "content": content},
            ],
        )
    except Exception:
        return False, ""

    translated = _strip_reasoning(_message_content(trans_response)).strip()
    if not translated:
        return False, ""

    return False, translated

import json
import os
import re
import time

import ollama

# Ollama client host: full URL or host:port (OLLAMA_HOST is the env var Ollama itself uses).
_raw_host = os.getenv("OLLAMA_HOST", "127.0.0.1:11434").strip()
OLLAMA_URL = _raw_host if _raw_host.startswith("http") else f"http://{_raw_host}"

# Set via OLLAMA_MODEL (e.g. docker-compose); default must match a pulled Ollama tag.
MODEL_NAME = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

client = ollama.Client(host=OLLAMA_URL)

# One LLM round-trip instead of classify + translate (roughly halves post latency).
system_prompt = """You are a highly accurate forum translation assistant.
Output ONLY a JSON object with two keys: "is_english" (boolean) and "translated_content" (string).

Rules:
- If the text is English, "is_english" must be true, and "translated_content" is the exact original text.
- If the text is NOT English, "is_english" must be false, and "translated_content" is an accurate English translation.
- Preserve any HTML tags exactly as they appear."""


def _strip_reasoning(text: str) -> str:
    """Remove chain-of-thought blocks common in reasoning models (e.g. deepseek-r1)."""
    open_t = "<" + "think" + ">"
    close_t = "<" + "/" + "think" + ">"
    pattern = re.escape(open_t) + r".*?" + re.escape(close_t)
    return re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE).strip()


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
            return str(m.content)  # type: ignore
    return ""


def _extract_json_object(text: str) -> dict | None:
    text = text.strip()
    m = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if m:
        text = m.group(1).strip()
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        obj = json.loads(text[start:end])
        return obj if isinstance(obj, dict) else None
    except (ValueError, json.JSONDecodeError):
        return None


def _snippet(s: str, max_len: int = 240) -> str:
    s = s.replace("\n", "\\n")
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def translate_content(content: str) -> tuple[bool, str, dict]:
    """
    Single Ollama call: JSON with is_english + translated_content.

    Returns:
        (is_english, text, meta):
        meta includes ollama_model, ollama_ms, ollama_parse_ok, branch, optional model_reply_snippet.
    """
    meta: dict = {
        "ollama_model": MODEL_NAME,
        "ollama_ms": None,
        "ollama_parse_ok": False,
        "branch": "empty_input",
    }

    if not content:
        return True, "", meta

    meta["branch"] = "error"
    meta["input_chars"] = len(content)

    t0 = time.perf_counter()
    try:
        response = client.chat(
            model=MODEL_NAME,
            format="json",
            options={
                "temperature": 0.0,  # Make the model deterministic
                "top_p": 0.1,
            },
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "<p>Hello</p>"},
                {
                    "role": "assistant",
                    "content": '{"is_english": true, "translated_content": "<p>Hello</p>"}',
                },
                {"role": "user", "content": "Bonjour tout le monde"},
                {
                    "role": "assistant",
                    "content": '{"is_english": false, "translated_content": "Hello everyone"}',
                },
                {"role": "user", "content": content},  # The actual user input
            ],
        )
    except Exception as ex:
        meta["ollama_ms"] = int((time.perf_counter() - t0) * 1000)
        meta["branch"] = "ollama_exception"
        meta["error"] = str(ex)
        return True, content, meta

    meta["ollama_ms"] = int((time.perf_counter() - t0) * 1000)

    raw = _strip_reasoning(_message_content(response))
    meta["model_reply_chars"] = len(raw)
    data = _extract_json_object(raw)
    if not isinstance(data, dict):
        meta["branch"] = "json_parse_failed"
        meta["model_reply_snippet"] = _snippet(raw)
        return True, content, meta

    meta["ollama_parse_ok"] = True

    is_eng = data.get("is_english")
    if isinstance(is_eng, str):
        is_eng = is_eng.strip().lower() in ("true", "1", "yes")
    elif not isinstance(is_eng, bool):
        is_eng = True

    translated = data.get("translated_content")
    translated = str(translated).strip() if translated is not None else ""

    meta["api_is_english"] = is_eng
    meta["translated_chars"] = len(translated)

    if not translated:
        if is_eng:
            meta["branch"] = "empty_translation_treated_english"
            return True, content, meta
        meta["branch"] = "non_english_empty_translation"
        return False, "", meta

    if is_eng:
        meta["branch"] = "model_says_english"
        return True, content, meta

    meta["branch"] = "model_says_non_english"
    return False, translated, meta

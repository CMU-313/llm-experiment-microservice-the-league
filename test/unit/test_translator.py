"""Unit tests for translate_content with a mocked Ollama client."""

import json
from unittest.mock import MagicMock, patch

from src import translator


def _chat_response(content: str):
    r = MagicMock()
    r.message = MagicMock(content=content)
    return r


def _json_chat(is_english: bool, translated: str) -> str:
    return json.dumps({"is_english": is_english, "translated_content": translated})


@patch.object(translator.client, "chat")
def test_english_skips_translation(mock_chat):
    mock_chat.return_value = _chat_response(_json_chat(True, "Hello there"))

    is_en, text, meta = translator.translate_content("Hello there")

    assert is_en is True
    assert text == "Hello there"
    assert mock_chat.call_count == 1
    assert meta.get("branch") == "model_says_english"


@patch.object(translator.client, "chat")
def test_french_translated(mock_chat):
    mock_chat.return_value = _chat_response(
        _json_chat(False, "Hello, my name is Bob"),
    )

    is_en, text, meta = translator.translate_content("Bonjour, je m'appelle Bob")

    assert is_en is False
    assert text == "Hello, my name is Bob"
    assert mock_chat.call_count == 1
    assert meta.get("branch") == "model_says_non_english"


@patch.object(translator.client, "chat")
def test_invalid_json_keeps_original(mock_chat):
    mock_chat.return_value = _chat_response(
        "I don't understand your request at all really"
    )

    post = "Hier ist dein erstes Beispiel."
    is_en, text, meta = translator.translate_content(post)

    assert is_en is True
    assert text == post
    assert meta.get("branch") == "json_parse_failed"
    mock_chat.assert_called_once()


@patch.object(translator.client, "chat")
def test_empty_model_output_keeps_original(mock_chat):
    mock_chat.return_value = _chat_response("")

    post = "Bonjour tout le monde"
    is_en, text, meta = translator.translate_content(post)

    assert is_en is True
    assert text == post
    assert meta.get("branch") == "json_parse_failed"


@patch.object(translator.client, "chat")
def test_classification_error_keeps_original(mock_chat):
    mock_chat.side_effect = RuntimeError("connection refused")

    post = "Something"
    is_en, text, meta = translator.translate_content(post)

    assert is_en is True
    assert text == post
    assert meta.get("branch") == "ollama_exception"


@patch.object(translator.client, "chat")
def test_non_english_empty_translation(mock_chat):
    mock_chat.return_value = _chat_response(_json_chat(False, ""))

    is_en, text, meta = translator.translate_content("Hola, cómo estás?")

    assert is_en is False
    assert text == ""
    assert meta.get("branch") == "non_english_empty_translation"


@patch.object(translator.client, "chat")
def test_non_english_translation_error(mock_chat):
    mock_chat.side_effect = RuntimeError("timeout")

    is_en, text, meta = translator.translate_content("Привет")

    assert is_en is True
    assert text == "Привет"
    assert meta.get("branch") == "ollama_exception"


@patch.object(translator.client, "chat")
def test_translation_returns_unusual_string(mock_chat):
    mock_chat.return_value = _chat_response(_json_chat(False, "@@@ ###"))

    is_en, text, meta = translator.translate_content("Привет")

    assert is_en is False
    assert text == "@@@ ###"
    assert meta.get("branch") == "model_says_non_english"


@patch.object(translator.client, "chat")
def test_strip_think_tags_from_response(mock_chat):
    ot, ct = "<" + "think" + ">", "<" + "/" + "think" + ">"
    inner = _json_chat(False, "Hello")
    wrapped = f"{ot}\nstep\n{ct}\n{inner}"
    mock_chat.return_value = _chat_response(wrapped)

    is_en, text, meta = translator.translate_content("Guten Tag")

    assert is_en is False
    assert text == "Hello"
    assert meta.get("branch") == "model_says_non_english"


@patch.object(translator.client, "chat")
def test_json_in_markdown_fence(mock_chat):
    inner = _json_chat(True, "Hi")
    body = f"```json\n{inner}\n```"
    mock_chat.return_value = _chat_response(body)

    is_en, text, meta = translator.translate_content("Hi")

    assert is_en is True
    assert text == "Hi"
    assert meta.get("branch") == "model_says_english"


def test_empty_input():
    with patch.object(translator.client, "chat") as mock_chat:
        is_en, text, meta = translator.translate_content("")
        assert is_en is True
        assert text == ""
        assert meta.get("branch") == "empty_input"
        mock_chat.assert_not_called()

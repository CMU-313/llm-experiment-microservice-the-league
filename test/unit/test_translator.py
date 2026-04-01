"""Unit tests for translate_content with a mocked Ollama client."""

from unittest.mock import MagicMock, patch

from src import translator


def _chat_response(content: str):
    r = MagicMock()
    r.message = MagicMock(content=content)
    return r


@patch.object(translator.client, "chat")
def test_english_skips_translation(mock_chat):
    mock_chat.return_value = _chat_response("English")

    is_en, text = translator.translate_content("Hello there")

    assert is_en is True
    assert text == "Hello there"
    assert mock_chat.call_count == 1


@patch.object(translator.client, "chat")
def test_french_translated(mock_chat):
    mock_chat.side_effect = [
        _chat_response("French"),
        _chat_response("Hello, my name is Bob"),
    ]

    is_en, text = translator.translate_content("Bonjour, je m'appelle Bob")

    assert is_en is False
    assert text == "Hello, my name is Bob"
    assert mock_chat.call_count == 2


@patch.object(translator.client, "chat")
def test_verbose_classification_keeps_original(mock_chat):
    mock_chat.return_value = _chat_response(
        "I don't understand your request at all really"
    )

    post = "Hier ist dein erstes Beispiel."
    is_en, text = translator.translate_content(post)

    assert is_en is True
    assert text == post
    mock_chat.assert_called_once()


@patch.object(translator.client, "chat")
def test_empty_classification_keeps_original(mock_chat):
    mock_chat.return_value = _chat_response("")

    post = "Bonjour tout le monde"
    is_en, text = translator.translate_content(post)

    assert is_en is True
    assert text == post


@patch.object(translator.client, "chat")
def test_classification_error_keeps_original(mock_chat):
    mock_chat.side_effect = RuntimeError("connection refused")

    post = "Something"
    is_en, text = translator.translate_content(post)

    assert is_en is True
    assert text == post


@patch.object(translator.client, "chat")
def test_non_english_empty_translation(mock_chat):
    mock_chat.side_effect = [
        _chat_response("Spanish"),
        _chat_response(""),
    ]

    is_en, text = translator.translate_content("Hola, cómo estás?")

    assert is_en is False
    assert text == ""


@patch.object(translator.client, "chat")
def test_non_english_translation_error(mock_chat):
    mock_chat.side_effect = [
        _chat_response("Russian"),
        RuntimeError("timeout"),
    ]

    is_en, text = translator.translate_content("Привет")

    assert is_en is False
    assert text == ""


@patch.object(translator.client, "chat")
def test_translation_returns_unusual_string(mock_chat):
    mock_chat.side_effect = [
        _chat_response("Russian"),
        _chat_response("@@@ ###"),
    ]

    is_en, text = translator.translate_content("Привет")

    assert is_en is False
    assert text == "@@@ ###"


@patch.object(translator.client, "chat")
def test_strip_think_tags_from_translation(mock_chat):
    ot, ct = "<" + "think" + ">", "<" + "/" + "think" + ">"
    wrapped = f"{ot}\nstep\n{ct}\nHello"
    mock_chat.side_effect = [
        _chat_response("German"),
        _chat_response(wrapped),
    ]

    is_en, text = translator.translate_content("Guten Tag")

    assert is_en is False
    assert text == "Hello"


def test_empty_input():
    with patch.object(translator.client, "chat") as mock_chat:
        is_en, text = translator.translate_content("")
        assert is_en is True
        assert text == ""
        mock_chat.assert_not_called()

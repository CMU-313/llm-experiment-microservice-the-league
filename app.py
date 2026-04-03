import os
import time
from flask import Flask
from flask import request, jsonify
from src.translator import translate_content

app = Flask(__name__)


def _json_translate(content: str, route: str):
    t0 = time.perf_counter()
    is_english, translated_content, meta = translate_content(content)
    total_ms = int((time.perf_counter() - t0) * 1000)
    meta = dict(meta)
    meta["translator_total_ms"] = total_ms
    meta["route"] = route

    return jsonify(
        {
            "is_english": is_english,
            "translated_content": translated_content,
            "translator_meta": meta,
        }
    )


@app.route("/translate", methods=["POST"])
def translator_post():
    """Prefer this from NodeBB: full post HTML in the body avoids URL length limits on GET /."""
    body = request.get_json(silent=True) or {}
    raw = body.get("content", "")
    if not isinstance(raw, str):
        raw = str(raw) if raw is not None else ""
    return _json_translate(raw, "POST /translate")


@app.route("/")
def translator():
    content = request.args.get("content", default="", type=str)
    return _json_translate(content, "GET /")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

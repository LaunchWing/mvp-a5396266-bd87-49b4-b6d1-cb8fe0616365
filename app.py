"""LaunchWing Agent API — Clean and aligned with Cloudflare MVP handler"""

from __future__ import annotations

import logging
import openai
import os
from flask import Flask, jsonify, request

from launchwing_agent import build_app_files

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# OpenAI setup
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY environment variable must be set.")
client = openai.OpenAI(api_key=api_key)

app = Flask(__name__)

@app.get("/")
def index() -> tuple[str, int]:
    logging.info("Health check requested")
    return "LaunchWing Agent is running", 200

@app.post("/generate")
def generate_files():
    data = request.get_json(silent=True) or {}
    prompt = data.get("prompt")
    if not prompt:
        logging.warning("POST /generate missing 'prompt'")
        return jsonify({"error": "Missing 'prompt' in JSON body"}), 400

    logging.info("Generating app files for prompt: %r", prompt)

    try:
        files = build_app_files(prompt)
        result = [{"path": f.path, "content": f.content} for f in files]
        logging.info("✅ Generated %d files", len(result))
        return jsonify({"files": result}), 200
    except Exception as exc:
        logging.exception("Error during file generation")
        return jsonify({"error": str(exc)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
import os
from flask import Flask, request, jsonify
from gradio_client import Client
from werkzeug.utils import secure_filename

# Use your existing util so pdf/docx also work
from utils.file_utils import extract_text_from_file
# Optional: if you want auto language later, you already have this util:
# from utils.language_utils import detect_language

app = Flask(__name__)

# --- Config that matches Unity usage ---
ALLOWED_EXT = {".txt", ".pdf", ".docx"}
MAX_CONTENT_LENGTH = 8 * 1024 * 1024  # 8 MB cap
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# Connect to your Hugging Face Space (unchanged)
SPACE_URL = "https://rathod31-kannada-english-sim.hf.space"
client = Client(SPACE_URL)


def _allowed(filename: str) -> bool:
    filename = (filename or "").lower()
    return any(filename.endswith(ext) for ext in ALLOWED_EXT)


@app.route("/", methods=["GET"])
def index():
    return "Hello from Unity similarity proxy", 200


@app.route("/healthz", methods=["GET"])
def healthz():
    return jsonify({"status": "ok"}), 200


@app.route("/compare", methods=["POST"])
def compare_texts():
    try:
        # 1) Read Unity form-data
        audio_text = (request.form.get("audio_text") or "").strip()
        uploaded_file = request.files.get("file")

        if not audio_text or not uploaded_file:
            return jsonify({"error": "Both 'audio_text' and 'file' are required."}), 400

        # 2) Validate and store temp file
        filename = secure_filename(uploaded_file.filename or "")
        if not filename or not _allowed(filename):
            return jsonify({
                "error": f"Unsupported or empty file. Allowed: {', '.join(sorted(ALLOWED_EXT))}"
            }), 400

        os.makedirs("uploads", exist_ok=True)
        temp_path = os.path.join("uploads", filename)
        uploaded_file.save(temp_path)

        try:
            # 3) Extract document text (supports txt, docx, pdf)
            file_text = extract_text_from_file(temp_path)
        finally:
            # Cleanup temp file regardless of success/failure
            try:
                os.remove(temp_path)
            except Exception:
                pass

        if not file_text or not file_text.strip():
            return jsonify({"error": "Could not extract text from file."}), 400

        # 4) Call your Hugging Face Space (UNCHANGED core behavior)
        #    If you want auto-language later, you can detect and pass it.
        # lang = detect_language(audio_text)  # "Kannada" or "English"
        # For now keep your original forced lang:
        result_json, similarity = client.predict(
            lang="Kannada",
            a=audio_text,
            b=file_text,
            api_name="/_on_click"
        )

        # 5) Unity expects ONLY: {"similarity_score": <0..100>}
        #    Your Space returns similarity in 0..1 (based on your earlier curl).
        #    Convert to percent, clamp, round.
        try:
            # Prefer explicit field if present
            raw_sim = result_json.get("similarity", similarity)
        except Exception:
            raw_sim = similarity

        # Fallback: if the space returns cosine in [-1,1], convert that
        if raw_sim is None and isinstance(result_json, dict):
            scores = result_json.get("scores", {})
            cos = scores.get("cosine")
            if cos is not None:
                # map [-1,1] -> [0,100]
                raw_sim = (float(cos) + 1.0) / 2.0

        # Final safety defaults
        try:
            raw_sim = float(raw_sim)
        except Exception:
            return jsonify({"error": "Invalid similarity from Space."}), 502

        # If already 0..1 -> percent; if someone changes to -1..1, clamp then map
        if raw_sim < 0 or raw_sim > 1:
            # assume -1..1 scale
            raw_sim = max(min(raw_sim, 1.0), -1.0)
            percent = ((raw_sim + 1.0) / 2.0) * 100.0
        else:
            percent = raw_sim * 100.0

        return jsonify({"similarity_score": round(percent, 2)}), 200

    except Exception as e:
        # Always return JSON so Unity can parse cleanly
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render provides PORT
    app.run(host="0.0.0.0", port=port)

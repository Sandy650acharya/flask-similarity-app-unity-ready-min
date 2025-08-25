import os
import requests
import docx
from flask import Flask, request, jsonify

app = Flask(__name__)

# Hugging Face Space API URL (replace if different)
SPACE_API_URL = "https://rathod31-kannada-english-sim.hf.space/api/predict"

def extract_text_from_file(file_path):
    """Extract plain text from .txt or .docx file."""
    if file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    elif file_path.endswith(".docx"):
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs]).strip()
    return ""

@app.route("/", methods=["GET"])
def home():
    return "Server is running"

@app.route("/compare", methods=["POST"])
def compare_texts():
    try:
        # 1. Extract form data
        audio_text = request.form.get("audio_text", "").strip()
        uploaded_file = request.files.get("file")

        if not audio_text or not uploaded_file:
            return jsonify({"error": "Both 'audio_text' and 'file' are required."}), 400

        # 2. Save file temporarily
        os.makedirs("uploads", exist_ok=True)
        file_path = os.path.join("uploads", uploaded_file.filename)
        uploaded_file.save(file_path)

        # 3. Extract text from file
        file_text = extract_text_from_file(file_path)
        if not file_text:
            return jsonify({"error": "Could not extract text from the file."}), 400

        # 4. Prepare payload for Hugging Face Space
        payload = {
            "data": [
                "Kannada",  # or "Auto" if supported by HF Space
                audio_text,
                file_text
            ]
        }

        # 5. Call Hugging Face Space API
        response = requests.post(SPACE_API_URL, json=payload)
        if response.status_code != 200:
            return jsonify({"error": "Hugging Face API request failed.", "details": response.text}), 500

        result = response.json()

        # Expected HF response: {"data": [{"scores": {"cosine": 0.5}, "similarity": 0.5}]}
        try:
            data = result["data"][0]
            cosine_score = data.get("scores", {}).get("cosine", 0)
            similarity_score = data.get("similarity", 0)
        except Exception:
            return jsonify({"error": "Invalid response from Hugging Face API.", "details": result}), 500

        # 6. Return similarity scores
        return jsonify({
            "cosine": round(float(cosine_score), 4),
            "similarity": round(float(similarity_score), 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render assigns PORT env
    app.run(host="0.0.0.0", port=port)

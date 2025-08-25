import os
import io
import logging
from typing import Optional

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# ML
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, logging as hf_logging

# Text extraction libs
import docx
import PyPDF2

# --- Configuration ---
ALLOWED_EXT = {".txt", ".pdf", ".docx"}
MAX_CONTENT_LENGTH = 8 * 1024 * 1024  # 8 MB
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEVICE = torch.device("cpu")

# Reduce HF warnings
hf_logging.set_verbosity_error()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vr-text-sim")

HF_TOKEN = os.environ.get("HUGGINGFACE_HUB_TOKEN")
hf_token_kw = {"use_auth_token": HF_TOKEN} if HF_TOKEN else {}

# ---- Load model on startup ----
logger.info("Loading model: %s", MODEL_NAME)
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, **hf_token_kw)
    model = AutoModel.from_pretrained(MODEL_NAME, **hf_token_kw)
    model.to(DEVICE)
    model.eval()
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.exception("Failed to load model: %s", e)
    tokenizer = None
    model = None


def allowed_filename(filename: str) -> bool:
    filename = (filename or "").lower()
    return any(filename.endswith(ext) for ext in ALLOWED_EXT)


def extract_text_from_file(path: str) -> str:
    """Extract text from txt, docx, or pdf file at 'path'."""
    _, ext = os.path.splitext(path.lower())
    if ext == ".txt":
        # Try UTF-8, fall back to latin-1
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            with open(path, "r", encoding="latin-1", errors="ignore") as f:
                return f.read()
    elif ext == ".docx":
        doc = docx.Document(path)
        return "\n".join([p.text for p in doc.paragraphs])
    elif ext == ".pdf":
        text = ""
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                # extract_text may return None; guard it
                page_text = page.extract_text() or ""
                text += page_text + "\n"
        return text
    else:
        raise ValueError("Unsupported file type")


def mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


def compute_similarity(text_a: str, text_b: str) -> float:
    if tokenizer is None or model is None:
        raise RuntimeError("Model not loaded")

    texts = [text_a, text_b]
    with torch.no_grad():
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded = {k: v.to(DEVICE) for k, v in encoded.items()}
        out = model(**encoded)
        embeddings = mean_pooling(out.last_hidden_state, encoded["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        sim = torch.matmul(embeddings[0:1], embeddings[1:2].T).item()
        return float(max(min(sim, 1.0), -1.0))


@app.route("/", methods=["GET"])
def index():
    return "Hello, VR Text Similarity App!", 200


@app.route("/healthz", methods=["GET"])
def healthz():
    return jsonify({"status": "ok"}), 200


@app.route("/compare", methods=["POST"])
def compare():
    try:
        if tokenizer is None or model is None:
            logger.error("Model not loaded")
            return jsonify({"error": "Model not loaded on server."}), 500

        # Validate inputs
        audio_text = request.form.get("audio_text")
        if not audio_text:
            return jsonify({"error": "Missing form field 'audio_text'."}), 400

        if "file" not in request.files:
            return jsonify({"error": "Missing 'file' in form-data."}), 400

        file = request.files["file"]
        if not file or file.filename == "":
            return jsonify({"error": "Empty filename in uploaded file."}), 400

        filename = secure_filename(file.filename)
        if not allowed_filename(filename):
            return jsonify({"error": f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXT))}"}), 400

        # Save temp file
        os.makedirs("uploads", exist_ok=True)
        temp_path = os.path.join("uploads", filename)
        file.save(temp_path)

        # Extract text
        try:
            doc_text = extract_text_from_file(temp_path)
        finally:
            # Always try to clean up temp file
            try:
                os.remove(temp_path)
            except Exception:
                pass

        if not doc_text or len(doc_text.strip()) == 0:
            return jsonify({"error": "Document contains no extractable text."}), 400

        # Compute similarity (cosine in [-1,1]) and convert to percent [0,100]
        sim = compute_similarity(audio_text, doc_text)
        sim_percent = ((sim + 1.0) / 2.0) * 100.0

        return jsonify({"similarity_score": round(sim_percent, 2)}), 200

    except Exception as e:
        logger.exception("Unhandled error in /compare: %s", e)
        return jsonify({"error": "Unhandled server error."}), 500


if __name__ == "__main__":
    # When running locally
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

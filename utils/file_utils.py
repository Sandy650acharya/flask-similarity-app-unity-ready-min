import os
import docx
import PyPDF2

def extract_text_from_file(file_path: str) -> str:
    """
    Extract text from txt, docx, or pdf files.
    """
    _, ext = os.path.splitext(file_path)

    if ext.lower() == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    elif ext.lower() == ".docx":
        doc = docx.Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])

    elif ext.lower() == ".pdf":
        text = ""
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text

    else:
        raise ValueError("Unsupported file type. Use txt, docx, or pdf.")

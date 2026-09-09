import fitz  # PyMuPDF
from typing import List
import os

def load_pdfs(pdf_files) -> List[str]:
    texts = []
    for file_obj in pdf_files:
        pdf = fitz.open(stream=file_obj.read(), filetype="pdf")
        content = ""
        for page in pdf:
            content += page.get_text("text") + "\n"
        texts.append(content)
    return texts

def chunk_text(text: str, chunk_size=500, overlap=50) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

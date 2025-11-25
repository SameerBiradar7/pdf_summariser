# app/extract.py
from pathlib import Path
import pdfplumber
from pptx import Presentation
import docx
import os

# Optional: image OCR dependencies (pytesseract + PIL)
try:
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False


def extract_text_from_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def extract_text_from_pdf(path: Path) -> str:
    text_pages = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            txt = page.extract_text()
            if txt:
                text_pages.append(f"[page:{i}]\n" + txt)
    return "\n\n".join(text_pages)


def extract_text_from_pptx(path: Path) -> str:
    prs = Presentation(str(path))
    slides = []
    for i, slide in enumerate(prs.slides, start=1):
        texts = []
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                t = shape.text.strip()
                if t:
                    texts.append(t)
        if texts:
            slides.append(f"[slide:{i}]\n" + "\n".join(texts))
    return "\n\n".join(slides)


def extract_text_from_docx(path: Path) -> str:
    doc = docx.Document(str(path))
    paragraphs = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
    return "\n\n".join(paragraphs)


def extract_text_from_image(path: Path) -> str:
    if not OCR_AVAILABLE:
        raise RuntimeError("OCR not available. Install pytesseract and pillow, and ensure tesseract is installed on system.")
    img = Image.open(path)
    return pytesseract.image_to_string(img)


def extract_text(filepath: str) -> str:
    """
    Generic extractor: chooses method by extension.
    Returns extracted text string (may include page/slide markers).
    """
    p = Path(filepath)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    ext = p.suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(p)
    if ext in {".pptx", ".ppt"}:
        return extract_text_from_pptx(p)
    if ext in {".docx", ".doc"}:
        return extract_text_from_docx(p)
    if ext == ".txt" or ext == ".md":
        return extract_text_from_txt(p)
    if ext in {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}:
        return extract_text_from_image(p)
    raise ValueError(f"Unsupported file extension: {ext}")
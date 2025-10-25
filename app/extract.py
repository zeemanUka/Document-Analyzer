# app/extract.py
import io
from typing import List, Optional

import pdfplumber
from pypdf import PdfReader
from pdf2image import convert_from_bytes
import pytesseract

def _unlock_pdf(raw_bytes: bytes, password: Optional[str]) -> bytes:
    """
    If encrypted, verify the password. We return the same bytes since
    pdfplumber can accept the password directly.
    """
    bio = io.BytesIO(raw_bytes)
    reader = PdfReader(bio)
    if reader.is_encrypted:
        if not password:
            raise ValueError("PDF is password-protected; no password provided.")
        res = reader.decrypt(password)
        if res in (0, False, None):
            raise ValueError("Incorrect PDF password.")
    return raw_bytes

def extract_text_by_page(raw_bytes: bytes, password: Optional[str]) -> List[str]:
    pages_text: List[str] = []
    try:
        with pdfplumber.open(io.BytesIO(raw_bytes), password=password) as pdf:
            for p in pdf.pages:
                text = (p.extract_text() or "").strip()
                pages_text.append(text)
    except Exception:
        # Fallback: pypdf (robust but less table-aware)
        bio = io.BytesIO(raw_bytes)
        reader = PdfReader(bio)
        if reader.is_encrypted:
            if not password:
                raise ValueError("PDF is encrypted and no password provided.")
            res = reader.decrypt(password)
            if res in (0, False, None):
                raise ValueError("Could not open encrypted PDF with provided password.")
        for page in reader.pages:
            pages_text.append((page.extract_text() or "").strip())
    return pages_text

def needs_ocr(pages_text: List[str]) -> bool:
    if not pages_text:
        return False
    empty_ratio = sum(1 for t in pages_text if not t) / float(len(pages_text))
    return empty_ratio > 0.6

def extract_with_ocr(raw_bytes: bytes) -> List[str]:
    images = convert_from_bytes(raw_bytes, dpi=300)
    texts: List[str] = []
    for img in images:
        txt = pytesseract.image_to_string(img)
        texts.append((txt or "").strip())
    return texts

def get_pages_text(raw_bytes: bytes, password: Optional[str]) -> List[str]:
    raw_bytes = _unlock_pdf(raw_bytes, password)
    pages_text = extract_text_by_page(raw_bytes, password)

    # If most pages came back empty (likely scanned), OCR them
    if needs_ocr(pages_text):
        ocr_pages = extract_with_ocr(raw_bytes)
        L = max(len(pages_text), len(ocr_pages))
        merged: List[str] = []
        for i in range(L):
            t = pages_text[i] if i < len(pages_text) else ""
            o = ocr_pages[i] if i < len(ocr_pages) else ""
            merged.append(o if not t and o else t or o)
        pages_text = merged

    return pages_text

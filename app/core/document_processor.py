"""PDF parsing, structure-aware chunking, and legal clause classification."""

import hashlib
import logging
import os
import re
from typing import Any, Dict, List

import fitz  # PyMuPDF
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class DocumentChunk(BaseModel):
    """A single indexed text chunk from a legal contract."""
    chunk_id: str
    text: str
    source_file: str
    page_number: int
    section_heading: str = "General"
    clause_type: str = "general"
    chunk_index: int = 0
    char_count: int = 0

    @property
    def metadata(self) -> Dict[str, Any]:
        """Flat dict for ChromaDB metadata storage."""
        return {
            "source_file": self.source_file,
            "page_number": self.page_number,
            "section_heading": self.section_heading,
            "clause_type": self.clause_type,
            "chunk_index": self.chunk_index,
            "char_count": self.char_count,
        }


class DocumentProcessor:
    """Parses legal PDFs into structured, clause-tagged chunks ready for indexing."""

    # Matches numbered clause headings, all-caps headings, and known legal section names
    _SECTION_RE = re.compile(
        r"^(?:(?:ARTICLE|Article|SECTION|Section|Clause|CLAUSE)\s+[\dIVXLCDM]+[.:]?"
        r"|\d+(?:\.\d+)*\s+[A-Z]|[A-Z\s]{4,}:"
        r"|\b(?:RECITALS|DEFINITIONS|INDEMNIFICATION|TERMINATION|CONFIDENTIALITY"
        r"|GOVERNING LAW|WARRANTIES|LIABILITY)\b)",
        re.MULTILINE,
    )

    # Keyword patterns for legal clause classification
    _CLAUSE_PATTERNS: Dict[str, re.Pattern] = {
        "indemnification":         re.compile(r"\b(indemnif|indemnity|hold harmless)\b", re.I),
        "termination":             re.compile(r"\b(terminat|cancellation|expiration|cure period)\b", re.I),
        "governing_law":           re.compile(r"\b(governing law|jurisdiction|venue|arbitration)\b", re.I),
        "limitation_of_liability": re.compile(r"\b(limitation of liability|liability cap|aggregate liability)\b", re.I),
        "confidentiality":         re.compile(r"\b(confidential|non-disclosure|trade secret)\b", re.I),
        "intellectual_property":   re.compile(r"\b(intellectual property|patent|trademark|copyright)\b", re.I),
        "force_majeure":           re.compile(r"\b(force majeure|act of god)\b", re.I),
        "payment_terms":           re.compile(r"\b(compensation|payment|invoic|fees|billing)\b", re.I),
        "non_compete":             re.compile(r"\b(non-compete|non-solicitation|restrictive covenant)\b", re.I),
        "warranties":              re.compile(r"\b(representations and warranties|disclaimer of warranties)\b", re.I),
    }

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 75) -> None:
        self.max_chars = chunk_size * 4       # ~4 chars per token
        self.step_words = max(1, chunk_size - chunk_overlap)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def process_pdf(self, pdf_path: str) -> List[DocumentChunk]:
        """Parse one PDF into structured chunks."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        source_file = os.path.basename(pdf_path)
        chunks: List[DocumentChunk] = []
        idx = 0

        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            raise RuntimeError(f"Cannot open {source_file}: {e}") from e

        try:
            heading = "Preamble"
            page_count = len(doc)
            for page_idx in range(page_count):
                page_text = self._clean(doc[page_idx].get_text("text"))
                if not page_text:
                    continue

                for para in page_text.split("\n\n"):
                    para = para.strip()
                    if not para:
                        continue

                    first_line = para.split("\n")[0].strip()
                    if self._SECTION_RE.search(first_line) or (len(first_line) < 80 and first_line.isupper()):
                        heading = first_line[:100]

                    if len(para) <= self.max_chars:
                        chunks.append(self._make_chunk(para, source_file, page_idx + 1, heading, idx))
                        idx += 1
                    else:
                        sub = self._sliding_window(para, source_file, page_idx + 1, heading, idx)
                        chunks.extend(sub)
                        idx += len(sub)
        finally:
            doc.close()

        logger.info(f"'{source_file}': {len(chunks)} chunks from {page_count} pages")
        return chunks

    def process_directory(self, directory_path: str) -> List[DocumentChunk]:
        """Process all PDFs in a directory."""
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        pdfs = sorted(f for f in os.listdir(directory_path) if f.lower().endswith(".pdf"))
        if not pdfs:
            logger.warning(f"No PDFs found in: {directory_path}")
            return []

        all_chunks: List[DocumentChunk] = []
        for name in pdfs:
            logger.info(f"Processing: {name}")
            all_chunks.extend(self.process_pdf(os.path.join(directory_path, name)))

        logger.info(f"Total: {len(all_chunks)} chunks from {len(pdfs)} documents")
        return all_chunks

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _clean(self, text: str) -> str:
        """Normalize whitespace in raw PDF text."""
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _classify(self, text: str, heading: str = "") -> str:
        """Return the first matching clause type, or 'general'."""
        combined = f"{heading} {text}"
        for clause_type, pattern in self._CLAUSE_PATTERNS.items():
            if pattern.search(combined):
                return clause_type
        return "general"

    def _chunk_id(self, source_file: str, page: int, idx: int, text: str) -> str:
        """Deterministic 24-char hash for deduplication."""
        key = f"{source_file}::p{page}::i{idx}::{text[:100]}"
        return hashlib.sha256(key.encode()).hexdigest()[:24]

    def _make_chunk(self, text: str, source_file: str, page: int, heading: str, idx: int) -> DocumentChunk:
        return DocumentChunk(
            chunk_id=self._chunk_id(source_file, page, idx, text),
            text=text,
            source_file=source_file,
            page_number=page,
            section_heading=heading,
            clause_type=self._classify(text, heading),
            chunk_index=idx,
            char_count=len(text),
        )

    def _sliding_window(self, text: str, source_file: str, page: int, heading: str, start_idx: int) -> List[DocumentChunk]:
        """Split an oversized paragraph into overlapping word-window chunks."""
        words = text.split()
        chunks, i, idx = [], 0, start_idx
        window = max(50, self.step_words)
        while i < len(words):
            chunk_text = " ".join(words[i: i + window])
            if chunk_text:
                chunks.append(self._make_chunk(chunk_text, source_file, page, heading, idx))
                idx += 1
            i += self.step_words
        return chunks

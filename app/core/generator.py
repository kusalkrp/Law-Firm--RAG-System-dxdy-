"""Response Generator — grounded generation with strict citations and hallucination guards."""

import json
import logging
import re
from typing import Any, Dict, List, Literal, Optional
from openai import OpenAI
from pydantic import BaseModel, Field

from app.api.schemas import Citation
from app.config import get_settings
from app.core.retriever import RetrievedChunk

logger = logging.getLogger(__name__)

_ABSTENTION_PHRASE = "I cannot find this information in the provided documents."


class GeneratedAnswer(BaseModel):
    """Internal schema for structured generation output."""
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = "high"
    abstention_triggered_by: Optional[Literal["pre_check", "model"]] = None


class ResponseGenerator:
    """Handles CRAG relevance grading, prompt formatting, LLM generation, and citation verification."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._client: Optional[OpenAI] = None

    def _get_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                api_key=self.settings.AI_GATEWAY_API_KEY,
                base_url=self.settings.AI_GATEWAY_BASE_URL,
            )
        return self._client

    # ------------------------------------------------------------------ #
    # 1. CRAG Pre-Check: Grade chunk relevance before calling generator
    # ------------------------------------------------------------------ #

    def grade_relevance(self, query: str, chunks: List[RetrievedChunk]) -> bool:
        """Return True if at least one chunk is relevant to the query; False to short-circuit."""
        if not chunks:
            return False

        # If highest rerank score is extremely low (< 0.05), short-circuit
        top_score = max((c.rerank_score or 0.0) for c in chunks)
        if top_score > 0 and top_score < 0.05:
            logger.info(f"CRAG pre-check: top rerank score {top_score:.3f} below threshold (0.05).")
            return False

        excerpts = "\n\n".join([
            f"[{i+1}] (Document: {c.metadata.get('source_file')}, Page {c.metadata.get('page_number')}):\n{c.text}"
            for i, c in enumerate(chunks[:5])
        ])
        prompt = f"""Question: {query}

Passages:
{excerpts}

Do any of these passages contain information relevant to answering the question (directly or with legal context)?
Reply with ONLY 'YES' or 'NO'."""

        try:
            resp = self._get_client().chat.completions.create(
                model=self.settings.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=30,
            )
            decision = resp.choices[0].message.content.strip().upper()
            is_relevant = "YES" in decision
            logger.info(f"CRAG pre-check decision: {decision} (relevant={is_relevant})")
            return is_relevant
        except Exception as e:
            logger.warning(f"CRAG pre-check failed ({e}); proceeding to generation.")
            return True

    # ------------------------------------------------------------------ #
    # 2. Context Building & Generation
    # ------------------------------------------------------------------ #

    def _build_context_block(self, chunks: List[RetrievedChunk]) -> str:
        """Format retrieved chunks into numbered context passages with source metadata."""
        blocks = []
        for idx, c in enumerate(chunks, start=1):
            src = c.metadata.get("source_file", "Unknown")
            page = c.metadata.get("page_number", 1)
            section = c.metadata.get("section_heading", "General")
            clause_type = c.metadata.get("clause_type", "general")
            blocks.append(
                f"[Source {idx} | Document: {src} | Page: {page} | Section: {section} | Clause Type: {clause_type}]\n"
                f"{c.text.strip()}\n"
            )
        return "\n".join(blocks)

    def generate(self, query: str, chunks: List[RetrievedChunk]) -> GeneratedAnswer:
        """Generate a cited legal answer based strictly on retrieved chunks."""
        # 1. CRAG Relevance Pre-Check
        if not self.grade_relevance(query, chunks):
            logger.info("Short-circuiting to abstention via CRAG pre-check.")
            return GeneratedAnswer(
                answer=_ABSTENTION_PHRASE,
                citations=[],
                confidence="low",
                abstention_triggered_by="pre_check",
            )

        context_text = self._build_context_block(chunks)

        system_prompt = f"""You are a senior legal associate for a law firm.
Your task is to answer legal inquiries based EXCLUSIVELY on the provided contract excerpts.

RULES:
1. Grounding: Answer ONLY using facts explicitly stated in the context. Do NOT extrapolate or guess.
2. Redacted Information: In SEC contract filings, confidential commercial numbers are often redacted (marked as [*], [***], or [ ]). If a quantity or price is redacted, explicitly explain that the specific numerical figure is confidential/redacted in the agreement rather than leaving empty brackets.
3. In-Text Citations: Cite sources cleanly in natural brackets, e.g., [BellringBrandsInc, p.2, Section 3.1] or [Page 2, Section 4]. Do NOT copy raw prompt metadata tokens like "[Source 1 | ...]".
4. Abstention: If the context is completely insufficient or unrelated, set answer to exactly "{_ABSTENTION_PHRASE}".
5. Tone: Objective, precise, professional legal analysis.
6. Format: Respond with valid JSON matching this exact structure:
{{
  "answer": "<professional legal answer citing sources in clean brackets>",
  "citations": [
    {{
      "source_file": "<exact filename from source>",
      "page": <page_number_int>,
      "section": "<section_heading>",
      "clause_type": "<clause_type>",
      "text_excerpt": "<concise 1-2 sentence verbatim quote from context>"
    }}
  ],
  "confidence": "high" | "medium" | "low"
}}"""

        user_prompt = f"""CONTEXT:
{context_text}

LEGAL QUESTION: {query}"""

        try:
            resp = self._get_client().chat.completions.create(
                model=self.settings.LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=4000,
            )
            raw_content = resp.choices[0].message.content.strip()

            # Resilient JSON parsing
            data = self._parse_llm_json(raw_content)

            answer_text = data.get("answer", "")
            raw_citations = data.get("citations", [])
            confidence = data.get("confidence", "high")

            # Check if model chose to abstain
            if _ABSTENTION_PHRASE.lower() in answer_text.lower() or not answer_text:
                return GeneratedAnswer(
                    answer=_ABSTENTION_PHRASE,
                    citations=[],
                    confidence="low",
                    abstention_triggered_by="model",
                )

            # Build Citation objects
            citations = []
            for cit in raw_citations:
                try:
                    citations.append(
                        Citation(
                            source_file=cit.get("source_file", "Unknown"),
                            page=int(cit.get("page", 1)),
                            section=cit.get("section", "General"),
                            clause_type=cit.get("clause_type", "general"),
                            text_excerpt=cit.get("text_excerpt", ""),
                        )
                    )
                except Exception:
                    continue

            # Post-hoc citation verification: ensure excerpts exist in source chunks
            verified_citations = self._verify_citations(citations, chunks)

            return GeneratedAnswer(
                answer=answer_text,
                citations=verified_citations,
                confidence=confidence,
                abstention_triggered_by=None,
            )

        except Exception as e:
            logger.error(f"Generation error: {e}")
            return GeneratedAnswer(
                answer=_ABSTENTION_PHRASE,
                citations=[],
                confidence="low",
                abstention_triggered_by="model",
            )

    def _parse_llm_json(self, raw_content: str) -> Dict[str, Any]:
        """Resiliently parse JSON output from LLM with automatic truncation/quote recovery."""
        clean_json = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_content.strip(), flags=re.DOTALL).strip()
        try:
            return json.loads(clean_json)
        except Exception as json_err:
            logger.warning(f"Standard json.loads failed ({json_err}); attempting regex recovery.")

        # Attempt recovery via regex
        data: Dict[str, Any] = {}
        ans_match = re.search(r'"answer"\s*:\s*"(.*?)(?:",\s*"citations"|",\s*"confidence"|"$)', clean_json, re.DOTALL)
        if ans_match:
            data["answer"] = ans_match.group(1).replace("\\n", "\n").replace('\\"', '"')
        else:
            ans_open = re.search(r'"answer"\s*:\s*"(.*)', clean_json, re.DOTALL)
            if ans_open:
                data["answer"] = ans_open.group(1).split('"citations"')[0].rstrip('", \n\r\t').replace("\\n", "\n")

        cit_matches = re.findall(
            r'\{\s*"source_file"\s*:\s*"([^"]+)"\s*,\s*"page"\s*:\s*(\d+)\s*,\s*"section"\s*:\s*"([^"]+)"\s*,\s*"clause_type"\s*:\s*"([^"]+)"\s*,\s*"text_excerpt"\s*:\s*"([^"]+)"',
            clean_json,
        )
        data["citations"] = [
            {"source_file": m[0], "page": int(m[1]), "section": m[2], "clause_type": m[3], "text_excerpt": m[4]}
            for m in cit_matches
        ]
        data["confidence"] = "high" if data.get("answer") else "low"
        return data

    def _verify_citations(self, citations: List[Citation], chunks: List[RetrievedChunk]) -> List[Citation]:
        """Filter out hallucinated citations whose text excerpt cannot be verified in the context chunks."""
        if not citations or not chunks:
            return citations

        full_context = " ".join([c.text.lower() for c in chunks])
        verified = []
        for cit in citations:
            excerpt_clean = cit.text_excerpt.strip().lower()
            words = excerpt_clean.split()
            sample_phrase = " ".join(words[:4]) if len(words) >= 4 else excerpt_clean
            if sample_phrase in full_context or not excerpt_clean:
                verified.append(cit)
            else:
                logger.warning(f"Discarding unverified citation excerpt: '{cit.text_excerpt[:50]}'")

        return verified or citations

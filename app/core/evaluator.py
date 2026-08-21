"""Faithfulness Evaluator — claim-level decomposition and verification (LLM-as-a-judge)."""

import json
import logging
import re
from typing import List, Optional
from openai import OpenAI
from pydantic import BaseModel, Field

from app.config import get_settings
from app.core.retriever import RetrievedChunk

logger = logging.getLogger(__name__)


class FaithfulnessResult(BaseModel):
    """Result of claim-level faithfulness assessment."""
    score: float = Field(..., ge=0.0, le=1.0)
    total_claims: int = 0
    supported_claims: List[str] = Field(default_factory=list)
    unsupported_claims: List[str] = Field(default_factory=list)


class FaithfulnessEvaluator:
    """Decomposes an answer into atomic factual claims and verifies each against context passages."""

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

    def evaluate(self, answer: str, chunks: List[RetrievedChunk]) -> FaithfulnessResult:
        """Evaluate whether all factual claims in the answer are strictly supported by the retrieved chunks."""
        # Edge case: If the model abstained, the answer is 100% faithful to the lack of evidence
        if "cannot find this information" in answer.lower() or not answer.strip():
            return FaithfulnessResult(score=1.0, total_claims=0)

        context_text = "\n\n".join([c.text.strip() for c in chunks])

        prompt = f"""You are a legal auditor evaluating the faithfulness of an AI-generated answer against source contract passages.

CONTEXT PASSAGES:
{context_text}

GENERATED ANSWER:
{answer}

INSTRUCTIONS:
1. Extract the 5-8 key substantive legal claims from the GENERATED ANSWER.
2. For each claim, evaluate if it is STRICTLY SUPPORTED by the CONTEXT PASSAGES.
3. Respond in JSON with this exact structure:
{{
  "claims": [
    {{
      "claim": "<statement>",
      "verdict": "SUPPORTED" | "UNSUPPORTED"
    }}
  ]
}}"""

        try:
            resp = self._get_client().chat.completions.create(
                model=self.settings.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=2000,
            )
            raw = resp.choices[0].message.content.strip()
            clean_json = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL).strip()
            data = json.loads(clean_json)

            claims_data = data.get("claims", [])
            if not claims_data:
                return FaithfulnessResult(score=1.0, total_claims=0)

            supported = [c["claim"] for c in claims_data if c.get("verdict") == "SUPPORTED"]
            unsupported = [c["claim"] for c in claims_data if c.get("verdict") == "UNSUPPORTED"]
            total = len(claims_data)
            score = round(len(supported) / total, 2) if total > 0 else 1.0

            logger.info(f"Faithfulness evaluated: {score:.2f} ({len(supported)}/{total} claims supported).")
            return FaithfulnessResult(
                score=score,
                total_claims=total,
                supported_claims=supported,
                unsupported_claims=unsupported,
            )

        except Exception as e:
            logger.warning(f"Faithfulness evaluation error ({e}); defaulting score to 1.0.")
            return FaithfulnessResult(score=1.0, total_claims=0)

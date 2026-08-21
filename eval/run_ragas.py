"""Offline evaluation script — benchmark RAG pipeline against the 15-question golden dataset."""

import json
import logging
import os
import sys
import time
from typing import Any, Dict, List

sys.path.insert(0, ".")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.core.evaluator import FaithfulnessEvaluator
from app.core.generator import ResponseGenerator
from app.core.retriever import HybridRetriever

logger = logging.getLogger(__name__)


def run_evaluation(golden_path: str = "eval/golden_qa.json") -> Dict[str, Any]:
    """Run full evaluation suite across the golden dataset and compute core RAG metrics."""
    if not os.path.exists(golden_path):
        raise FileNotFoundError(f"Golden dataset not found at: {golden_path}")

    with open(golden_path, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    retriever = HybridRetriever()
    generator = ResponseGenerator()
    evaluator = FaithfulnessEvaluator()

    results = []
    total = len(qa_data)
    faithfulness_scores = []
    citation_hits = 0
    abstention_correct = 0
    abstention_total = 0

    print("=" * 80)
    print(f"Starting Methodical RAG Evaluation across {total} Golden Test Questions")
    print("=" * 80)

    for item in qa_data:
        qid = item["id"]
        question = item["question"]
        ground_truth = item["ground_truth"]
        expected_doc = item.get("expected_doc", "")
        expected_page = item.get("expected_page", 0)
        is_abstention_test = expected_page == 0

        if is_abstention_test:
            abstention_total += 1

        t0 = time.perf_counter()

        # Step 1: Retrieval
        chunks = retriever.retrieve(query=question, top_k=5)

        # Step 2: Generation
        generated = generator.generate(query=question, chunks=chunks)

        # Step 3: Faithfulness Evaluation
        faith_res = evaluator.evaluate(answer=generated.answer, chunks=chunks)

        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        faithfulness_scores.append(faith_res.score)

        # Citation verification
        cited_files = [c.source_file.lower() for c in generated.citations]
        citation_hit = any(expected_doc.lower() in f for f in cited_files) if not is_abstention_test else False
        if citation_hit:
            citation_hits += 1

        # Abstention check
        abstained_ok = False
        if is_abstention_test:
            if "cannot find this information" in generated.answer.lower():
                abstained_ok = True
                abstention_correct += 1

        results.append({
            "id": qid,
            "question": question,
            "answer": generated.answer,
            "confidence": generated.confidence,
            "faithfulness": faith_res.score,
            "citations_count": len(generated.citations),
            "citation_hit": citation_hit if not is_abstention_test else "N/A (Abstention)",
            "abstention_triggered_by": generated.abstention_triggered_by,
            "latency_ms": latency_ms,
        })

        print(f"[{qid}] {question[:55]}...")
        print(f"      Ans: {generated.answer[:75]}...")
        print(f"      Faithfulness: {faith_res.score:.2f} | Citations: {len(generated.citations)} | Latency: {latency_ms}ms\n")

    # Aggregate Metrics
    in_scope_total = total - abstention_total
    avg_faithfulness = sum(faithfulness_scores) / total if total > 0 else 0.0
    citation_accuracy = (citation_hits / in_scope_total * 100.0) if in_scope_total > 0 else 0.0
    abstention_accuracy = (abstention_correct / abstention_total * 100.0) if abstention_total > 0 else 0.0

    summary = {
        "total_queries": total,
        "in_scope_queries": in_scope_total,
        "abstention_queries": abstention_total,
        "average_faithfulness": round(avg_faithfulness, 3),
        "citation_accuracy_pct": round(citation_accuracy, 1),
        "abstention_accuracy_pct": round(abstention_accuracy, 1),
        "results": results,
    }

    # Save results to eval directory
    out_path = "eval/evaluation_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 80)
    print("EVALUATION SUMMARY RESULTS")
    print("=" * 80)
    print(f"Total Evaluated Queries  : {total}")
    print(f"Average Faithfulness     : {avg_faithfulness:.1%}")
    print(f"Citation Precision/Recall: {citation_hits}/{in_scope_total} ({citation_accuracy:.1f}%)")
    print(f"Abstention Precision     : {abstention_correct}/{abstention_total} ({abstention_accuracy:.1f}%)")
    print(f"Detailed results saved to: {out_path}")
    print("=" * 80)

    return summary


if __name__ == "__main__":
    run_evaluation()

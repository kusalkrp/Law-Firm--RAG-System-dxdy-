import json
import os
import sys

sys.path.insert(0, ".")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.core.retriever import HybridRetriever


def run_check() -> float:
    dataset_path = "eval/retrieval_spot_check.json"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Missing {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        queries = json.load(f)

    retriever = HybridRetriever()
    hits = 0
    total = len(queries)

    print("=" * 70)
    print(f"Running Early Retrieval Quality Verification ({total} test queries)")
    print("=" * 70)

    for q in queries:
        query_text = q["question"]
        expected_doc = q["expected_doc"].lower()
        expected_keywords = [k.lower() for k in q.get("expected_keywords", [])]

        results = retriever.retrieve(query_text, top_k=5)

        # Check if any of the top 5 results matches the expected doc and content
        hit = False
        hit_details = []
        for rank, chunk in enumerate(results, start=1):
            src = chunk.metadata.get("source_file", "").lower()
            text = chunk.text.lower()
            heading = chunk.metadata.get("section_heading", "").lower()

            doc_match = expected_doc in src
            keyword_match = any(k in text or k in heading for k in expected_keywords)

            if doc_match and keyword_match:
                hit = True
                hit_details.append(f"Rank {rank} (score={chunk.rerank_score or chunk.rrf_score:.3f})")

        status_str = f"✓ HIT ({', '.join(hit_details)})" if hit else "✗ MISS"
        if hit:
            hits += 1

        print(f"[{q['id']}] {query_text[:50]}...")
        print(f"     Expected: {q['expected_doc']} | Result: {status_str}\n")

    hit_rate = (hits / total) * 100.0
    print("=" * 70)
    print(f"Retrieval Hit@5 Rate: {hits}/{total} ({hit_rate:.1f}%)")
    print(f"Verification Gate (Threshold >= 80%): {'PASSED ✓' if hit_rate >= 80.0 else 'FAILED ✗'}")
    print("=" * 70)

    return hit_rate


if __name__ == "__main__":
    rate = run_check()
    sys.exit(0 if rate >= 80.0 else 1)

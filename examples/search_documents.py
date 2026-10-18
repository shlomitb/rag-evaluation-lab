

"""Try document retrieval with the local embedding model.

Good results should include:
Rank 1 should use the best correct policy.
The content may or may not answer the question,

Questions and Rank 1 responses:
1 - “How long does standard shipping take?”
Source: shipping_policy.txt
Content includes: “Standard shipping typically takes 3–5 business days after the order ships.”

2 - “How much does the wireless mouse cost?”
Source: product_catalog.txt
Content includes: “Wireless Mouse: price $24.99.”

3 - “How long do I have to return an item?”
Source: return_policy.txt
Content includes the standard 30-day return window. A chunk mentioning the 60-day exception for defective items may also be relevant, depending on the wording.

"""

import sys
from pathlib import Path

# Add the project root so Python can import the src package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval_pipeline import search_documents


DOCUMENTS_DIR = PROJECT_ROOT / "data" / "sample_documents"

questions = [
    "How long does standard shipping take?",
    "How much does the wireless mouse cost?",
    "How long do I have to return an item?",
]

for question in questions:
    print(f"\nQuestion: {question}")

    results = search_documents(
        query=question,
        directory=DOCUMENTS_DIR,
        top_k=3,
    )

    for rank, result in enumerate(results, start=1):
        print(f"\n  Rank: {rank}")
        print(f"  Source: {result['source']}")
        print(f"  Similarity: {result['similarity']:.4f}")
        print(f"  Content: {result['content']}")

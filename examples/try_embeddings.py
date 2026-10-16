"""
A small script you used to experiment with the embedding provider.

Run with: python examples/try_embeddings.py
"""

from src.embedding_provider import LocalEmbeddingProvider

provider = LocalEmbeddingProvider()

texts = [
    "Standard shipping takes three to five business days.",
    "How long does delivery take?",
    "The mechanical keyboard has white backlighting.",
]

embeddings = provider.embed_texts(texts)

for text, vector in zip(texts, embeddings):
    print(f"\nText: {text}")
    print(f"Vector dimensions: {len(vector)}")
    print(f"First five values: {vector[:5]}")

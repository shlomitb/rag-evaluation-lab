# RAG Testing and Evaluation Guide

## 1. What is RAG?

**RAG** stands for **Retrieval-Augmented Generation**. It is a way to give a Large Language Model (LLM) relevant information from an external source when answering a question.

Instead of relying only on what the model learned during training, a RAG system retrieves relevant information from documents and supplies that information to the model as context.

### The main stages of a RAG system

1. **Document loading:** Read the source documents.
2. **Chunking:** Split documents into smaller pieces of text.
3. **Embedding:** Convert each chunk into a numerical vector representing aspects of its meaning.
4. **Indexing:** Store the chunks and their embeddings so they can be retrieved efficiently.
5. **Retrieval:** Convert the user's question into an embedding and find chunks that are semantically similar.
6. **Context construction:** Select and prepare retrieved chunks for the LLM.
7. **Generation:** The LLM uses the question and retrieved context to produce an answer.

Our project currently implements document loading, chunking, embedding, and similarity-based retrieval. We are beginning to evaluate how well retrieval works. Answer generation and end-to-end RAG evaluation will come later.

## 2. Why does RAG need testing?

A RAG system can fail in several different ways:

- Important information might be lost or separated awkwardly during chunking.
- The embedding model might not represent a question and its answer-related text closely enough.
- Retrieval might return irrelevant chunks or rank the right chunk too low.
- The correct information might not be retrieved at all.
- The LLM might ignore relevant context, misunderstand it, or invent information.
- Changes intended to improve one part of the system might make another part worse.

A system can produce a fluent, convincing answer and still be wrong. Therefore, answer quality alone is not enough to establish that a RAG system is reliable.

**Key principle:** Test the individual stages as well as the complete system. When something fails, we want to identify which stage caused the failure.

## 3. Chunking

Chunking divides a document into smaller pieces that can be embedded and retrieved separately.

### Character-based chunking

The document is split according to a maximum number of characters, often with some overlap between adjacent chunks.

Advantages:
- Simple to implement.
- Predictable chunk sizes.
- Useful as a baseline.

Disadvantages:
- A chunk may start or end in the middle of a sentence or fact.
- Related information may be split across chunks.
- A retrieved chunk may contain much more information than the question needs.

### Semantic or structure-aware chunking

Chunks are created using meaningful boundaries, such as paragraphs, headings, or numbered policy items.

Advantages:
- A chunk is more likely to contain a complete fact.
- Retrieved context can be more focused.
- Headings can preserve important context about the information.

Disadvantages:
- Some chunks may become too long.
- Documents do not all have the same structure.
- A chunking strategy that works well for policies might not work equally well for every document type.

### How to test chunking

Useful tests include:

- All meaningful source text is preserved.
- Empty documents produce no chunks.
- Chunk metadata retains the original filename.
- Chunk IDs are unique within the ingested dataset.
- Overlap behaves as intended.
- Numbered facts remain separate when the strategy is designed to preserve them.
- Chunks do not exceed the configured size unless the design explicitly permits it.

**Important:** Passing these tests proves that the implementation follows its rules. It does not prove that the chunks are optimal for retrieval. That requires retrieval experiments.

## 4. Embeddings and similarity

An embedding model converts text into a numerical vector. Texts with related meanings often have vectors that point in similar directions.

Our project uses a local Sentence Transformers model, so basic experiments can run without a paid embedding API.

### Cosine similarity

Cosine similarity measures the angle between two non-zero vectors.

- A value near **1** means the vectors point in similar directions.
- A value near **0** means they are approximately perpendicular.
- A value near **-1** means they point in opposite directions.

The interpretation depends on the embedding model and data.

**A similarity score is not a probability or a guarantee of correctness.** A score of 0.80 does not mean there is an 80% chance that a chunk is relevant. Scores are most useful when comparing results under a consistent setup.

### How to test embeddings

- The provider returns one vector per input text.
- Every vector has the same dimension.
- Vector values are valid finite numbers.
- Similarity calculations handle invalid inputs correctly.
- Known test vectors produce expected similarity values.

These are primarily deterministic tests: we can specify the expected result without asking an LLM to judge it.

## 5. Retrieval evaluation

Retrieval evaluation asks:

**Did the system retrieve the information needed to answer the question, and did it rank that information highly enough?**

To measure this, we need a small evaluation dataset containing questions and the chunk or chunks that are known to be relevant.

This is sometimes called a **golden dataset** or **evaluation dataset**.

For example:

| Question | Relevant chunk |
|---|---|
| How long does standard shipping take? | The standard-shipping delivery-time item |
| What is the price of the wireless mouse? | The wireless-mouse catalog item |
| How long do I have to return an item? | The standard return-window item |

The expected results are defined before running the retrieval system. This lets us measure its behavior rather than judging it only by whether a result looks plausible.

### Hit Rate@K

Hit Rate@K measures the proportion of evaluation questions for which at least one relevant chunk appears among the top K retrieved results.

For example, if we evaluate 10 questions using the top 3 results and at least one relevant chunk appears in the top 3 for 8 questions:

**Hit Rate@3 = 8 / 10 = 0.80 (80%)**

A higher value means relevant information is found more often within the chosen cutoff.

### Mean Reciprocal Rank (MRR)

MRR measures how highly the first relevant result is ranked.

For one question:

- Relevant result at rank 1: reciprocal rank = 1
- Relevant result at rank 2: reciprocal rank = 1/2
- Relevant result at rank 3: reciprocal rank = 1/3
- No relevant result: reciprocal rank = 0

MRR is the average reciprocal rank across all evaluation questions.

A higher MRR means the first relevant result tends to appear closer to the top.

### Why use both metrics?

Hit Rate@K tells us whether a relevant result appears within the cutoff. MRR tells us whether the first relevant result appears near the top.

A system can have a good Hit Rate@3 but a weaker MRR if it often finds the right chunk only at rank 3.

## 6. Testing retrieval versus testing generated answers

These are different evaluation tasks.

### Retrieval evaluation

Checks whether the right information was found and ranked appropriately.

Examples:
- Was the standard-shipping delivery time retrieved?
- Did the wireless-mouse price appear in the top three results?
- Was a relevant chunk ranked above unrelated chunks?

Retrieval metrics can often be calculated deterministically using known relevant chunk IDs.

### Answer evaluation

Checks whether the final answer is good, given the question and retrieved context.

Important dimensions include:

- **Answer relevance:** Does the answer address the user's question?
- **Faithfulness or groundedness:** Are the answer's claims supported by the retrieved context?
- **Correctness:** Does the answer agree with the known facts or expected answer?
- **Completeness:** Does it include the important information needed to answer the question?

These dimensions overlap, but they are not interchangeable. An answer can be relevant to a question yet contain unsupported claims.

Once our project includes answer generation, we will test these dimensions separately.

## 7. Deterministic tests versus LLM-as-judge evaluation

### Deterministic tests

Use explicit rules and expected values.

Examples:
- The number of embeddings matches the number of input texts.
- Similarity between two identical non-zero vectors is 1.
- The expected relevant chunk appears in the top three results.
- A chunk retains its source filename.

Advantages:
- Usually inexpensive and fast.
- Repeatable.
- Failures are easier to diagnose.
- No judge model is required.

Limitation: deterministic checks can only evaluate properties that we have explicitly defined.

### LLM-as-judge evaluation

Uses an LLM to assess qualities that are difficult to capture with simple rules, such as whether an answer is sufficiently relevant or whether a paraphrase is supported by the context.

Advantages:
- Can assess nuanced natural-language behavior.
- Useful for large or varied answer sets.

Limitations:
- Can be inconsistent or incorrect.
- May be sensitive to the judge prompt and model.
- Can incur API cost and latency.
- Needs its own validation and calibration.

**Best practice:** Use deterministic checks wherever possible, and introduce LLM-based evaluation when the property being measured genuinely requires semantic judgment.

## 8. Regression testing

Regression testing checks whether a change has broken behavior that previously worked.

For example, after changing from character-based to semantic chunking, we should rerun the same retrieval evaluation dataset.

We want to know:
- Did Hit Rate@3 improve or decline?
- Did MRR improve or decline?
- Did some questions become harder to retrieve correctly?
- Did a change improve one kind of question but harm another?

A single successful example is encouraging, but it cannot establish that the system has improved overall.

## 9. Interpreting evaluation results carefully

A useful evaluation process should:

1. Use a fixed set of representative questions.
2. Define relevant chunks or expected facts before evaluating the system.
3. Compare alternative approaches using the same questions and metrics.
4. Inspect individual failures instead of relying only on an aggregate score.
5. Keep track of configuration changes, such as the chunking strategy, embedding model, and top-K setting.
6. Rerun evaluations after meaningful changes.

Small evaluation datasets are useful for learning and regression checks, but their scores may not generalize to real-world users. We should gradually expand the dataset to cover different question types and failure cases.

## 10. Our project roadmap

We will build and test the system incrementally.

- [x] Load text documents and preserve source filenames.
- [x] Implement and test character-based chunking.
- [x] Preserve source metadata and create chunk IDs.
- [x] Validate embedding vectors.
- [x] Generate local embeddings.
- [x] Implement and test cosine similarity.
- [x] Retrieve and rank chunks by similarity.
- [x] Implement structure-aware semantic chunking.
- [x] Connect ingestion, embeddings, and retrieval into a pipeline.
- [ ] Create a retrieval evaluation dataset.
- [ ] Implement and test Hit Rate@K and MRR.
- [ ] Run a baseline evaluation and investigate failures.
- [ ] Compare chunking strategies using the same evaluation dataset.
- [ ] Add answer generation and evaluate answer quality.
- [ ] Add end-to-end RAG regression tests.
- [ ] Explore a framework such as Ragas and compare its metrics with our own implementation.

This guide will evolve as we implement each stage and learn from the results.

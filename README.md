# RAG Evaluation Lab

A hands-on project for learning how to build, test, and evaluate Retrieval-Augmented Generation (RAG) systems.

The project focuses on software quality and evaluation throughout the RAG pipeline, including document ingestion, chunking, embeddings, retrieval, answer generation, answer quality, and regression testing.

## Project Goals

- Build a RAG pipeline incrementally, with tests for each component.
- Learn how design choices affect retrieval quality and answer reliability.
- Develop deterministic tests for predictable behavior.
- Add evaluation methods for properties that cannot be verified with ordinary assertions alone.
- Distinguish retrieval failures from answer-generation failures.
- Build a portfolio project demonstrating AI quality engineering skills.

## Current Capabilities

**Document processing**
- **Document loading:** Load text files and preserve their source filenames.
- **Text chunking:** Split text into overlapping chunks, including semantic chunking.
- **Chunk metadata:** Preserve source filenames, chunk indexes, and IDs.
- **Document ingestion:** Process a directory of documents into a list of metadata-rich chunks.

**Embeddings and retrieval**
- **Local embeddings:** Generate text embeddings using Sentence Transformers and `all-MiniLM-L6-v2`.
- **Embedding validation:** Check vector counts, dimensions, and values.
- **Embedding provider interface:** Inject alternative embedding implementations for testing.
- **Similarity-based retrieval:** Retrieve relevant document chunks using query and chunk embeddings.
- **Retrieval evaluation:** Evaluate retrieval using a JSON dataset, including relevance, ranking, paraphrased queries, multi-fact questions, and unanswerable questions.

**RAG answer generation**
- **Context-based prompts:** Build prompts from retrieved document chunks.
- **Generator interface:** Use a `TextGenerator` protocol to separate answer-generation logic from a specific model implementation.
- **No-context handling:** Return a predefined response when no chunks are retrieved.
- **RAG pipeline:** Connect retrieval and answer generation through a single question-answering interface.
- **Evaluation evidence:** Preserve retrieved chunks alongside generated answers for subsequent failure analysis.

**Answer evaluation**
- **Evaluation dataset:** Define answer-evaluation cases in JSON, covering product information, shipping policies, and return policies.
- **Deterministic checks:** Check for required and forbidden phrases.
- **Coverage measurement:** Calculate required-phrase coverage for each answer.
- **Evaluation runner:** Evaluate multiple answers and summarize pass/fail results and overall pass rate.
- **Dataset validation:** Check dataset structure, required fields, unique IDs, and phrase definitions.

**Automated testing**
- Use Pytest to test individual components and their integration.
- Use fake embedding providers and text generators to test behavior without external model calls.
- Validate evaluation logic and error handling independently of actual LLM output.

## Technology

- Python
- Pytest
- Sentence Transformers
- `all-MiniLM-L6-v2` embedding model

The embedding model runs locally after its initial download. The current implementation does not require a paid embedding API.

The answer-generation component defines a model-independent interface. Evaluation can be tested using fake generators without calling a paid LLM API.

## Project Structure

```text
rag-evaluation-lab/
├── data/
│   ├── sample_documents/
│   ├── retrieval_eval.json
│   └── answer_eval.json
├── docs/
│   └── RAG_TESTING_GUIDE.md
├── examples/
│   └── try_embeddings.py
├── src/
│   ├── document_loader.py
│   ├── chunking.py
│   ├── chunk_metadata.py
│   ├── ingestion.py
│   ├── embeddings.py
│   ├── embedding_provider.py
│   ├── retrieval.py
│   ├── retrieval_pipeline.py
│   ├── answer_generation.py
│   ├── rag_pipeline.py
│   ├── answer_evaluation.py
│   ├── answer_evaluation_runner.py
│   └── rag_evaluation.py
├── tests/
├── requirements.txt
├── requirements-dev.txt
├── README.md
└── .gitignore
```

## Setup

Create and activate a virtual environment before installing dependencies.

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
```

## Run the Tests

Run the complete test suite:

```powershell
python -m pytest -v
```

Run tests for an individual component:

```powershell
python -m pytest tests/test_chunking.py -v
```

## Evaluation Approach

The project uses separate datasets and evaluation methods for different RAG quality dimensions.

### Retrieval evaluation

`data/retrieval_eval.json` defines questions and expected retrieval behavior.

Retrieval metrics include Hit Rate@k and Mean Reciprocal Rank (MRR). Additional checks evaluate whether retrieved results contain the information needed to answer multi-fact questions.

### Answer evaluation

`data/answer_eval.json` defines expected-answer checks for questions about the sample product catalog, shipping policy, and return policy.

The deterministic evaluator checks required

# RAG Evaluation Lab

A hands-on project for learning how to build, test, and evaluate
Retrieval-Augmented Generation (RAG) systems.

The project focuses on software quality and evaluation throughout
the RAG pipeline, including document ingestion, chunking, embeddings,
retrieval, answer quality, and regression testing.

## Project Goals

- Build a RAG pipeline incrementally, with tests for each component.
- Learn how design choices affect retrieval quality and answer reliability.
- Develop deterministic tests for predictable behavior.
- Add evaluation methods for properties that cannot be verified
  with ordinary assertions alone.
- Build a portfolio project demonstrating AI quality engineering skills.

## Current Capabilities

- **Document loading:** Load text files and preserve their source filenames.
- **Text chunking:** Split text into overlapping chunks.
- **Chunk metadata:** Preserve source filenames, chunk indexes, and IDs.
- **Document ingestion:** Process a directory of documents into a single
  list of metadata-rich chunks.
- **Embedding validation:** Check vector counts, dimensions, and values.
- **Local embeddings:** Generate text embeddings using
  Sentence Transformers and `all-MiniLM-L6-v2`.

## Technology

- Python
- Pytest
- Sentence Transformers
- `all-MiniLM-L6-v2` embedding model

The embedding model runs locally after its initial download.
The current implementation does not require a paid embedding API.

## Project Structure

```text
rag-evaluation-lab/
├── data/
│   └── sample_documents/
├── docs/
├── examples/
│   └── try_embeddings.py
├── src/
│   ├── document_loader.py
│   ├── chunking.py
│   ├── chunk_metadata.py
│   ├── ingestion.py
│   ├── embeddings.py
│   └── embedding_provider.py
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

## Learning and Evaluation Roadmap

- [x] Load text documents
- [x] Split documents into overlapping chunks
- [x] Preserve source and chunk metadata
- [x] Build a document ingestion pipeline
- [x] Test chunk integrity
- [x] Validate embedding output
- [x] Generate embeddings with a local model
- [ ] Evaluate embedding behavior and similarity calculations
- [ ] Implement vector-based retrieval
- [ ] Evaluate retrieval relevance and ranking
- [ ] Build a RAG question-answering pipeline
- [ ] Evaluate answer correctness, faithfulness, and groundedness
- [ ] Add regression evaluation reports

## Quality Engineering Approach

The project uses deterministic tests wherever expected behavior
can be specified precisely. More complex properties, such as
semantic relevance and answer faithfulness, will require additional
evaluation techniques.

The goal is not just to make a RAG pipeline work, but to understand
how to measure its quality, identify failures, and prevent regressions.

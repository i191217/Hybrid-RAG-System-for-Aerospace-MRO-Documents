# Hybrid RAG System for Aerospace MRO Documents

A retrieval-augmented generation (RAG) system tailored for aerospace maintenance, repair and overhaul (MRO) documentation. This solution processes complex aerospace documents (PDFs, scanned manuals etc.), builds a vector database, and supports intelligent query/response workflows.

---

## Motivation

In aerospace MRO domains, documentation is often voluminous, diverse (PDFs, scans, manuals, OEM instructions) and richly structured.
Traditional search methods struggle with semantics and context. By combining retrieval (vector embeddings + document chunks) with generation (LLM-based responses), we can enable:

* Rapid access to relevant manual sections
* Context-aware answers (e.g., safety instructions, repair steps)
* Hybrid processing of text + scanned content
  This system is built to support such workflows with a modular Python codebase.

---

## Overview

The system implements a hybrid RAG pipeline comprising:

1. **Document ingestion & filtering**: ingest aerospace-MRO documentation, perform OCR if needed, filter irrelevant content.
2. **Chunking & embedding**: break into logical units (blocks/paragraphs), embed into a vector database.
3. **Retrieval engine**: given a user query, retrieve relevant chunks from the vector database.
4. **Generation engine**: pass retrieved context + query into a large language model (LLM) to produce coherent, context-aware answers.
5. **API / Front-end**: expose endpoints or a simple front-end to allow users to ask questions and receive responses.

---

## Features

* 🧠 Hybrid ingestion of textual + scanned PDF documents (via OCR).
* Document filtering (to remove boilerplate, duplicates).
* Chunk-based content analysis to maintain semantic coherence.
* Vector store integration for fast semantic retrieval.
* Query engine to combine retrieval + generation.
* Modular pipeline with reusable components (ingest, embed, query, generation).
* Designed for aerospace MRO-specific docs but extensible to other domains.

---

## Getting Started

### Prerequisites

* Python 3.8+
* Virtual environment (venv, conda)
* Access to an LLM / embedding model (open-source or API)
* (Optional) A vector store (e.g., FAISS, Pinecone, etc)
* Documents: aerospace MRO manuals, scanned PDFs, OEM MRO docs

### Installation

```bash
git clone https://github.com/i191217/Hybrid-RAG-System-for-Aerospace-MRO-Documents.git  
cd Hybrid-RAG-System-for-Aerospace-MRO-Documents  
python3 -m venv venv  
source venv/bin/activate       # (on Windows: venv\Scripts\activate)  
pip install -r requirements.txt  
```

### Configuration

Edit `config.py` to set:

* Model paths / API keys (for embedding model / LLM)
* Vector DB connection settings
* Document ingestion paths (raw docs folder, output folder)
* OCR settings (if scanning)
* Logging / output paths

---

## Usage

### 1. Ingest Documents

```bash
python ingest.py --input_dir path/to/raw_docs --output_dir path/to/processed_docs  
```

This handles filtering, OCR, chunking and saving intermediate outputs.

### 2. Build Vector DB

```bash
python extract_vectordb_chunks.py --docs_dir path/to/processed_docs --vector_db path/to/vector_store  
python embedding_service.py --input_chunks chunks.json --vector_db path/to/vector_store  
```

### 3. Run the Pipeline

```bash
python run_pipeline.py  
```

This orchestrates ingestion → embedding → vector DB → indexing.

### 4. Query the System

Via CLI:

```bash
python query.py --query "What are the safety inspection intervals for the hydraulic pump in aircraft model XYZ?"  
```

Via API:

```bash
python run_api.py  
# Then send HTTP request: POST /ask { "question": "..."}  
```

### 5. Front-end (optional)

A minimal web UI (if you build one) can call the API server `app.py` and display answers plus link back to original document chunks.

---

## Example Workflow

1. Drop new OEM MRO manuals (PDFs) into the `raw_docs` folder.
2. Run ingestion to OCR and chunk them.
3. Build embeddings and vector DB.
4. Ask the system: “What is the torque spec for fastener ABC 123?”
5. The system retrieves relevant chunk(s) + uses LLM to generate a user-friendly answer with reference to the original manual.
6. Logs & retrieval analysis (using `retrieval_analysis.txt`) can help you fine-tune chunking parameters or filtering logic.

---

## Contributing

We welcome contributions!
If you would like to improve the system:

* Fork the repo.
* Create a new branch (`feature/…` or `bugfix/…`).
* Add appropriate tests or documentation.
* Submit a pull request describing your change.

Please ensure:

* Code is formatted according to Python best practices (PEP 8).
* New dependencies are added to `requirements.txt`.
* Document your change in this README or comment blocks.

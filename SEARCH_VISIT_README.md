# Enhanced RAG with Search and Visit Site Functionality

This implementation provides an enhanced RAG (Retrieval-Augmented Generation) system that combines search functionality with the ability to visit specific pages for detailed content.
## Setup

PyPI [coming soon](https://pypi.org/project/verifiers/), for now just do:
```bash
git clone https://github.com/willccbb/verifiers.git
cd verifiers
uv sync
uv pip install flash-attn --no-build-isolation
uv pip install -e .
# for start rag server
uv pip install faiss-cpu sentence-transformers
source .venv/bin/activate
```

## Features

### 1. Enhanced Search (`search_with_urls`)

- Returns search results with URLs and previews
- Each result includes:
    - Title
    - URL/Document ID
    - Preview (first 2-3 sentences)
- Allows the model to make informed decisions about which pages to visit

### 2. Visit Site (`visit_site`)

- Retrieves full content for a specific URL/document
- Returns complete document content
- Enables deep dive into promising search results

### 3. Combined Workflow

- Model first searches for relevant information
- Reviews search results with previews
- Decides which pages need full content
- Visits specific pages for detailed information


## Requirements

- FlashRAG server running on localhost:2223
- Properly indexed document corpus
- Python packages: requests, json, re, typing

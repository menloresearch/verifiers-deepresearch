# Enhanced RAG with Search and Visit Site Functionality

This implementation provides an enhanced RAG (Retrieval-Augmented Generation) system that combines search functionality with the ability to visit specific pages for detailed content.

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

## Usage

### Basic Search

```python
from verifiers.tools.search_visit_rag import search_with_urls

results = search_with_urls("machine learning algorithms", num_results=3)
print(results)
```

### Visit Specific Page

```python
from verifiers.tools.search_visit_rag import visit_site

content = visit_site("doc_1")
print(content)
```

### Model Interaction Example

The model can now use both tools in sequence:

1. **Search Phase:**

```xml
<tool>
{"name": "search_with_urls", "args": {"query": "McDonald's founding date", "num_results": 3}}
</tool>
```

2. **Review Results:**

```
Result 1:
Title: McDonald's Corporation History
URL: doc_1
Preview: McDonald's was founded in 1940 by Richard and Maurice McDonald...

Result 2:
Title: Ray Kroc Biography
URL: doc_2
Preview: Ray Kroc joined McDonald's in 1955 and transformed it...
```

3. **Visit Promising Pages:**

```xml
<tool>
{"name": "visit_site", "args": {"url": "doc_1"}}
</tool>
```

## Training Configuration

The training script (`trl_deepresearch_search_visit_site_offline.py`) has been updated to:

- Import the new search and visit functions
- Include both tools in the environment
- Update the system prompt with examples of the new workflow
- Increase max_steps to accommodate multi-step search and visit patterns

## Key Benefits

1. **Efficient Information Gathering**: Model can preview results before committing to full content retrieval
2. **Targeted Deep Dives**: Only visit pages that appear most relevant based on previews
3. **Better Resource Utilization**: Avoid processing irrelevant full documents
4. **Improved Decision Making**: Model can make informed choices about which sources to explore

## Implementation Details

### Search Function

- Connects to FlashRAG server on localhost:2223
- Returns formatted results with metadata
- Includes preview generation from document text
- Handles errors gracefully

### Visit Function

- Retrieves specific documents by URL/ID
- Returns full content with title and metadata
- Optimized for detailed content analysis

### Backward Compatibility

- Original `search_rag` function still available
- Existing code continues to work
- Gradual migration path available

## Testing

Run the test script to verify functionality:

```bash
python test_search_visit.py
```

This will demonstrate both search and visit capabilities with sample queries.

## Requirements

- FlashRAG server running on localhost:2223
- Properly indexed document corpus
- Python packages: requests, json, re, typing

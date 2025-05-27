import re

import requests


def search_with_urls(query: str, num_results: int = 3) -> dict:
    """Enhanced search function that returns URLs and previews for each result.

    Args:
        query: The search query string
        num_results: Number of results to return (default: 3)

    Returns:
        Dictionary with search results including URLs and previews
    """
    server_url = "http://localhost:2223"

    try:
        payload = {
            "queries": [query],
            "topk_retrieval": max(num_results * 3, 10),
            "topk_rerank": num_results,
            "return_scores": True,
        }

        response = requests.post(
            f"{server_url}/retrieve", json=payload, headers={"Content-Type": "application/json"}, timeout=600
        )

        if response.status_code != 200:
            return {"error": f"RAG server returned status {response.status_code}: {response.text}"}

        result = response.json()
        documents = result.get("result", [[]])[0]

        if not documents:
            return {"result": [], "error": "No results found"}

        # Format results with URLs and previews
        formatted_results = []
        for i, doc in enumerate(documents, 1):
            title = doc.get("title", f"Document {i}")
            text = doc.get("text", "").strip()

            # Extract or generate URL from metadata
            metadata = doc.get("metadata", {})
            url = metadata.get("url", metadata.get("source", f"doc_{i}"))

            # Create preview (first 2-3 sentences)
            preview = _create_preview(text)

            formatted_results.append(
                {
                    "doc_id": doc.get("doc_id", f"doc_{i}"),
                    "title": title,
                    "url": url,
                    "text": preview,
                    "score": doc.get("score", 0.0),
                }
            )

        return {"result": [formatted_results]}

    except requests.exceptions.ConnectionError:
        return {"error": "Could not connect to RAG server. Please ensure the server is running."}
    except requests.exceptions.Timeout:
        return {"error": "Request to RAG server timed out."}
    except Exception as e:
        return {"error": str(e)}


def visit_site(url: str) -> dict:
    """Visit a specific URL and return its full content.

    Args:
        url: The URL to visit

    Returns:
        Dictionary with full content of the page
    """
    server_url = "http://localhost:2223"

    try:
        # For RAG server, we need to query for the specific document
        payload = {"queries": [f"url:{url}"], "topk_retrieval": 1, "topk_rerank": 1, "return_scores": False}

        response = requests.post(
            f"{server_url}/retrieve", json=payload, headers={"Content-Type": "application/json"}, timeout=600
        )

        if response.status_code != 200:
            return {"error": f"Could not visit {url}. Server returned status {response.status_code}"}

        result = response.json()
        documents = result.get("result", [[]])[0]

        if not documents:
            return {"error": f"Could not find content for {url}"}

        doc = documents[0]
        title = doc.get("title", "Untitled")
        content = doc.get("text", "").strip()

        return {"result": [{"doc_id": doc.get("doc_id", url), "title": title, "url": url, "text": content}]}

    except Exception as e:
        return {"error": f"Error visiting {url}: {str(e)}"}


def _create_preview(text: str, max_sentences: int = 3, max_chars: int = 200) -> str:
    """Create a preview from text content."""
    if not text:
        return "No preview available"

    # Split into sentences
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    preview = ""
    for i, sentence in enumerate(sentences[:max_sentences]):
        if len(preview + sentence) > max_chars:
            break
        preview += sentence + ". "

    if len(preview) > max_chars:
        preview = preview[:max_chars].rsplit(" ", 1)[0] + "..."

    return preview.strip()

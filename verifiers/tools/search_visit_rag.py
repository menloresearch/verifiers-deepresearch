import requests
import re
import os

def web_search(query: str) -> str:
    """Enhanced search function that returns URLs and previews for each result.
    
    Args:
        query: The search query string
        
    Returns:
        Formatted string with search results including URLs and previews
    """
    server_url = os.environ.get("RAG_SERVER_URL", "http://localhost:2223") 
    num_results = 10
    
    try:
        payload = {
            "queries": [query],
            "topk_retrieval": max(num_results * 3, 15),
            "topk_rerank": num_results,
            "return_scores": False
        }
        
        response = requests.post(
            f"{server_url}/retrieve",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=600
        )
        
        if response.status_code != 200:
            return f"Error: RAG server returned status {response.status_code}: {response.text}"
        
        result = response.json()
        documents = result.get('result', [[]])[0]
        
        if not documents:
            return "No results found"
        
        # Format results with URLs and previews
        formatted_results = []
        for i, doc in enumerate(documents, 1):
            title = doc.get('title', f'Document {i}')
            text = doc.get('text', '').strip()
            
            # Extract or generate URL from metadata
            # metadata = doc.get('metadata', {})
            url = f"doc_{doc.get('doc_id')}"
            
            # Create preview (first 2-3 sentences)
            preview = _create_preview(text)
            
            formatted_results.append(
                f"Result {i}:\n"
                f"Title: {title}\n"
                f"URL: {url}\n"
                f"Preview: {preview}\n"
            )
        
        return "\n".join(formatted_results)
        
    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to RAG server. Please ensure the server is running."
    except requests.exceptions.Timeout:
        return "Error: Request to RAG server timed out."
    except Exception as e:
        return f"Error: {str(e)}"


def visit_tool(url: str) -> str:
    """Visit a specific URL and return its full content.
    
    Args:
        url: The URL to visit
        
    Returns:
        Full content of the page
    """
    server_url = os.environ.get("RAG_SERVER_URL", "http://localhost:2223") 
    
    try:
        # For RAG server, we need to query for the specific document
        payload = {
            "url": url
        }
        
        response = requests.post(
            f"{server_url}/visit",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=600
        )
        
        if response.status_code != 200:
            return f"Error: Could not visit {url}. Server returned status {response.status_code}"
        
        result = response.json()
        documents = result.get('result', [[]])[0]
        
        if not documents:
            return f"Error: Could not find content for {url}"
        
        doc = documents[0]
        title = doc.get('title', 'Untitled')
        content = doc.get('text', '').strip()
        
        return f"Title: {title}\nURL: {url}\n\nFull Content:\n{content}"
        
    except Exception as e:
        return f"Error visiting {url}: {str(e)}"


def search_and_visit_rag(query: str, num_results: int = 3, visit_threshold: float = 0.7) -> str:
    """Combined search and visit function that intelligently decides when to visit pages.
    
    Args:
        query: The search query string
        num_results: Number of search results to return
        visit_threshold: Threshold for deciding whether to visit a page (not used in this simple version)
        
    Returns:
        Search results with option to visit specific pages
    """
    # First, perform search
    search_results = search_with_urls(query, num_results)
    
    if "Error:" in search_results or "No results found" in search_results:
        return search_results
    
    # Add instructions for visiting pages
    instructions = (
        "\nTo visit any of these pages for full content, use the visit_site tool with the URL.\n"
        "Example: visit_site(\"doc_1\") or visit_site(\"https://example.com\")\n"
    )
    
    return search_results + instructions


def _create_preview(text: str, max_sentences: int = 2, max_chars: int = 200) -> str:
    """Create a preview from text content."""
    if not text:
        return "No preview available"

    # Split into sentences
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    preview = sentences[0]
    for sentence in sentences[1:max_sentences]:
        if len(preview + sentence) > max_chars:
            break
        preview += sentence + ". "

    if len(preview) > max_chars:
        preview = preview[:max_chars].rsplit(' ', 1)[0] + "..."

    return preview.strip()


# For backward compatibility
def search_rag(query: str, num_results: int = 3) -> str:
    """Original search_rag function for backward compatibility."""
    return search_and_visit_rag(query, num_results) 
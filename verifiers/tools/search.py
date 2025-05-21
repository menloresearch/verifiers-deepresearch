def search_rag(query: str, num_results: int = 3, 
               server_url: str = "http://localhost:2223") -> str:
    """Searches using FlashRAG server and returns formatted results.
    
    Args:
        query: The search query string
        num_results: Number of results to return (default: 3)
        server_url: URL of the FlashRAG server (default: http://localhost:2223)
        
    Returns:
        Formatted string with bullet points of top results, each with title and text content
        
    Examples:
        {"query": "what is machine learning", "num_results": 3}
    """
    import requests
    import json
    
    try:
        # Prepare the request payload
        payload = {
            "queries": [query],
            "topk_retrieval": max(num_results * 3, 10),  # Retrieve more to ensure good reranking
            "topk_rerank": num_results,
            "return_scores": False
        }
        
        # Make the API call
        response = requests.post(
            f"{server_url}/retrieve",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=600
        )
        
        # Check if the request was successful
        if response.status_code != 200:
            return f"Error: RAG server returned status {response.status_code}: {response.text}"
        
        # Parse the response
        result = response.json()
        documents = result.get('result', [[]])[0]  # Get documents for the first (and only) query
        
        if not documents:
            return "No results found"
        
        # Format the results similar to other search functions
        summaries = []
        for i, doc in enumerate(documents, 1):
            title = doc.get('title', f'Document {i}')
            text = doc.get('text', '').strip()
            
            # Truncate text if too long (similar to other search functions)
            if len(text) > 300:
                text = text[:300].rsplit('.', 1)[0] + '...'
            elif len(text) > 200:
                text = text[:200].rsplit(' ', 1)[0] + '...'
            
            summaries.append(f"• {title}\n  {text}")
        
        return "\n\n".join(summaries)
        
    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to RAG server. Please ensure the server is running."
    except requests.exceptions.Timeout:
        return "Error: Request to RAG server timed out."
    except requests.exceptions.RequestException as e:
        return f"Error: Request failed: {str(e)}"
    except json.JSONDecodeError:
        return "Error: Invalid response format from RAG server."
    except Exception as e:
        return f"Error: {str(e)}"

def search_ddg(query: str, num_results: int = 5) -> str:
    """Searches DuckDuckGo and returns concise summaries of top results.
    
    Args:
        query: The search query string
        num_results: Number of results to return (default: 5, max: 10)
        
    Returns:
        Formatted string with bullet points of top results, each with title and brief summary
        
    Examples:
        {"query": "who invented the lightbulb", "num_results": 3}
    """
    from duckduckgo_search import DDGS

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=min(num_results, 10)))
            if not results:
                return "No results found"

            summaries = []
            for r in results:
                title = r['title']
                snippet = r['body'][:200].rsplit('.', 1)[0] + '.'
                summaries.append(f"• {title}\n  {snippet}")

            return "\n\n".join(summaries)
    except Exception as e:
        return f"Error: {str(e)}" 
    
def search(query: str) -> str:
    """Searches the web and returns summaries of top results.
    
    Args:
        query: The search query string

    Returns:
        Formatted string with bullet points of top 3 results, each with title, source, url, and brief summary

    Examples:
        {"query": "who invented the lightbulb"} -> ["Thomas Edison (1847-1931) - Inventor of the lightbulb", ...]
        {"query": "what is the capital of France"} -> ["Paris is the capital of France", ...]
        {"query": "when was the Declaration of Independence signed"} -> ["The Declaration of Independence was signed on July 4, 1776", ...]
    """
    from brave import Brave
    from typing import List, Dict, Any

    try:
        brave = Brave()
        results = brave.search(q=query, count=10, raw=True) # type: ignore
        web_results = results.get('web', {}).get('results', []) # type: ignore
        
        if not web_results:
            return "No results found"

        summaries = []
        for r in web_results:
            if 'profile' not in r:
                continue
            header = f"{r['profile']['name']} ({r['profile']['long_name']})"
            title = r['title']
            snippet = r['description'][:300] + " ..."
            url = r['url'] 
            summaries.append(f"•  {header}\n   {title}\n   {snippet}\n   {url}")

        return "\n\n".join(summaries[:3])
    except Exception as e:
        return f"Error: {str(e)}"

if __name__=="__main__":
    result = search_rag("what is machine learning", num_results=3)
    print("Search Results:")
    print(result)
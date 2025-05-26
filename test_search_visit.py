#!/usr/bin/env python3

from verifiers.tools.search_visit_rag import search_with_urls, visit_site


def test_search_and_visit():
    """Test the new search and visit functionality."""

    print("=== Testing Search with URLs ===")
    query = "machine learning algorithms"
    search_results = search_with_urls(query, num_results=3)
    print(f"Query: {query}")
    print(f"Results:\n{search_results}")

    print("\n=== Testing Visit Site ===")
    # Try to visit the first document from search results
    url = "doc_1"  # This would be extracted from search results in practice
    visit_results = visit_site(url)
    print(f"Visiting: {url}")
    print(f"Content:\n{visit_results}")


if __name__ == "__main__":
    test_search_and_visit()

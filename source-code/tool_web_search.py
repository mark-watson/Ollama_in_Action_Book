"""
Provides functions for web searching and HTML to Markdown conversion
and for returning the contents of a URI as plain text (with minimal markdown)
"""

from typing import Dict, Any
import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urlparse
import html
from ollama import chat
import json
from tool_summarize_text import summarize_text

import requests
import os
import logging
from pprint import pprint
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)

api_key = os.environ.get("BRAVE_SEARCH_API_KEY")
if not api_key:
    raise ValueError(
        "API key not found. Set 'BRAVE_SEARCH_API_KEY' environment variable."
    )


def replace_html_tags_with_text(html_string):
    soup = BeautifulSoup(html_string, "html.parser")
    return soup.get_text()


def uri_to_markdown(a_uri: str) -> Dict[str, Any]:
    """
    Fetches content from a URI and converts HTML to markdown-style text

    Args:
        a_uri (str): URI to fetch and convert

    Returns:
        web page text converted converted markdown content
    """
    try:
        # Validate URI
        parsed = urlparse(a_uri)
        if not all([parsed.scheme, parsed.netloc]):
            return f"Invalid URI: {a_uri}"

        # Fetch content
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(a_uri, headers=headers, timeout=10)
        response.raise_for_status()

        # Parse HTML
        soup = BeautifulSoup(response.text, "html.parser")

        # Get title
        title = soup.title.string if soup.title else ""

        # Get text and clean up
        text = soup.get_text()

        # Clean up the text
        text = re.sub(r"\n\s*\n", "\n\n", text)  # Remove multiple blank lines
        text = re.sub(r" +", " ", text)  # Remove multiple spaces
        text = html.unescape(text)  # Convert HTML entities
        text = text.strip()

        return f"Contents of URI {a_uri} is:\n# {title}\n\n{text}\n"

    except requests.RequestException as e:
        return f"Network error: {str(e)}"

    except Exception as e:
        return f"Error processing URI: {str(e)}"


def search_web(query: str, max_results: int = 5) -> str:
    """
    Performs a web search and returns results
    Note: This is a placeholder. Implement with your preferred search API.

    Args:
        query (str): Search query
        max_results (int): Maximum number of results to return

    Returns:
        Dict[str, Any]: Dictionary containing:
            - 'results': List of search results
            - 'count': Number of results found
            - 'error': Error message if any, None otherwise
    """

    # Placeholder for search implementation
    return {
        "results": [],
        "count": 0,
        "error": "Web search not implemented. Please implement with your preferred search API.",
    }


def brave_search_summaries(
    query,
    num_results=3,
    url="https://api.search.brave.com/res/v1/web/search",
    api_key=api_key,
):
    headers = {"X-Subscription-Token": api_key, "Content-Type": "application/json"}
    params = {"q": query, "count": num_results}

    response = requests.get(url, headers=headers, params=params)
    ret = []

    if response.status_code == 200:
        search_results = response.json()
        ret = [
            {
                "title": result.get("title"),
                "url": result.get("url"),
                "description": replace_html_tags_with_text(result.get("description")),
            }
            for result in search_results.get("web", {}).get("results", [])
        ]
        logging.info("Successfully retrieved results.")
    else:
        try:
            error_info = response.json()
            logging.error(f"Error {response.status_code}: {error_info.get('message')}")
        except json.JSONDecodeError:
            logging.error(f"Error {response.status_code}: {response.text}")

    return ret


def brave_search_text(query, num_results=3):
    summaries = brave_search_summaries(query, num_results)
    ret = ""
    for s in summaries:
        url = s["url"]
        text = uri_to_markdown(url)
        summary = summarize_text(
            f"Given the query:\n\n{query}\n\nthen, summarize text removing all material that is not relevant to the query and then be very concise for a very short summary:\n\n{text}\n"
        )
        ret += ret + summary
    print("\n\n-----------------------------------")
    return ret

# Function metadata for Ollama integration
uri_to_markdown.metadata = {
    "name": "uri_to_markdown",
    "description": "Converts web page content to markdown-style text",
    "parameters": {"a_uri": "URI of the web page to convert"},
}

search_web.metadata = {
    "name": "search_web",
    "description": "Performs a web search and returns results",
    "parameters": {
        "query": "Search query",
        "max_results": "Maximum number of results to return",
    },
}

# Export the functions
__all__ = ["uri_to_markdown", "search_web"]

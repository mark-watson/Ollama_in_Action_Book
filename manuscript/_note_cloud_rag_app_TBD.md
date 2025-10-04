

## Advanced Application: The Search-Augmented Generation (RAG) Pattern

The true power of the Web Search API is realized when it is used not as a standalone tool, but as the first step in a Retrieval-Augmented Generation (RAG) pipeline. This pattern uses the search results to provide context to a language model, enabling it to generate an informed and grounded response.

This reveals a powerful symbiotic relationship between Ollama's two cloud services. The Web Search API can return thousands of tokens of text content from multiple sources. To make effective use of this rich context, a model with a very large context window is required. Stuffing this much information into a local model with a limited context window (e.g., 4,000 or 8,000 tokens) is often inefficient or impossible.   

This is precisely where the High-Performance Inference service becomes essential. The cloud models are explicitly designed for this type of workload, as they "run at the full context length," which can be orders of magnitude larger than what is feasible on local hardware. For example, a model like    

kimi-k2:1t-cloud is capable of handling enormous contexts, making it an ideal candidate for RAG applications. The Web Search API, therefore, creates a compelling technical reason to use the Cloud Inference service, demonstrating how the two services are designed to function as a single, powerful system.   

A conceptual blueprint for a simple RAG agent using Ollama Cloud would be:

Receive User Query: Start with the user's original question (e.g., "What were the key findings from the latest UN climate report?").

Perform Web Search: Call ollama.web_search() with the user's query to retrieve a list of relevant articles and snippets.

Construct Context: Format the search results into a single string of context. This could be a simple concatenation of the content snippets from the API response.

Augment Prompt: Create a new, detailed prompt for a language model. This prompt should include the formatted context and instruct the model to answer the original user query based solely on the provided information.

Generate Response: Send this augmented prompt to a large-context cloud model (e.g., gpt-oss:120b-cloud) using the client.chat() method.

Return Grounded Answer: The model's response will be grounded in the real-time information retrieved from the web, resulting in an accurate, up-to-date, and non-hallucinated answer.

TBD code.. TDB

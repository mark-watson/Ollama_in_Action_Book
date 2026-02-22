# Advanced RAG Using zvec Vector Datastore and Local Model

The **zvec** library implements a lightweight, lightning-fast, in-process vector database. Allibaba released **zvec** in February 2026. We will see how to use **zvec** and then build a high performance RAG system. We will use the tiny model **qwen3:1.7b** as part of the application.

Note: the source code for this example can be found in **Ollama_in_Action_Book/source-code/RAG_zvec**. Not all the code in this file is listed here.

## Introduction and Architecture

Building a Retrieval-Augmented Generation (RAG) pipeline entirely locally ensures absolute data privacy, eliminates API latency costs, and provides full control over the embedding and generation models. In this chapter, we construct a fully offline RAG system utilizing Ollama for both embeddings (embeddinggemma) and inference (qwen3:1.7b), paired with zvec, a lightweight, high-performance local vector database.

The architecture follows a classic two-phase RAG pattern:

- Ingestion: Parse local text files, chunk the content, generate embeddings via Ollama, and index them into zvec.
- Retrieval & Generation: Embed the user query, perform a similarity search in zvec, and inject the retrieved top-k chunks into the context window of a local Ollama chat model.

## Design Analysis: Dependency Minimization

A notable design choice in our implementation is the reliance on Python's standard library for network calls. By utilizing **urllib.request** instead of third-party libraries like **requests** or the official **ollama-python** client library, the dependency footprint is minimized exclusively to **zvec**. This reduces virtual environment overhead and potential version conflicts, prioritizing a lean deployment.

## Implementation Walkthrough

TBD

### Embedding and Chunking Strategy
The ingestion phase relies on a fixed-size overlapping window strategy.

```python
def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks
```

Analysis:

- Chunk Size (500 chars): This relatively small chunk size yields high-granularity embeddings. It reduces the risk of retrieving "diluted" context where a single chunk contains multiple disparate concepts.
- Overlap (50 chars): Crucial for preventing context loss at the boundaries of chunks. It ensures that a semantic concept bisected by a hard character limit is still captured cohesively in at least one chunk.
- Embedding Model: The system uses embeddinggemma. The Ollama API endpoint (/api/embeddings) is called directly. If the server fails to respond, a fallback zero-vector [0.0] * 768 is returned to prevent pipeline crashes, though logging or raising an exception might be preferred in production.

### Vector Storage with zvec
The **zvec** integration demonstrates a strictly typed, schema-driven approach to local vector storage.

```python
    schema = zvec.CollectionSchema(
        name="example",
        vectors=zvec.VectorSchema("embedding", zvec.DataType.VECTOR_FP32, 768),
        fields=zvec.FieldSchema("text", zvec.DataType.STRING),
    )
```

Analysis:

-- Dimensionality Matching: The vector schema is hardcoded to 768 dimensions (FP32), which strictly matches the output tensor of the embeddinggemma model. Any change to the embedding model in the configuration must be accompanied by a corresponding update to this schema.
-- Storage Path: The database is initialized locally at ./zvec_example. The implementation includes a defensive teardown (shutil.rmtree) of existing databases on startup. This is excellent for testing and iterative development, though destructive in a persistent production environment.

### Retrieval and LLM Synthesis

The synthesis phase bridges the vector database and the Generative LLM.

```python
def search(collection, query, topk=5):
    """Search the zvec collection for chunks relevant to the query."""
    query_vector = get_embedding(query)
    results = collection.query(
        zvec.VectorQuery("embedding", vector=query_vector),
        topk=topk,
    )
    chunks = []
    for res in results:
        text = res.fields.get("text", "") if res.fields else ""
        if text:
            chunks.append(text)
    return chunks
```

Analysis:

- Top-K Retrieval: The default topk=5 retrieves roughly 2,500 characters of context. This easily fits within the context window of modern small models like qwen3:1.7b without causing attention dilution ("lost in the middle" syndrome).
- System Prompt Engineering: The **ask_ollama** function utilizes strict prompt constraints: "Answer the user's question using ONLY the context provided below. If the context does not contain enough information, say so." This significantly mitigates hallucination by forcing the model to ground its response exclusively in the retrieved data.
- Stateless Execution: The /api/chat call sets "stream": False and does not maintain a conversation history array across loop iterations. This makes it a pure Q&A interface rather than a continuous chat, ensuring each answer is cleanly tied to a fresh zvec retrieval.

### Execution

To run the pipeline, ensure the Ollama daemon is running locally on port 11434 and that both models (embeddinggemma and qwen3:1.7b) have been pulled. Place your .txt files in the **../data** directory and execute the script. The system will build the index and immediately drop you into a REPL loop for interactive querying.

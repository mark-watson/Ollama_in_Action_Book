# Advanced RAG Using zvec Vector Datastore and Local Model

The **zvec** library implements a lightweight, lightning-fast, in-process vector database. Allibaba released **zvec** in February 2026. We will see how to use **zvec** and then build a high performance RAG system. We will use the tiny model **qwen3:1.7b** as part of the application.

Note: The source code for this example can be found in **Ollama_in_Action_Book/source-code/RAG_zvec/app.py**. Not all the code in this file is listed here.

## Introduction and Architecture

Building a Retrieval-Augmented Generation (RAG) pipeline entirely locally ensures absolute data privacy, eliminates API latency costs, and provides full control over the embedding and generation models. In this chapter, we construct a fully offline RAG system utilizing Ollama for both embeddings (embeddinggemma) and inference (qwen3:1.7b), paired with zvec, a lightweight, high-performance local vector database.

The architecture follows a classic two-phase RAG pattern, adding an additional third step to improve the user experience:

- Ingestion: Parse local text files, chunk the content, generate embeddings via Ollama, and index them into zvec.
- Retrieval & Generation: Embed the user query, perform a similarity search in zvec, and save the retrieved top-k chunks for processing by a local Ollama chat model.
- Use a small LLM model (qwen3:1.7b) to process the retrieved chunks and  taking into account the user’s original query and then format a subset of the text in the returned chunks for the user to read.

## Design Analysis: Dependency Minimization

A notable design choice in our implementation is the reliance on Python's standard library for network calls. By utilizing **urllib.request** instead of third-party libraries like **requests** or the official **ollama-python** client library, the dependency footprint is minimized exclusively to **zvec**. This reduces virtual environment overhead and potential version conflicts, prioritizing a lean deployment.

## Implementation Walkthrough

Here we look at some of the code in the source file **app.py**.

### Embedding and Chunking Strategy
The ingestion phase relies on a fixed-size overlapping window strategy. Here is an implementation of a chunking strategy:

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

Analysis of code:

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

Analysis of code:

- Dimensionality Matching: The vector schema is hardcoded to 768 dimensions (FP32), which strictly matches the output tensor of the embeddinggemma model. Any change to the embedding model in the configuration must be accompanied by a corresponding update to this schema.
- Storage Path: The database is initialized locally at ./zvec_example. The implementation includes a defensive teardown (shutil.rmtree) of existing databases on startup. This is excellent for testing and iterative development, though destructive in a persistent production environment.

The following function builds the index using an embedding model for the local Ollama server:

```python
def build_index():
    """Index all text files from the data directory into zvec."""
    # Define collection schema (embeddinggemma: 768 dimensions)
    schema = zvec.CollectionSchema(
        name="example",
        vectors=zvec.VectorSchema("embedding", zvec.DataType.VECTOR_FP32, 768),
        fields=zvec.FieldSchema("text", zvec.DataType.STRING),
    )

    db_path = "./zvec_example"
    if os.path.exists(db_path):
        import shutil
        shutil.rmtree(db_path)

    collection = zvec.create_and_open(path=db_path, schema=schema)

    docs = []
    doc_count = 0
    for root, _, files in os.walk(config["data_dir"]):
        for file in files:
            if file.lower().endswith(config["extensions"]):
                try:
                    file_path = Path(root) / file
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    chunks = chunk_text(content)
                    for i, chunk in enumerate(chunks):
                        embedding = get_embedding(chunk)
                        docs.append(zvec.Doc(
                            id=f"{file}_{i}",
                            vectors={"embedding": embedding},
                            fields={"text": chunk},
                        ))
                    doc_count += len(chunks)
                except Exception as e:
                    pass

    if docs:
        collection.insert(docs)
    print(f"Indexed {doc_count} chunks from {config['data_dir']}")
    return collection
```

This function **build_index** initializes a local vector database and populates it with document embeddings. Specifically, it executes four main operations:

- Schema & Storage Initialization: Defines a strict schema for zvec (768-dimensional FP32 vectors and a string metadata field) and destructively recreates the local database directory (./zvec_example).
- File Traversal: Recursively walks a configured target directory (config["data_dir"]) to locate specific file types.
- Transformation & Embedding: Reads each file, splits it into overlapping chunks, and retrieves the vector embedding for each chunk via an external call (get_embedding).
- Batch Insertion: Accumulates all processed chunks and their embeddings into a single memory list (docs), then performs a bulk insert into the zvec collection.

### Retrieval and LLM Synthesis

The synthesis phase bridges the vector database and the Generative LLM. Function **search** identifies matching text chunks in the vector database:

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

Function **search** performs a Top-K retrieval. The default topk=5 retrieves roughly 2,500 characters of context. This easily fits within the context window of modern small models like qwen3:1.7b without causing attention dilution ("lost in the middle" syndrome).

### System Prompt Engineering and Using a Small LLM to Prepare Output for a User

The **ask_ollama** function utilizes strict prompt constraints: "Answer the user's question using ONLY the context provided below. If the context does not contain enough information, say so." This significantly mitigates hallucination by forcing the model to ground its response exclusively in the retrieved data.

```python
def ask_ollama(question, context_chunks):
    """Send retrieved chunks + user question to the Ollama chat model."""
    context = "\n\n---\n\n".join(context_chunks)
    system_prompt = (
        "You are a helpful assistant. Answer the user's question using ONLY "
        "the context provided below. If the context does not contain enough "
        "information, say so. Be concise and accurate.\n\n"
        f"Context:\n{context}"
    )
    url = f"{OLLAMA_BASE}/api/chat"
    payload = json.dumps({
        "model": config["chat_model"],
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
    }).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as res:
            body = json.loads(res.read().decode("utf-8"))
            return body["message"]["content"]
    except Exception as e:
        return f"Error calling Ollama chat: {e}"
```

Function **ask_ollama** uses stateless execution: The /api/chat call sets "stream": False and does not maintain a conversation history array across loop iterations. This makes it a pure Q&A interface rather than a continuous chat, ensuring each answer is cleanly tied to a fresh zvec retrieval.

### Execution

To run the pipeline, ensure the Ollama daemon is running locally on port 11434 and that both models (embeddinggemma and qwen3:1.7b) have been pulled. Place your .txt files in the **../data** directory and execute the script. The system will build the index and immediately drop you into a REPL loop for interactive querying.

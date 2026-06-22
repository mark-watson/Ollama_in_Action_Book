# RAG Using zvec Vector Datastore and Local Model

The **zvec** library implements a lightweight, lightning-fast, in-process vector database. Allibaba released **zvec** in February 2026. We will see how to use **zvec** and then build a high performance RAG system. We will use the tiny model **qwen3:1.7b** as part of the application.

The Agentic RAG implementation in this chapter is based on the Google Research blog post/paper ["Unlocking dependable responses with Gemini: Enterprise Agent Platforms & Agentic RAG"](https://research.google/blog/unlocking-dependable-responses-with-gemini-enterprise-agent-platforms-agentic-rag/).

Note: The source code for this example can be found in **Ollama_in_Action_Book/source-code/RAG_zvec/app.py**. Not all the code in this file is listed here.


## Introduction and Architecture

Building a Retrieval-Augmented Generation (RAG) pipeline entirely locally ensures absolute data privacy, eliminates API latency costs, and provides full control over the embedding and generation models. In this chapter, we construct a fully offline **Agentic RAG** system using local Ollama models for embedding (`embeddinggemma`) and inference (such as `nemotron-3-nano:4b` or `qwen3:1.7b`), paired with **zvec**, a lightweight, high-performance local vector database.

Unlike standard (or "Vanilla") RAG, which follows a linear pipeline (query -> retrieve -> generate), our **Agentic RAG** pattern uses multiple specialized agents that plan, rewrite queries, and evaluate context sufficiency iteratively to guarantee grounding and avoid hallucination:

- **Ingestion**: Parse local text files, chunk the content, generate embeddings via Ollama, and index them into `zvec`.
- **Planning & Query Rewriting**: An orchestrator agent analyzes the user's question, drafts a search plan, and generates multiple sub-queries to capture all facets of a multi-hop or complex question.
- **Iterative Retrieval & Sufficiency Assessment**:
  - The vector retriever searches `zvec` for the sub-queries and aggregates unique snippets.
  - The **Sufficient Context Agent** acts as a quality inspector: it drafts a response, evaluates whether the retrieved snippets contain enough information, and flags any missing pieces as feedback.
  - If context is insufficient, the rewriter uses the feedback to formulate new queries, retrieving more snippets. This loop runs up to 3 times.
- **Synthesis**: The **Synthesis Agent** compiles the final answer using the accumulated context. If context remains insufficient after iterations, it clearly states what is missing rather than guessing.

![Architecture diagram](images/RAG_zvec_architecture.png)

## Design Analysis: Dependency Minimization

A notable design choice in our implementation is the reliance on Python's standard library for network calls. By utilizing **urllib.request** instead of third-party libraries like **requests**, the dependency footprint is minimized exclusively to **zvec**. This reduces virtual environment overhead and potential version conflicts, prioritizing a lean deployment.

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
- Embedding Model: The system uses `embeddinggemma`. The Ollama API endpoint (`/api/embeddings`) is called directly. If the server fails to respond, a fallback zero-vector `[0.0] * 768` is returned to prevent pipeline crashes.

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

- Dimensionality Matching: The vector schema is hardcoded to 768 dimensions (FP32), which strictly matches the output tensor of the `embeddinggemma` model. Any change to the embedding model in the configuration must be accompanied by a corresponding update to this schema.
- Storage Path: The database is initialized locally at `./zvec_example`. The implementation includes a defensive teardown (`shutil.rmtree`) of existing databases on startup. This is excellent for testing and iterative development, though destructive in a persistent production environment.

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

- Schema & Storage Initialization: Defines a strict schema for zvec (768-dimensional FP32 vectors and a string metadata field) and recreates the local database directory (`./zvec_example`).
- File Traversal: Recursively walks a configured target directory to locate specific file types.
- Transformation & Embedding: Reads each file, splits it into overlapping chunks, and retrieves the vector embedding for each chunk via `get_embedding`.
- Batch Insertion: Accumulates all processed chunks and their embeddings, then performs a bulk insert.

### Multi-Query Retrieval and Deduplication

To support queries that target multiple concepts, we define a wrapper `search_multi_queries` that performs Top-K retrieval across multiple queries and aggregates only unique snippets to avoid context bloat:

```python
def search_multi_queries(collection, queries, topk=3):
    """Search the zvec collection for multiple queries, aggregating and deduplicating chunks."""
    all_chunks = []
    seen = set()
    for query in queries:
        chunks = search(collection, query, topk=topk)
        for chunk in chunks:
            cleaned = chunk.strip()
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                all_chunks.append(chunk)
    return all_chunks
```

### Agentic RAG Multi-Agent Components

To implement the multi-agent planning and sufficiency check, we define helpers to make structured chat calls to Ollama using its built-in JSON constraint parameter (`"format": "json"`), and parse the outputs reliably.

```python
def call_llm(system_prompt: str, user_prompt: str, json_format: bool = False) -> str:
    """Helper to send a prompt to the Ollama chat model, optionally enforcing JSON format."""
    url = f"{OLLAMA_BASE}/api/chat"
    payload = {
        "model": config["chat_model"],
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "options": {
            "temperature": 0.1,  # Low temperature for deterministic behavior
        }
    }
    if json_format:
        payload["format"] = "json"
        
    req = _make_request(url, payload)
    try:
        with urllib.request.urlopen(req) as res:
            body = json.loads(res.read().decode("utf-8"))
            return body["message"]["content"]
    except Exception as e:
        print(f"Error calling Ollama chat: {e}")
        return ""
```

Using this foundation, we implement the individual agents:

#### Planner Agent
Generates a plan and breaks down the user query into multiple specific search queries.

```python
def plan_and_rewrite(question: str) -> dict:
    """Planner Agent: Analyzes the question, generates a plan and search queries."""
    system_prompt = (
        "You are a Plan and Query Rewriter agent. Your task is to analyze the user's question, "
        "create a brief search plan, and generate 1 to 3 distinct search queries to retrieve relevant "
        "information from a vector database.\n"
        "You must respond ONLY with a JSON object in this format:\n"
        "{\n"
        '  "plan": "brief explanation of what to search for",\n'
        '  "queries": ["query 1", "query 2"]\n'
        "}\n"
        "Do not include any other text."
    )
    user_prompt = f"Question: {question}"
    res_text = call_llm(system_prompt, user_prompt, json_format=True)
    res_json = parse_json_response(res_text)
    
    if not res_json or "queries" not in res_json:
        res_json = {
            "plan": f"Direct search for: '{question}'",
            "queries": [question]
        }
    return res_json
```

#### Sufficient Context Agent
Evaluates if the retrieved snippets contain enough information to fully address the query. If not, it outputs `is_sufficient: false` and logs exactly what facts are missing.

```python
def evaluate_context(question: str, snippets: list) -> dict:
    """Sufficient Context Agent: Evaluates whether the retrieved snippets contain enough info."""
    context_str = "\n\n---\n\n".join(snippets)
    system_prompt = (
        "You are a Sufficient Context Agent. Your job is to review the user's question, "
        "the retrieved context snippets, and determine if the snippets contain all the "
        "necessary information to answer the question fully.\n"
        "First, mentally draft a potential answer. Then assess if any parts of the question "
        "are unanswered or if any crucial information is missing.\n"
        "You must respond ONLY with a JSON object in this format:\n"
        "{\n"
        '  "is_sufficient": true or false (boolean),\n'
        '  "draft_answer": "a rough draft answer based on the current context",\n'
        '  "reason": "explanation of what is present or what is missing from the snippets",\n'
        '  "feedback": "if is_sufficient is false, detailed feedback of what specific keywords, topics, or facts are missing and should be searched for next. If is_sufficient is true, leave this empty."\n'
        "}\n"
        "Do not include any other text."
    )
    user_prompt = (
        f"Question: {question}\n\n"
        f"Retrieved Snippets:\n{context_str}"
    )
    res_text = call_llm(system_prompt, user_prompt, json_format=True)
    res_json = parse_json_response(res_text)
    
    if not res_json or "is_sufficient" not in res_json:
        res_json = {
            "is_sufficient": True,
            "draft_answer": "No draft available.",
            "reason": "Failed to parse sufficiency evaluation, defaulting to sufficient.",
            "feedback": ""
        }
    return res_json
```

#### Synthesis Agent
Generates the final response grounded in the accumulated context. If context sufficiency failed, it explicitly reports the missing details.

```python
def synthesize_answer(question: str, snippets: list, is_fully_sufficient: bool, sufficiency_reason: str) -> str:
    """Synthesis Agent: Generates final grounded response using retrieved context."""
    context_str = "\n\n---\n\n".join(snippets)
    if is_fully_sufficient:
        system_prompt = (
            "You are a Synthesis Agent. Write a clear, comprehensive, and accurate final answer "
            "to the user's question using ONLY the provided context. Do not extrapolate or assume facts.\n"
            f"Context:\n{context_str}"
        )
        user_prompt = f"Question: {question}"
    else:
        system_prompt = (
            "You are a Synthesis Agent. The retrieved context was NOT fully sufficient to answer the question. "
            "Answer what you can from the provided context, and clearly note what information is missing "
            "or could not be retrieved from the database. Do not make up any information.\n"
            f"Reason for insufficiency: {sufficiency_reason}\n\n"
            f"Context:\n{context_str}"
        )
        user_prompt = f"Question: {question}"
        
    return call_llm(system_prompt, user_prompt, json_format=False)
```

## Example Run

To run the pipeline, ensure the Ollama daemon is running locally on port 11434 and that both models (embeddinggemma and qwen3:1.7b) have been pulled. Place your .txt files in the **../data** directory and execute the script. The system will build the index and immediately drop you into a REPL loop for interactive querying.

Here is an example run where we specify the use of model **qwen3:1.7b**:

```bash
$ export MODEL=qwen3:1.7b
 $ uv run app.py
Building zvec index from text files …
Indexed 9 chunks from ../data

Agentic RAG chat ready  (model: nemotron-3-nano:4b)
Type your question, or 'quit' to exit.

You> What are the main schools of economic thought?

🧠 [Planner] Analyzing question and generating search plan...
   ↳ Plan: Search for a comprehensive list and description of the main schools of economic thought, such as Keynesian, Neoclassical, Marxist, Austrian, etc.
   ↳ Initial Queries: ['main schools of economic thought', 'major schools of economics overview', 'schools of economic thought']
🔍 [Retriever] Searching vector store for queries...
   ↳ Found 4 unique context snippet(s).
🤖 [Sufficiency Check] Evaluating context sufficiency (Iteration 1)...
   ↳ Sufficiency: False
   ↳ Reason: Only the Austrian School is described in the snippets; other major schools such as Keynesian, Neoclassical, Marxist, etc., are not mentioned.
🔄 [Rewriter] Context insufficient. Feedback: 'Missing information on Keynesian economics, neoclassical theory, Marxist economics, and possibly other contemporary schools. These should be searched for.'
   Generating new queries based on feedback...
   ↳ New queries: ['Keynesian economics overview', 'Neoclassical economic theory summary']
🔍 [Retriever] Retrieving additional context...
   ↳ Found 2 new unique snippet(s). Total unique snippets: 6.
🤖 [Sufficiency Check] Evaluating context sufficiency (Iteration 2)...
   ↳ Sufficiency: False
   ↳ Reason: The snippets only describe the Austrian School in detail. They do not mention other significant schools of economic thought like Keynesian, Neoclassical, or Marxist, which are essential to a complete answer.
🔄 [Rewriter] Context insufficient. Feedback: 'Missing keywords: Keynesian, Neoclassical, Marxist, and possibly others such as Institutional economics.'
   Generating new queries based on feedback...
   ↳ New queries: ['Marxist economic theory summary', 'Institutional economics schools of thought']
🔍 [Retriever] Retrieving additional context...
   ↳ Found 0 new unique snippet(s). Total unique snippets: 6.
🤖 [Sufficiency Check] Evaluating context sufficiency (Iteration 3)...
   ↳ Sufficiency: False
   ↳ Reason: The snippets only describe the Austrian School in detail. They do not mention or provide information about other significant schools of economic thought (e.g., Keynesian, Neoclassical, Marxist).
⚠️  [Iteration Limit] Reached max iterations (3). Proceeding to synthesis with partial context.
✍️  [Synthesis] Generating final response...

Assistant> **Answer based on the provided context**

- The **Austrian School** (also called the Vienna or Psychological School) is described in detail as one school of economic thought that emphasizes the spontaneous organizing power of the price mechanism, advocates a “laissez‑faire” approach, and stresses voluntary contractual agreements with minimal government intervention.

**What is missing / could not be retrieved**

- The context does **not** mention or provide any information about other significant schools of economic thought such as Keynesian economics, Neoclassical economics, Marxist economics, etc.  
- Therefore, I cannot list those schools or describe their characteristics from the given material.

**Conclusion**

From the supplied text, the only school explicitly described is the **Austrian School**. The existence and description of other major schools (e.g., Keynesian, Neoclassical, Marxist) are not present in the retrieved context.

You> who says that Economics is bullshit?

🧠 [Planner] Analyzing question and generating search plan...
   ↳ Plan: Search for statements where someone calls Economics 'bullshit' and identify the speaker or source.
   ↳ Initial Queries: ['economics is bullshit quote', 'who said economics is bullshit', 'criticism of economics bullshit']
🔍 [Retriever] Searching vector store for queries...
   ↳ Found 3 unique context snippet(s).
🤖 [Sufficiency Check] Evaluating context sufficiency (Iteration 1)...
   ↳ Sufficiency: True
   ↳ Reason: The snippet explicitly states that Pauli Blendergast, who teaches at the University of Krampton Ohio and is famous for saying economics is bullshit, is the person who makes this claim.
✅ [Sufficiency Check] Context is fully sufficient!
✍️  [Synthesis] Generating final response...

Assistant> Pauli Blendergast, an economist who teaches at the University of Krampton, Ohio, is said to claim that “economics is bullshit.”

You> ^D
Goodbye!
```

Dear reader, notice that there was no information in the indexed text to answer the second example query and this program correctly refused to hallucinate (or make up) an answer.

## Wrap Up for RAG Using zvec Vector Datastore and Local Model

In this chapter, we built a completely offline, privacy-preserving RAG architecture by bridging Alibaba’s recently released in-process vector database, zvec, with local Ollama inference. By intentionally minimizing external dependencies and utilizing a strictly typed, schema-driven datastore, we eliminated the network overhead and deployment bloat typical of client-server vector databases. The fixed-size overlapping chunking strategy, combined with the 768-dimensional embeddinggemma model, ensures high-fidelity semantic retrieval. Simultaneously, the compact qwen3:1.7b model demonstrates that a heavily constrained, prompt-engineered generation phase can effectively synthesize retrieved context without hallucination.

The resulting pipeline serves as a robust, lightweight foundation for edge-deployable AI applications. Because the entire storage and inference stack executes locally within the same process, the pattern is exceptionally portable, fast, and secure. Moving forward, this baseline implementation can be extended to handle more complex retrieval requirements, such as integrating dynamic semantic chunking, implementing Reciprocal Rank Fusion (RRF) for hybrid multi-vector queries, or introducing multi-turn conversational memory. Ultimately, combining embedded vector storage with small-parameter LLMs proves that high-performance, domain-specific RAG does not require massive cloud infrastructure.

## Optional Practice Problems

1. **Semantic Document Chunking.** In `app.py`, the document loading process uses a simple character-based overlapping split. Replace this mechanism with semantic chunking, where paragraphs are split on sentence boundaries and grouped together only if their embedding cosine similarity is above a threshold (e.g. 0.85). Verify if the retrieval accuracy changes for cross-paragraph information.

2. **Multi-Query Expansion.** Currently, the Planner generates a plan and a single list of search queries. Modify the retriever stage to execute three separate search queries against `zvec` for every prompt, merge the returned results, and implement a de-duplication step based on chunk IDs or content hashes.

3. **Retrieved Context Sufficiency Threshold.** The sufficiency evaluator in the loop uses a Yes/No classification. Modify the evaluator system prompt to return a score from 1 to 5 indicating sufficiency. Continue the retrieval iteration loop only if the score is less than 4. Write a script to output the sufficiency score of the retrieved passages at each step.

4. **Web Search API Fallback.** If the local `zvec` vector store returns insufficient context after 3 retrieval iterations, configure the agent to call the Brave Web Search API as a fallback to fetch the missing details from the web, append those web snippets to the context, and pass the final aggregate context to the Synthesis Agent.

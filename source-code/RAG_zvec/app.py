import os
import json
import urllib.request
from pathlib import Path
import zvec

# Configuration
config = {
    "data_dir": os.getenv("DATA_DIR", "../data"),
    "extensions": (".txt",),
    "embedding_model": os.getenv("EMBEDDING_MODEL", "embeddinggemma"),
    "chat_model": os.getenv("CHAT_MODEL", "qwen3:1.7b"),
}

OLLAMA_BASE = "http://localhost:11434"


def get_embedding(text):
    """Get embedding from local Ollama instance."""
    url = f"{OLLAMA_BASE}/api/embeddings"
    data = json.dumps({
        "model": config["embedding_model"],
        "prompt": text,
    }).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as res:
            return json.loads(res.read().decode("utf-8"))["embedding"]
    except Exception as e:
        print(f"Error calling Ollama embeddings: {e}")
        return [0.0] * 768


def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


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


def main():
    print("Building zvec index from text files …")
    collection = build_index()
    print(f"\nRAG chat ready  (model: {config['chat_model']})")
    print("Type your question, or 'quit' to exit.\n")

    while True:
        try:
            question = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not question or question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        chunks = search(collection, question)
        if not chunks:
            print("No relevant chunks found in the index.\n")
            continue

        answer = ask_ollama(question, chunks)
        print(f"\nAssistant> {answer}\n")


if __name__ == "__main__":
    main()

import os
import json
import urllib.request
import re
from pathlib import Path
import sys
import zvec

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ollama_config import get_model

# Configuration
config = {
    "data_dir": os.getenv("DATA_DIR", "../data"),
    "extensions": (".txt",),
    "embedding_model": os.getenv("EMBEDDING_MODEL", "embeddinggemma"),
    "chat_model": os.getenv("CHAT_MODEL", get_model()),
}

if os.environ.get("CLOUD"):
    OLLAMA_BASE = "https://ollama.com"
    OLLAMA_HEADERS = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + os.environ.get("OLLAMA_API_KEY", ""),
    }
else:
    OLLAMA_BASE = "http://localhost:11434"
    OLLAMA_HEADERS = {"Content-Type": "application/json"}


def _make_request(url, data):
    """Helper to make an HTTP POST with appropriate headers."""
    encoded = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url, data=encoded, headers=OLLAMA_HEADERS)
    return req


def get_embedding(text):
    """Get embedding from Ollama (local or cloud)."""
    url = f"{OLLAMA_BASE}/api/embeddings"
    data = {
        "model": config["embedding_model"],
        "prompt": text,
    }
    req = _make_request(url, data)
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


# ==========================================
# Agentic RAG Multi-Agent Components
# ==========================================

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


def parse_json_response(response_text: str) -> dict:
    """Extract and parse JSON from the LLM response text."""
    if not response_text:
        return {}
    # Find anything between the first '{' and the last '}'
    match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass
    return {}


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


def rewrite_with_feedback(question: str, previous_queries: list, feedback: str) -> dict:
    """Query Rewriter Agent (Feedback Loop): Generates new search queries based on missing info feedback."""
    system_prompt = (
        "You are a Query Rewriter agent. Your task is to generate 1 to 2 new search queries "
        "to retrieve the missing information described in the feedback, based on the original question.\n"
        "You must respond ONLY with a JSON object in this format:\n"
        "{\n"
        '  "queries": ["new query 1", "new query 2"]\n'
        "}\n"
        "Do not include any other text."
    )
    user_prompt = (
        f"Original Question: {question}\n"
        f"Previous Queries: {json.dumps(previous_queries)}\n"
        f"Feedback on missing information: {feedback}"
    )
    res_text = call_llm(system_prompt, user_prompt, json_format=True)
    res_json = parse_json_response(res_text)
    
    if not res_json or "queries" not in res_json:
        res_json = {"queries": [feedback]}
    return res_json


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


def run_agentic_rag(collection, question: str) -> str:
    """Orchestrates the Agentic RAG multi-step/multi-agent loop."""
    print(f"\n🧠 \033[94m[Planner]\033[0m Analyzing question and generating search plan...")
    plan_res = plan_and_rewrite(question)
    print(f"   ↳ Plan: {plan_res.get('plan')}")
    print(f"   ↳ Initial Queries: {plan_res.get('queries')}")

    queries = plan_res.get('queries', [question])
    print(f"🔍 \033[92m[Retriever]\033[0m Searching vector store for queries...")
    context_snippets = search_multi_queries(collection, queries, topk=3)
    print(f"   ↳ Found {len(context_snippets)} unique context snippet(s).")

    iteration = 0
    max_iterations = 3
    all_queries = list(queries)
    is_sufficient = True
    reason = ""

    while iteration < max_iterations:
        if not context_snippets:
            print("⚠️  \033[91m[Warning]\033[0m No context found. Breaking loop.")
            is_sufficient = False
            reason = "No context retrieved from database."
            break
            
        print(f"🤖 \033[95m[Sufficiency Check]\033[0m Evaluating context sufficiency (Iteration {iteration + 1})...")
        eval_res = evaluate_context(question, context_snippets)
        
        is_sufficient = eval_res.get("is_sufficient", True)
        reason = eval_res.get("reason", "No reason provided.")
        feedback = eval_res.get("feedback", "")
        
        print(f"   ↳ Sufficiency: {is_sufficient}")
        print(f"   ↳ Reason: {reason}")
        
        if is_sufficient:
            print("✅ \033[92m[Sufficiency Check]\033[0m Context is fully sufficient!")
            break
            
        iteration += 1
        if iteration >= max_iterations:
            print(f"⚠️  \033[91m[Iteration Limit]\033[0m Reached max iterations ({max_iterations}). Proceeding to synthesis with partial context.")
            break
            
        print(f"🔄 \033[93m[Rewriter]\033[0m Context insufficient. Feedback: '{feedback}'")
        print(f"   Generating new queries based on feedback...")
        rewrite_res = rewrite_with_feedback(question, all_queries, feedback)
        new_queries = rewrite_res.get("queries", [])
        print(f"   ↳ New queries: {new_queries}")
        
        all_queries.extend(new_queries)
        
        print(f"🔍 \033[92m[Retriever]\033[0m Retrieving additional context...")
        new_snippets = search_multi_queries(collection, new_queries, topk=3)
        
        added_count = 0
        for s in new_snippets:
            if s not in context_snippets:
                context_snippets.append(s)
                added_count += 1
        print(f"   ↳ Found {added_count} new unique snippet(s). Total unique snippets: {len(context_snippets)}.")

    print(f"✍️  \033[96m[Synthesis]\033[0m Generating final response...")
    answer = synthesize_answer(question, context_snippets, is_sufficient, reason)
    return answer


def ask_ollama(question, context_chunks):
    """Legacy helper function to send retrieved chunks + user question to the Ollama chat model."""
    context = "\n\n---\n\n".join(context_chunks)
    system_prompt = (
        "You are a helpful assistant. Answer the user's question using ONLY "
        "the context provided below. If the context does not contain enough "
        "information, say so. Be concise and accurate.\n\n"
        f"Context:\n{context}"
    )
    url = f"{OLLAMA_BASE}/api/chat"
    payload = {
        "model": config["chat_model"],
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
    }
    req = _make_request(url, payload)
    try:
        with urllib.request.urlopen(req) as res:
            body = json.loads(res.read().decode("utf-8"))
            return body["message"]["content"]
    except Exception as e:
        return f"Error calling Ollama chat: {e}"


def main():
    print("Building zvec index from text files …")
    collection = build_index()
    print(f"\nAgentic RAG chat ready  (model: {config['chat_model']})")
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

        answer = run_agentic_rag(collection, question)
        print(f"\nAssistant> {answer}\n")


if __name__ == "__main__":
    main()

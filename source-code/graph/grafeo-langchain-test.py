from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_experimental.graph_transformers import LLMGraphTransformer
from grafeo_langchain import GrafeoGraphStore, GrafeoGraphVectorStore

# 1. Initialize Ollama Models (Local LLM & Embeddings)
embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama(model="nemotron-3-nano:4b", temperature=0)

# ---------------------------------------------------------
# Part A: Vector Operations (GrafeoGraphVectorStore)
# ---------------------------------------------------------
vector_store = GrafeoGraphVectorStore(
    embedding=embeddings,
    db_path="./local_knowledge.db",
    embedding_dimensions=768,
)

docs = [
    Document(
        page_content="Grafeo is a fast, embedded graph database written in Rust.",
        metadata={"source": "grafeo_docs"}
    ),
    Document(
        page_content="Ollama allows users to run large language models locally on their own hardware without sending data to the cloud.",
        metadata={"source": "ollama_docs"}
    ),
    Document(
        page_content="LangChain is a framework for building applications powered by language models, providing tools for chaining prompts, memory, and retrieval.",
        metadata={"source": "langchain_docs"}
    ),
]

# Embed and store documents
vector_store.add_documents(docs)

# Perform local vector searches
queries = [
    "What language is Grafeo built with?",
    "How can I run LLMs without an internet connection?",
    "What tools does LangChain provide for retrieval?",
]
for query in queries:
    results = vector_store.similarity_search(query, k=1)
    print(f"Query : {query}")
    print(f"Result: {results[0].page_content}")
    print()

# ---------------------------------------------------------
# Part B: Graph Operations (GrafeoGraphStore)
# ---------------------------------------------------------
graph_store = GrafeoGraphStore(db_path="./local_knowledge.db")

# Use Ollama to extract nodes and relationships from text
graph_transformer = LLMGraphTransformer(llm=llm)
graph_docs = graph_transformer.convert_to_graph_documents(docs)

# Store the extracted triples in Grafeo
graph_store.add_graph_documents(graph_docs)

# Refresh schema (get_schema and get_structured_schema are properties, not methods)
graph_store.refresh_schema()
print("Graph Schema:")
print(graph_store.get_schema)
print()

# ---------------------------------------------------------
# Part C: LLM-powered graph query
# ---------------------------------------------------------
# Ask the LLM to generate a GQL query from a natural-language question,
# then execute it against the graph and summarise the results.

graph_question = "What entities are related to Rust?"

prompt = ChatPromptTemplate.from_template(
    "You are a graph database expert. Given the schema below, write a single GQL "
    "MATCH query to answer the question.\n\n"
    "Rules:\n"
    "- Relationships MUST use arrow syntax: (a)-[:REL_TYPE]->(b)\n"
    "- Never write relationships as bare words between nodes\n"
    "- Match on node properties with: (n {{node_id: 'value'}})\n"
    "- Example: MATCH (a {{node_id: 'Rust'}})-[r]-(b) RETURN a, r, b\n"
    "- Output ONLY the raw GQL query, no explanation, no markdown.\n\n"
    "Schema:\n{schema}\n\n"
    "Question: {question}\n\n"
    "GQL query:"
)

chain = prompt | llm
gql_response = chain.invoke({
    "schema": graph_store.get_schema,
    "question": graph_question,
})
gql_query = gql_response.content.strip()
print(f"LLM-generated GQL query: {gql_query}")

try:
    rows = graph_store.query(gql_query)
    print(f"Graph query results: {rows}")
except Exception as e:
    print(f"Query error: {e}")
print()

vector_store.close()
graph_store.close()

import kuzu
from langchain_community.chains.graph_qa.kuzu import KuzuQAChain
from langchain_community.graphs import KuzuGraph
from langchain_ollama.llms import OllamaLLM
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ollama_config import get_client, get_model
import shutil

db_path = "test_db"
if Path(db_path).exists():
    shutil.rmtree(db_path)

db = kuzu.Database(db_path)
conn = kuzu.Connection(db)

# Create two tables and a relation: Movie, Person, ActedIn
conn.execute("CREATE NODE TABLE Movie (name STRING, PRIMARY KEY(name))")
conn.execute(
    "CREATE NODE TABLE Person (name STRING, birthDate STRING, PRIMARY KEY(name))"
)
conn.execute("CREATE REL TABLE ActedIn (FROM Person TO Movie)")
conn.execute("CREATE (:Person {name: 'Al Pacino', birthDate: '1940-04-25'})")
conn.execute("CREATE (:Person {name: 'Robert De Niro', birthDate: '1943-08-17'})")
conn.execute("CREATE (:Movie {name: 'The Godfather'})")
conn.execute("CREATE (:Movie {name: 'The Godfather: Part II'})")
conn.execute(
    "CREATE (:Movie {name: 'The Godfather Coda: The Death of Michael Corleone'})"
)
conn.execute(
    "MATCH (p:Person), (m:Movie) WHERE p.name = 'Al Pacino' AND m.name = 'The Godfather' CREATE (p)-[:ActedIn]->(m)"
)
conn.execute(
    "MATCH (p:Person), (m:Movie) WHERE p.name = 'Al Pacino' AND m.name = 'The Godfather: Part II' CREATE (p)-[:ActedIn]->(m)"
)
conn.execute(
    "MATCH (p:Person), (m:Movie) WHERE p.name = 'Al Pacino' AND m.name = 'The Godfather Coda: The Death of Michael Corleone' CREATE (p)-[:ActedIn]->(m)"
)
conn.execute(
    "MATCH (p:Person), (m:Movie) WHERE p.name = 'Robert De Niro' AND m.name = 'The Godfather: Part II' CREATE (p)-[:ActedIn]->(m)"
)

conn.execute("CREATE (:Person {name: 'Marlon Brando', birthDate: '1924-04-03'})")
conn.execute("CREATE (:Person {name: 'Diane Keaton', birthDate: '1946-01-05'})")
conn.execute("CREATE (:Movie {name: 'Apocalypse Now'})")
conn.execute("CREATE (:Movie {name: 'Annie Hall'})")

conn.execute(
    "MATCH (p:Person), (m:Movie) WHERE p.name = 'Marlon Brando' AND m.name = 'Apocalypse Now' CREATE (p)-[:ActedIn]->(m)"
)
conn.execute(
    "MATCH (p:Person), (m:Movie) WHERE p.name = 'Diane Keaton' AND m.name = 'Annie Hall' CREATE (p)-[:ActedIn]->(m)"
)
conn.execute(
    "MATCH (p:Person), (m:Movie) WHERE p.name = 'Diane Keaton' AND m.name = 'The Godfather: Part II' CREATE (p)-[:ActedIn]->(m)"
)
conn.execute(
    "MATCH (p:Person), (m:Movie) WHERE p.name = 'Al Pacino' AND m.name = 'Apocalypse Now' CREATE (p)-[:ActedIn]->(m)"
)

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

graph = KuzuGraph(db, allow_dangerous_requests=True)

# Use ChatOllama for better instruction following
base_llm = ChatOllama(
    model=get_model(),
    temperature=0,
)

# Custom prompt using ChatPromptTemplate
CYPHER_GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a Kùzu Cypher expert. Generate ONLY the Cypher query.
Rules:
1. Do NOT use 'GROUP BY' or 'HAVING'. Cypher performs grouping implicitly.
2. For filtering on aggregations, use 'WITH' and 'WHERE'.
3. Start directly with MATCH or other Cypher keywords.
4. No preamble, no explanation, no markdown."""),
    ("human", """Schema:
{schema}

Example:
Question: Which actors appeared in more than one movie?
Cypher: MATCH (p:Person)-[:ActedIn]->(m:Movie) WITH p.name AS name, COUNT(m) AS count WHERE count > 1 RETURN name

Question: {question}
Cypher Query:""")
])

def clean_cypher_output(output):
    """Clean the LLM output to extract only the Cypher query."""
    content = output.content if hasattr(output, 'content') else str(output)
    
    # Remove markdown code blocks if present
    content = content.replace("```cypher", "").replace("```", "").strip()
    
    # Heuristic: Find the first Cypher keyword and strip everything before it
    keywords = ["MATCH", "CREATE", "MERGE", "RETURN", "WITH", "UNWIND"]
    for keyword in keywords:
        if keyword in content.upper():
            idx = content.upper().find(keyword)
            content = content[idx:]
            break
            
    # Take only the first line to avoid any trailing explanations
    return content.split("\n")[0].strip()

# Create a runnable that cleans the output
cypher_llm = base_llm | RunnableLambda(clean_cypher_output)

# Create a chain
# We use the cleaned LLM for Cypher generation and the base LLM for answering
chain = KuzuQAChain.from_llm(
    llm=cypher_llm,
    qa_llm=base_llm,
    graph=graph,
    verbose=True,
    allow_dangerous_requests=True,
    cypher_prompt=CYPHER_GENERATION_PROMPT,
)

print(graph.get_schema)

# Ask two questions
chain.invoke("Who acted in The Godfather: Part II?")
chain.invoke("Robert De Niro played in which movies?")
chain.invoke("Which actors acted in Apocalypse Now?")
chain.invoke("What movies did Diane Keaton act in?")
chain.invoke("Which actors appeared in more than one movie in the database?")

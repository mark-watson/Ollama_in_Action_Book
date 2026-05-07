import ladybug
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_community.graphs.graph_store import GraphStore
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

db = ladybug.Database(db_path)
conn = ladybug.Connection(db)


# ---------------------------------------------------------------------------
# Custom LadybugGraph wrapper for LangChain's GraphCypherQAChain
# ---------------------------------------------------------------------------
class LadybugGraph(GraphStore):
    """Minimal LangChain graph wrapper around a LadybugDB connection."""

    def __init__(self, database, allow_dangerous_requests: bool = False):
        self._db = database
        self._conn = ladybug.Connection(database)
        self._allow_dangerous_requests = allow_dangerous_requests
        self._schema = ""
        self.refresh_schema()

    @property
    def get_schema(self) -> str:
        return self._schema

    @property
    def get_structured_schema(self) -> dict:
        return {"schema": self._schema}

    def refresh_schema(self) -> None:
        """Build a human-readable schema string from the database metadata."""
        node_tables = []
        rel_tables = []
        try:
            result = self._conn.execute("CALL show_tables() RETURN *;")
            while result.has_next():
                row = result.get_next()
                table_name = row[1] if len(row) > 1 else row[0]
                table_type = row[2] if len(row) > 2 else ""
                if table_type == "NODE":
                    node_tables.append(table_name)
                elif table_type == "REL":
                    rel_tables.append(table_name)
        except Exception:
            pass

        parts = []
        for nt in node_tables:
            try:
                props_result = self._conn.execute(
                    f"CALL table_info('{nt}') RETURN *;"
                )
                props = []
                while props_result.has_next():
                    prow = props_result.get_next()
                    props.append(f"{prow[1]}: {prow[2]}")
                parts.append(f"Node: {nt} ({', '.join(props)})")
            except Exception:
                parts.append(f"Node: {nt}")

        for rt in rel_tables:
            parts.append(f"Relationship: {rt}")

        self._schema = "\n".join(parts) if parts else "No schema available."

    def query(self, query: str, params: dict = None) -> list[dict]:
        """Execute a Cypher query and return results as list of dicts."""
        try:
            result = self._conn.execute(query)
            columns = result.get_column_names()
            rows = []
            while result.has_next():
                values = result.get_next()
                rows.append(dict(zip(columns, values)))
            return rows
        except Exception as e:
            return [{"error": str(e)}]

    def add_graph_documents(self, graph_documents, include_source=False):
        raise NotImplementedError("Use Cypher queries to add data.")


# ---------------------------------------------------------------------------
# Build sample graph
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# LangChain QA chain using custom LadybugGraph wrapper
# ---------------------------------------------------------------------------
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

graph = LadybugGraph(db, allow_dangerous_requests=True)

# Use ChatOllama for better instruction following
base_llm = ChatOllama(
    model=get_model(),
    temperature=0,
)

# Custom prompt using ChatPromptTemplate
CYPHER_GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a Cypher expert for a property graph database. Generate ONLY the Cypher query.
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

# Create a chain using GraphCypherQAChain with our custom LadybugGraph wrapper
chain = GraphCypherQAChain.from_llm(
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

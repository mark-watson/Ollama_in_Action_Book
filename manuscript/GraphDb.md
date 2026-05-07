# Using Property Graph Database with Ollama

I have a long history of working with Knowledge Graphs (at Google and OliveAI) and I usually use RDF graph databases and the SPARQL query language. I have recently developed a preference for property graph databases because recent research has shown that using LLMs with RDF-based graphs have LLM context size issues due to large schemas, overlapping relations, and complex identifiers that exceed LLM context windows. Property graph databases like Neo4J and LadybugDB (which we use in this chapter) have more concise schemas. LadybugDB is the community-maintained successor to the Kùzu database, which was acquired by Apple and is no longer actively developed.

It is true that Google and other players are teasing 'infinite context' LLMs but since this book is about running smaller models locally I have chosen to only show a property graph example.

## Overview of Property Graphs

Property graphs represent a powerful and flexible data modeling paradigm that has gained significant traction in modern database systems and applications. At its core, a property graph is a directed graph structure where both vertices (nodes) and edges (relationships) can contain properties in the form of key-value pairs, providing rich contextual information about entities and their connections. Unlike traditional relational databases that rely on rigid table structures, property graphs offer a more natural way to represent highly connected data while maintaining the semantic meaning of relationships. This modeling approach is particularly valuable when dealing with complex networks of information where the relationships between entities are just as important as the entities themselves.
The distinguishing characteristics of property graphs make them especially well-suited for handling real-world data scenarios where relationships are multi-faceted and dynamic. Each node in a property graph can be labeled with one or more types (such as Person, Product, or Location) and can hold any number of properties that describe its attributes. Similarly, edges can be typed (like "KNOWS", "PURCHASED", or "LOCATED_IN") and augmented with properties that qualify the relationship, such as timestamps, weights, or quality scores. This flexibility allows for sophisticated querying and analysis of data patterns that would be cumbersome or impossible to represent in traditional relational schemas. The property graph model has proven particularly valuable in domains such as social network analysis, recommendation systems, fraud detection, and knowledge graphs, where understanding the intricate web of relationships between entities is crucial for deriving meaningful insights.

## Example Using Ollama, LangChain, and the LadybugDB Property Graph Database

The example shown here uses a custom LangChain wrapper around LadybugDB with `GraphCypherQAChain` to answer natural-language questions about a movie/actor graph. Since LadybugDB is a new project, we create a lightweight `LadybugGraph` adapter class that provides the schema and query interface that LangChain expects. Here is the file **graph_ladybug_property_example.py**:

```python
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
```

This code demonstrates the implementation of a graph database using LadybugDB, integrated with LangChain for question-answering capabilities. The code first defines a custom `LadybugGraph` wrapper class that provides the schema introspection and query execution interface that LangChain's `GraphCypherQAChain` expects. It then initializes a database connection and establishes a schema with two node types (Movie and Person) and a relationship type (ActedIn), creating a graph structure suitable for representing actors and their film appearances.

The implementation populates the database with specific data about "The Godfather" trilogy and several prominent actors (Al Pacino, Robert De Niro, Marlon Brando, and Diane Keaton). It uses Cypher query syntax to create nodes for both movies and actors, then establishes relationships between them using the ActedIn relationship type. The data model represents a typical many-to-many relationship between actors and movies.

This example then sets up a question-answering chain using LangChain, which combines the LadybugDB graph database with the Ollama language model. This chain enables natural language queries against the graph database, allowing users to ask questions about actor-movie relationships and receive responses based on the stored graph data. The implementation includes several example queries to demonstrate the system's functionality.


![Arcitecture diagram](images/graph_architecture.png)

Here is the output from this example:

```bash
$ rm -rf test_db 
$ uv run graph_ladybug_property_example.py
Node: Movie (name: STRING)
Node: Person (name: STRING, birthDate: STRING)
Relationship: ActedIn

> Entering new GraphCypherQAChain chain...
Generated Cypher:
MATCH (p:Person)-[:ActedIn]->(m:Movie) WHERE m.name = 'The Godfather: Part II' RETURN p.name
Full Context:
[{'p.name': 'Al Pacino'}, {'p.name': 'Robert De Niro'}, {'p.name': 'Diane Keaton'}]

> Finished chain.

> Entering new GraphCypherQAChain chain...
Generated Cypher:
MATCH (p:Person {name: 'Robert De Niro'})-[:ActedIn]->(m:Movie) RETURN m.name AS movieName
Full Context:
[{'movieName': 'The Godfather: Part II'}]

> Finished chain.

> Entering new GraphCypherQAChain chain...
Generated Cypher:
MATCH (p:Person)-[:ActedIn]->(m:Movie) WHERE m.name = 'Apocalypse Now' RETURN p.name AS actor
Full Context:
[{'actor': 'Al Pacino'}, {'actor': 'Marlon Brando'}]

> Finished chain.

> Entering new GraphCypherQAChain chain...
Generated Cypher:
MATCH (p:Person {name: 'Diane Keaton'})-[:ActedIn]->(m:Movie) RETURN m.name AS movieName;
Full Context:
[{'movieName': 'The Godfather: Part II'}, {'movieName': 'Annie Hall'}]

> Finished chain.

> Entering new GraphCypherQAChain chain...
Generated Cypher:
MATCH (p:Person)-[:ActedIn]->(m:Movie) WITH p.name AS name, COUNT(m) AS count WHERE count > 1 RETURN name
Full Context:
[{'name': 'Diane Keaton'}, {'name': 'Al Pacino'}]

> Finished chain.
```

The Cypher query language is commonly used in property graph databases. Here is a sample query:

```cypher
MATCH (p:Person)-[:ActedIn]->(m:Movie {name: 'The Godfather: Part II'})
RETURN p.name
```

This Cypher query performs a graph pattern matching operation to find actors who appeared in "The Godfather: Part II". Let's break it down:

- MATCH initiates a pattern matching operation
- (p:Person) looks for nodes labeled as "Person" and assigns them to variable p
- -[:ActedIn]-> searches for "ActedIn" relationships pointing outward
- (m:Movie {name: 'The Godfather: Part II'}) matches Movie nodes specifically with the name property equal to "The Godfather: Part II"
- RETURN p.name returns only the name property of the matched Person nodes

Based on the previous code's data, this query would return "Al Pacino" and "Robert De Niro" since they both acted in that specific film.

## Using LLMs to Create Graph Databases from Text Data

Using LadybugDB with local LLMs is simple to implement as seen in the last section. If you use large property graph databases hosted with LadybugDB or Neo4J, then the example in the last section is hopefully sufficient to get you started implementing natural language interfaces to property graph databases.

Now we will do something very different: use LLMs to generate data for property graphs, that is, to convert text to Python code to create a LadybugDB property graph database.

Specifically, we use the approach:

- Use the last example file **graph_ladybug_property_example.py** as an example for Claude Sonnet 3.5 to understand the LadybugDB Python APIs.
- Have Claude Sonnet 3.5 read the file **data/economics.txt** and create a schema for a new graph database and populate the schema from the contents of the file **data/economics.txt**.
- Ask Claude Sonnet 3.5 to also generate query examples.

Except for my adding the utility function **query_and_print_result**, this code was generated by Claude Sonnet 3.5:

```python
"""
Created by Claude Sonnet 3.5 from prompt:

Given some text, I want you to define Property graph schemas for
the information in the text. As context, here is some Python code
for defining two tables and a relation and querying the data:

[[CODE FROM graph_ladybug_property_example.py]]

NOW, HERE IS THE TEST TO CREATE SCHEME FOR, and to write code to
create nodes and links conforming to the scheme:

[[CONTENTS FROM FILE data/economics.txt]]

"""

import ladybug

db = ladybug.Database("economics_db")
conn = ladybug.Connection(db)

# Node tables
conn.execute("""
CREATE NODE TABLE School (
    name STRING,
    description STRING,
    PRIMARY KEY(name)
)""")

conn.execute("""
CREATE NODE TABLE Economist (
    name STRING,
    birthDate STRING,
    PRIMARY KEY(name)
)""")

conn.execute("""
CREATE NODE TABLE Institution (
    name STRING,
    type STRING,
    PRIMARY KEY(name)
)""")

conn.execute("""
CREATE NODE TABLE EconomicConcept (
    name STRING,
    description STRING,
    PRIMARY KEY(name)
)""")

# Relationship tables
conn.execute("CREATE REL TABLE FoundedBy (FROM School TO Economist)")
conn.execute("CREATE REL TABLE TeachesAt (FROM Economist TO Institution)")
conn.execute("CREATE REL TABLE Studies (FROM School TO EconomicConcept)")

# Insert some data
conn.execute("CREATE (:School {name: 'Austrian School', description: 'School of economic thought emphasizing spontaneous organizing power of price mechanism'})")

# Create economists
conn.execute("CREATE (:Economist {name: 'Carl Menger', birthDate: 'Unknown'})")
conn.execute("CREATE (:Economist {name: 'Eugen von Böhm-Bawerk', birthDate: 'Unknown'})")
conn.execute("CREATE (:Economist {name: 'Ludwig von Mises', birthDate: 'Unknown'})")
conn.execute("CREATE (:Economist {name: 'Pauli Blendergast', birthDate: 'Unknown'})")

# Create institutions
conn.execute("CREATE (:Institution {name: 'University of Krampton Ohio', type: 'University'})")

# Create economic concepts
conn.execute("CREATE (:EconomicConcept {name: 'Microeconomics', description: 'Study of individual agents and markets'})")
conn.execute("CREATE (:EconomicConcept {name: 'Macroeconomics', description: 'Study of entire economy and issues affecting it'})")

# Create relationships
conn.execute("""
MATCH (s:School), (e:Economist) 
WHERE s.name = 'Austrian School' AND e.name = 'Carl Menger' 
CREATE (s)-[:FoundedBy]->(e)
""")

conn.execute("""
MATCH (s:School), (e:Economist) 
WHERE s.name = 'Austrian School' AND e.name = 'Eugen von Böhm-Bawerk' 
CREATE (s)-[:FoundedBy]->(e)
""")

conn.execute("""
MATCH (s:School), (e:Economist) 
WHERE s.name = 'Austrian School' AND e.name = 'Ludwig von Mises' 
CREATE (s)-[:FoundedBy]->(e)
""")

conn.execute("""
MATCH (e:Economist), (i:Institution) 
WHERE e.name = 'Pauli Blendergast' AND i.name = 'University of Krampton Ohio' 
CREATE (e)-[:TeachesAt]->(i)
""")

# Link school to concepts it studies
conn.execute("""
MATCH (s:School), (c:EconomicConcept) 
WHERE s.name = 'Austrian School' AND c.name = 'Microeconomics' 
CREATE (s)-[:Studies]->(c)
""")

"""
Code written from the prompt:

Now that you have written code to create a sample graph database about
economics, you can write queries to extract information from the database.
"""

def query_and_print_result(query):
    """Basic pretty printer for Ladybug query results"""
    print(f"\n* Processing: {query}")
    result = conn.execute(query)
    if not result:
        print("No results found")
        return

    # Get column names
    while result.has_next():
        r = result.get_next()
        print(r)

# 1. Find all founders of the Austrian School
query_and_print_result("""
MATCH (s:School)-[:FoundedBy]->(e:Economist)
WHERE s.name = 'Austrian School'
RETURN e.name
""")

# 2. Find where Pauli Blendergast teaches
query_and_print_result("""
MATCH (e:Economist)-[:TeachesAt]->(i:Institution)
WHERE e.name = 'Pauli Blendergast'
RETURN i.name, i.type
""")

# 3. Find all economic concepts studied by the Austrian School
query_and_print_result("""
MATCH (s:School)-[:Studies]->(c:EconomicConcept)
WHERE s.name = 'Austrian School'
RETURN c.name, c.description
""")

# 4. Find all economists and their institutions
query_and_print_result("""
MATCH (e:Economist)-[:TeachesAt]->(i:Institution)
RETURN e.name as Economist, i.name as Institution
""")

# 5. Find schools and count their founders
query_and_print_result("""
MATCH (s:School)-[:FoundedBy]->(e:Economist)
RETURN s.name as School, COUNT(e) as NumberOfFounders
""")

# 6. Find economists who both founded schools and teach at institutions
query_and_print_result("""
MATCH (s:School)-[:FoundedBy]->(e:Economist)-[:TeachesAt]->(i:Institution)
RETURN e.name as Economist, s.name as School, i.name as Institution
""")

# 7. Find economic concepts without any schools studying them
query_and_print_result("""
MATCH (c:EconomicConcept)
WHERE NOT EXISTS {
    MATCH (s:School)-[:Studies]->(c)
}
RETURN c.name
""")

# 8. Find economists with no institutional affiliations
query_and_print_result("""
MATCH (e:Economist)
WHERE NOT EXISTS {
    MATCH (e)-[:TeachesAt]->()
}
RETURN e.name
""")
```

How might you use this example? Using one or two shot prompting in LLM input prompts to specify data format and other information and then generating structured data of Python code is a common implementation pattern for using LLMs.

Here, the “structured data” I asked an LLM to output was Python code.

I cheated in this example by using what is currently the best code generation LLM: Claude Sonnet 3.5. I also tried this same exercise using Ollama with the model **qwen2.5-coder:14b** and the results were not quite as good. This is a great segway into the final chapter **Book Wrap Up**.


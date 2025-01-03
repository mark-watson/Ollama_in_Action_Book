# Using Property Graph Database with Ollama

I have a long history of working with Knowledge Graphs (at Google and OliveAI) and I usually use RDF graph databases and the SPARQL query language. Very recently I have developed a preference for property graph databases because recent research has shown that using LLMs with RDF-based graphs have LLM context size issues due to large schemas, overlapping relations, and complex identifiers that exceed LLM context windows. Property graph databases like Neo4J and Kuzu (which we use in this chapter) have more concise schemas.

It is true that Google and other players are teasing 'infinite context' LLMs but since this book is about running smaller models locally I have chosen to only show a property graph example.

## Overview of Property Graphs

TBD

## Example Using Ollama, LangChain, and the Kuzu Property Graph Database

The example shown here is derived from an example in the LangChain documentation: [https://python.langchain.com/docs/integrations/graphs/kuzu_db/](https://python.langchain.com/docs/integrations/graphs/kuzu_db/).

```python
import kuzu
from langchain.chains import KuzuQAChain
from langchain_community.graphs import KuzuGraph
from langchain_ollama.llms import OllamaLLM

db = kuzu.Database("test_db")
conn = kuzu.Connection(db)

# Create a table
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

graph = KuzuGraph(db, allow_dangerous_requests=True)

# Create a chain
chain = KuzuQAChain.from_llm(
    llm=OllamaLLM(model="qwen2.5-coder:14b"),
    graph=graph,
    verbose=True,
    allow_dangerous_requests=True,
)

print(graph.get_schema)

# Ask two questions
chain.invoke("Who acted in The Godfather: Part II?")
chain.invoke("Robert De Niro played in which movies?")
```

This code demonstrates the implementation of a graph database using Kuzu, integrated with LangChain for question-answering capabilities. The code initializes a database connection and establishes a schema with two node types (Movie and Person) and a relationship type (ActedIn), creating a graph structure suitable for representing actors and their film appearances.

The implementation populates the database with specific data about "The Godfather" trilogy and two prominent actors (Al Pacino and Robert De Niro). It uses Cypher-like query syntax to create nodes for both movies and actors, then establishes relationships between them using the ActedIn relationship type. The data model represents a typical many-to-many relationship between actors and movies.

This example then sets up a question-answering chain using LangChain, which combines the Kuzu graph database with the Ollama language model (specifically the qwen2.5-coder:14b model). This chain enables natural language queries against the graph database, allowing users to ask questions about actor-movie relationships and receive responses based on the stored graph data. The implementation includes two example queries to demonstrate the system's functionality.

Here is the output from this example:

```bash
$ rm -rf test_db 
(venv) Marks-Mac-mini:OllamaExamples $ p graph_kuzu_property_example.py
Node properties: [{'properties': [('name', 'STRING')], 'label': 'Movie'}, {'properties': [('name', 'STRING'), ('birthDate', 'STRING')], 'label': 'Person'}]
Relationships properties: [{'properties': [], 'label': 'ActedIn'}]
Relationships: ['(:Person)-[:ActedIn]->(:Movie)']

> Entering new KuzuQAChain chain...
Generated Cypher:

MATCH (p:Person)-[:ActedIn]->(m:Movie {name: 'The Godfather: Part II'})
RETURN p.name

Full Context:
[{'p.name': 'Al Pacino'}, {'p.name': 'Robert De Niro'}]

> Finished chain.

> Entering new KuzuQAChain chain...
Generated Cypher:

MATCH (p:Person {name: "Robert De Niro"})-[:ActedIn]->(m:Movie)
RETURN m.name

Full Context:
[{'m.name': 'The Godfather: Part II'}]

> Finished chain.
```

The Cypher query language is commonly used in property graph databases. Here the query:

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

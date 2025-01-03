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

# Create two tables and a relation:
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

## Using LLMs to Create Graph Databases for Text Data

Here we take a different approach of:

- Use the last example file **graph_kuzu_property_example.py** as an example for Claude Sonnet 3.5 to understand the Kuzo Python APIs.
- Have Claude Sonnet 3.5 read the file **data/economics** and create a schema for a new graph database and populate the schema from the contents of the file **data/economics**.
- Ask Claude Sonnet 3.5 to also generate query examples.

Except for my adding the utility function **query_and_print_result**, this code was generated by Claude Sonnet 3.5:

```python
"""
Created by Claude Sonnet 3.5 from prompt:

Given some text, I want you to define Property graph schemas for
the information in the text. As context, here is some Python code
for defining two tables and a relation and querying the data:

[[CODE FROM graph_kuzu_property_example.py]]

NOW, HERE IS THE TEST TO CREATE SCHEME FOR, and to write code to
create nodes and links conforming to the scheme:

[[CONTENTS FROM FILE data/economics.txt]]

"""

import kuzu

db = kuzu.Database("economics_db")
conn = kuzu.Connection(db)

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
    """Basic pretty printer for Kuzu query results"""
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

How might you use this example? Using one or two shot prompting in LLM input prompts to specify data format and other information and then generating structured data is a common implementation pattern for using LLMs.

Here, the “structured data” I asked an LLM to output was Python code.

I cheated in this example by using what is currently the best code generation LLM: Claude Sonnet 3.5. I also tried this same exercise using Ollama with the model **qwen2.5-coder:14b** and the results were not quite as good. This is a great Segway into the final chapter **Book Wrap Up**.
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


# Congee Grasph Based Memory for LLM Agents

Cognee provides a graph-based memory layer for LLM agents by organizing information into nodes and relationships (i.e. a knowledge graph) instead of flat chunks or isolated embeddings. The workflow follows an ECL pipeline (Extract, Cognify, Load): you ingest raw data (text, code, logs, etc.), transform (cognify) it into structured semantic units and relationships, then store both vector embeddings and graph structure. During query time, the agent can traverse the graph to retrieve context not just by embedding similarity but by semantic connections and paths.  ￼

This hybrid graph + vector architecture lets agents reason over global context and relational structure, improving recall of related facts that standard RAG might miss. For instance, if a concept appears across many documents, the graph connects its occurrences so downstream queries can pull broader context via paths. The graph also helps with explanation and provenance (why two pieces are related). When combined with tools like LlamaIndex, Cognee enables a “GraphRAG” pipeline: ingestion --> graph creation --> semantic querying.  




















$ uv run ollama_docs_test.py
Using CPython 3.12.0
Creating virtual environment at: .venv
░░░░░░░░░░░░░░░░░░░░ [0/174] Installing wheels...                                  Installed 174 packages in 14.66s

2025-10-09T22:46:34.806048 [info     ] Logging initialized            [cognee.shared.logging_utils] cognee_version=0.3.4 database_path=/Users/markw/GITHUB/python_playground/cognee_graph_agent_memory_openai/.venv/lib/python3.12/site-packages/cognee/.cognee_system/databases graph_database_name= os_info='Darwin 25.0.0 (Darwin Kernel Version 25.0.0: Wed Sep 17 21:35:32 PDT 2025; root:xnu-12377.1.9~141/RELEASE_ARM64_T6020)' python_version=3.12.0 relational_config=cognee_db structlog_version=25.4.0 vector_config=lancedb

2025-10-09T22:46:34.806243 [info     ] Database storage: /Users/markw/GITHUB/python_playground/cognee_graph_agent_memory_openai/.venv/lib/python3.12/site-packages/cognee/.cognee_system/databases [cognee.shared.logging_utils]
Reading files from /Users/markw/GITHUB/python_playground/data
Adding content from chemistry.txt...
User cee3ddc8-0dbb-47f4-84ca-a99ba20c1a12 has registered.
Adding content from sports.txt...
Adding content from health.txt...
Adding content from economics.txt...
Pauli Blendergast, an economist teaching at the University of Krampton Ohio, is known for saying that economics is bullshit.
Glassware is not central to chemistry as many experiments in both experimental and industrial applications can be performed without it.


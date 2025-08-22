# personal_agent.py
import os, glob, time, pathlib, typer
import time
from agno.document import Document
from typing import Optional
from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.tools import tool
from agno.storage.sqlite import SqliteStorage
from agno.memory.v2.db.sqlite import SqliteMemoryDb
from agno.memory.v2.memory import Memory
from agno.embedder.ollama import OllamaEmbedder
from agno.vectordb.chroma import ChromaDb
from agno.knowledge.text import TextKnowledgeBase
from agno.knowledge.markdown import MarkdownKnowledgeBase
from agno.knowledge.combined import CombinedKnowledgeBase

APP_DIR = pathlib.Path(os.environ.get("AGENT_HOME", "~/.personal_agent")).expanduser()
NOTES_DIR = APP_DIR / "notes"
DB_FILE = str(APP_DIR / "agent.db")
CHROMA_PATH = str(APP_DIR / "chroma")
MODEL_ID = os.environ.get("OLLAMA_MODEL", "llama3.2:latest") # "gpt-oss:20b")
EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")

APP_DIR.mkdir(parents=True, exist_ok=True)
NOTES_DIR.mkdir(parents=True, exist_ok=True)

KB = None

def build_kb(notes_path: pathlib.Path) -> CombinedKnowledgeBase:
    vec = ChromaDb(collection="personal_kb", path=CHROMA_PATH, persistent_client=True,
                   embedder=OllamaEmbedder(id=EMBED_MODEL, dimensions=768))
    md = MarkdownKnowledgeBase(path=str(notes_path), vector_db=vec)
    txt = TextKnowledgeBase(path=str(notes_path), vector_db=vec)
    return CombinedKnowledgeBase(sources=[md, txt], vector_db=vec)

def build_agent(kb: CombinedKnowledgeBase) -> Agent:
    memory = Memory(model=Ollama(id=MODEL_ID), db=SqliteMemoryDb(table_name="user_memories", db_file=DB_FILE))
    storage = SqliteStorage(table_name="agent_sessions", db_file=DB_FILE)
    agent = Agent(
        model=Ollama(id=MODEL_ID),
        knowledge=kb, search_knowledge=True,
        memory=memory, enable_agentic_memory=True, enable_session_summaries=True,
        storage=storage, add_history_to_messages=True, num_history_runs=3,
        tools=[file_search, file_read, save_note],
        show_tool_calls=True, markdown=True,
    )
    return agent

@tool(show_result=True)
def file_search(pattern: str, limit: int = 20) -> list[str]:
    paths = sorted(glob.glob(pattern, recursive=True))[:limit]
    return [str(p) for p in paths]

@tool(show_result=True)
def file_read(path: str, max_chars: int = 40000) -> str:
    p = pathlib.Path(path)
    if not p.exists() or not p.is_file(): return "not found"
    data = p.read_text(errors="ignore")
    return data[:max_chars]

def load_text_with_meta(kb, text, tags, meta=None):
    if tags == "":
        tags = 'none'
    doc = Document(
          content=text,
          meta_data={"source":"inline","tags":tags},  # non-empty
          id=f"inline-{int(time.time())}")
    kb.vector_db.insert(documents=[doc], filters=meta)

@tool(show_result=False)
def save_note(text: str, tags: str = "") -> str:
    typer.echo(f"save_note({text}, {tags}")
    ts = time.strftime("%Y%m%d-%H%M%S")
    fn = NOTES_DIR / f"{ts}{('-'+tags) if tags else ''}.md"
    fn.write_text(text)
    if tags == "" or tags == None:
        tags = 'none'
    if KB is not None: load_text_with_meta(KB, text, tags)
    return str(fn)

app = typer.Typer()

@app.command()
def ingest_dir(path: str = typer.Option(str(NOTES_DIR), help="Directory of .md/.txt notes")):
    global KB; KB = build_kb(pathlib.Path(path))
    KB.load(recreate=False)
    typer.echo("KB loaded")

@app.command()
def chat(user: str = "me"):
    global KB
    if KB is None:
        KB = build_kb(NOTES_DIR)
        KB.load(recreate=False)
    agent = build_agent(KB)
    typer.echo("Chat started. Type 'exit' to quit.")
    while True:
        msg = typer.prompt(f"{user}>")
        if msg.strip().lower() in ("exit","quit"): break
        agent.print_response(msg, user_id=user, stream=True)

if __name__ == "__main__":
    app()

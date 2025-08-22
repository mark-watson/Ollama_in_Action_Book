# Run:

```
uv run personal_agent.py chat
uv run personal_agent.py ingest_dir
```

In chat mode you can ask "read file <path to file>", etc. Also ingest source code example files and then ask for code generation.

## Debuging notes:

```
uv run python -i -c "import personal_agent as pa"
```

then in REPL:

```
pa.KB = pa.build_kb(pa.NOTES_DIR)
pa.KB.load(recreate=False)
agent = pa.build_agent(pa.KB)
# e.g., drive it:
# agent.print_response('hello', user_id='me', stream=True)
```

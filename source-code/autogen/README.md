# Running with uv

```
uv venv
uv pip install pip
uv run autogen_python_example.py
```

The agent will generate Python code that it tries to run in a sandbox. pip is required for that to pull in libraries for the generated Python code.

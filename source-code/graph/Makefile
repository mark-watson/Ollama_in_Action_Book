.PHONY: check test slow-tests clean

check:
	ruff check .
	uv run python -c "import grafeo_langchain, ladybug, langchain_community, langchain_core, langchain_experimental, langchain_ollama"

test:
	uv run python ../fast_test.py

slow-tests:
	uv run python graph_ladybug_from_text.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true

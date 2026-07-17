.PHONY: check test slow-tests clean

check:
	ruff check .
	uv run python -c "import mem0, ollama"

test:
	uv run python ../fast_test.py

slow-tests:
	uv run python mem0_persistence.py "What is the capital of France?"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
	rm -rf db_local

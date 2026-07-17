.PHONY: check test slow-tests clean

check:
	ruff check .
	uv run python -c "import zvec"

test:
	uv run python ../fast_test.py

slow-tests:
	@echo "RAG_zvec app.py uses an interactive input loop - run manually"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true

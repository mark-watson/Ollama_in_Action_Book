.PHONY: check test slow-tests clean

check:
	ruff check .
	uv run python -c "import ollama"

test:
	uv run python ../fast_test.py

slow-tests:
	uv run python inference_speed_tests.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true

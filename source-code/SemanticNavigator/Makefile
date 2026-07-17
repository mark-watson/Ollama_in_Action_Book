.PHONY: check test slow-tests clean

check:
	ruff check .
	uv run python -c "import gradio, ollama"

test:
	uv run python ../fast_test.py

slow-tests:
	@echo "SemanticNavigator is a Gradio web app - run app.py manually"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true

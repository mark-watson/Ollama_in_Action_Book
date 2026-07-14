.PHONY: check test slow-tests clean

check:
	ruff check .

test:
	uv run python ../fast_test.py

slow-tests:
	@echo "tools/ is a library - import and use individual modules"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
	rm -f test.db

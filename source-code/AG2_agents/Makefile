.PHONY: check test slow-tests clean

check:
	ruff check .

test:
	uv run python ../fast_test.py

slow-tests:
	@echo "AG2_agents uses autogen initiate_chat (interactive) - run agent.py manually"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true

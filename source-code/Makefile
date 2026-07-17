.PHONY: check test slow-tests clean

SUBDIRS = AG2_agents autogen benchmarking_model_inference_speeds chains DSP \
          graph judges langgraph memory OllamaCloud prompt_caching PydanticToolUse \
          RAG_zvec reasoning SemanticNavigator short_programs smolagents \
          tool_examples tools

check:
	ruff check .
	@for dir in $(SUBDIRS); do \
		if [ -f "$$dir/Makefile" ]; then \
			echo "==> $$dir"; \
			$(MAKE) -C $$dir check || true; \
		fi; \
	done

test:
	@for dir in $(SUBDIRS); do \
		if [ -f "$$dir/Makefile" ]; then \
			echo "==> $$dir"; \
			$(MAKE) -C $$dir slow-tests || true; \
		fi; \
	done

slow-tests:
	@for dir in $(SUBDIRS); do \
		if [ -f "$$dir/Makefile" ]; then \
			echo "==> $$dir"; \
			$(MAKE) -C $$dir test || true; \
		fi; \
	done

clean:
	rm -r -f */economics_db */test_db */__pycache__ */venv */.venv autogen/coding */*~ memory/db_local
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null; true

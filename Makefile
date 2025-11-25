.PHONY: prompting analysing

prompting:
	uv run --directory prompting main.py

analysing:
	uv run --directory analysing main.py

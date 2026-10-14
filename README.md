# Multi-Agent Technical Assistant

A LangGraph-based AI assistant that routes technical queries to specialized agents and combines their outputs when needed.

## Features

- **Knowledge Agent:** Explains technical concepts.
- **Coding Agent:** Generates code and helps debug programming problems.
- **Smart Routing:** Classifies queries as `knowledge`, `coding`, or `both`.
- **Parallel Execution:** Runs both agents when a query requires explanation and implementation.
- **Result Aggregation:** Combines both agents' responses into one answer.

## Tech Stack

- Python
- LangGraph
- LangChain
- Groq LLM
- Pydantic
- python-dotenv

## Workflow

User Query → Router → Selected Agent(s) → Aggregator → Final Answer

When both agents are needed, they execute in parallel before their results are combined.



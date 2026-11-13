# ai-service

FastAPI service for HIRE FLOW's AI features.

```bash
uv sync                                         # install deps into .venv
uv run fastapi dev src/ai_service/main.py       # dev server on :8000 (docs at /docs)
uv run pytest                                   # tests
uv run ruff check . && uv run ruff format .     # lint + format
uv run mypy src                                 # type check
```

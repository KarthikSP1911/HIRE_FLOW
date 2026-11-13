#!/usr/bin/env bash
# Lint + format staged Python files with Ruff. Skips (without failing) if Ruff isn't installed.
set -e

if ! command -v ruff >/dev/null 2>&1; then
  echo "⚠  ruff not found — skipping Python lint/format. Install with: pipx install ruff"
  exit 0
fi

ruff check --fix "$@"
ruff format "$@"

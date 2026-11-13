# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

HIRE FLOW is a polyglot monorepo:

| Path              | What                              |
| ----------------- | --------------------------------- |
| `apps/web`        | Next.js frontend (TypeScript)     |
| `apps/backend`    | Spring Boot microservices (Java)  |
| `apps/ai-service` | Python AI service                 |
| `apps/capture`    | Capture + speech-to-text          |
| `contracts/`      | OpenAPI specs and shared schemas  |
| `infra/`          | docker-compose and infrastructure |
| `docs/`           | Documentation                     |

Root tooling is managed with npm (Node 22, see `.nvmrc`). `apps/*` are npm workspaces.

## Commands

```bash
npm install            # install tooling + git hooks
npm run lint           # ESLint (JS/TS)
npm run lint:fix
npm run lint:py        # Ruff lint + format check (Python)
npm run format         # Prettier (JS/TS, Java, JSON, YAML, Markdown)
npm run format:check
```

Before committing, make sure `npm run lint` and `npm run format:check` pass.

## Committing

Every commit MUST follow [Conventional Commits](https://www.conventionalcommits.org/). This is enforced by commitlint in the `commit-msg` hook; staged files are linted/formatted by lint-staged in the `pre-commit` hook.

Format:

```
<type>(<scope>): <subject>

[optional body]
```

- **type**: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`
- **scope**: `web`, `backend`, `ai-service`, `capture`, `contracts`, `infra`, `docs`, `ci`, `repo`, `deps`
- **subject**: imperative mood, lowercase, no trailing period, max ~72 chars (`add`, not `added`/`adds`)
- One logical change per commit. A change spanning several apps should be split by scope when practical.
- Breaking changes: add `!` after the scope (`feat(backend)!: ...`) and a `BREAKING CHANGE:` footer.

Examples:

```
feat(web): add candidate dashboard
fix(backend): handle empty resume upload
chore(deps): bump prettier to 3.10
```

Rules for Claude:

- **Do NOT add a `Co-Authored-By: Claude ...` trailer or any other Claude/AI attribution** to commit messages or PR descriptions. Commits are authored solely by the user.
- Only commit or push when the user asks.
- Never bypass hooks (`--no-verify`). If a hook fails, fix the underlying lint/format/message issue and create the commit again.
- Don't commit secrets or `.env` files (only `.env.example`).

## Code style

- Formatting is owned by Prettier (JS/TS/Java/JSON/YAML/MD) and Ruff (Python); don't hand-format against them.
- Indentation per `.editorconfig`: 2 spaces by default, 4 for Java and Python.
- ESLint config: `eslint.config.js` (flat config, ESLint 9 — v10 is not yet supported by `eslint-plugin-react`).
- Python lint config: `ruff.toml`. Java linting (Checkstyle/Spotless) will live in the backend build once it exists.

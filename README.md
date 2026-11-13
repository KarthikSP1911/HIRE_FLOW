# HIRE FLOW

Project description coming soon.

## Repository structure

```
hireflow/
├── apps/
│   ├── web/              # Next.js frontend
│   ├── backend/          # Spring Boot microservices
│   ├── ai-service/       # Python AI service
│   └── capture/          # capture + speech-to-text
├── contracts/            # OpenAPI + shared schemas
├── infra/                # docker-compose
├── docs/
└── .github/
```

## Development

### Prerequisites

- Node.js 22 (`nvm use`)
- JDK 21 (backend)
- Python 3.12+ and [Ruff](https://docs.astral.sh/ruff/) (`pipx install ruff`) for the AI service

### Setup

```bash
npm install   # installs tooling and sets up git hooks (Husky)
```

### Scripts

| Command                | Description                                  |
| ---------------------- | -------------------------------------------- |
| `npm run lint`         | ESLint (JS/TS)                               |
| `npm run lint:fix`     | ESLint with auto-fix                         |
| `npm run lint:py`      | Ruff lint + format check (Python)            |
| `npm run format`       | Prettier (JS/TS, Java, JSON, YAML, Markdown) |
| `npm run format:check` | Prettier check only                          |
| `npm run commit`       | Interactive conventional-commit prompt       |

### Commit convention

Commits follow [Conventional Commits](https://www.conventionalcommits.org/) and are checked by commitlint:

```
<type>(<scope>): <subject>

feat(web): add candidate dashboard
fix(backend): handle empty resume upload
```

Scopes: `web`, `backend`, `ai-service`, `capture`, `contracts`, `infra`, `docs`, `ci`, `repo`, `deps`.

On commit, lint-staged runs ESLint/Prettier on staged JS/TS, Prettier on Java/JSON/YAML/Markdown, and Ruff on Python.

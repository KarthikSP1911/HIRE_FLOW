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
- JDK 21 for the backend (make sure `JAVA_HOME` points to it)
- [uv](https://docs.astral.sh/uv/) for the AI service (`curl -LsSf https://astral.sh/uv/install.sh | sh`), plus `uv tool install ruff`
- Docker for local Postgres and Testcontainers

### Setup

```bash
npm install                                         # JS deps + git hooks (Husky)
(cd apps/ai-service && uv sync)                     # Python deps
docker compose -f infra/docker-compose.yml up -d    # Postgres
```

### Running the apps

| App          | Path                        | Run                                         | URL                   |
| ------------ | --------------------------- | ------------------------------------------- | --------------------- |
| Web          | `apps/web`                  | `npm run dev:web`                           | http://localhost:3000 |
| Core service | `apps/backend/core-service` | `./mvnw spring-boot:run`                    | http://localhost:8080 |
| AI service   | `apps/ai-service`           | `uv run fastapi dev src/ai_service/main.py` | http://localhost:8000 |

Each app has a `.env.example` listing its configuration.

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

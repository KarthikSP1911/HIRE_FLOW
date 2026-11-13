# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

HIRE FLOW is a polyglot monorepo:

| Path              | What                                                                                     |
| ----------------- | ---------------------------------------------------------------------------------------- |
| `apps/web`        | Next.js frontend (TypeScript, App Router, Tailwind)                                      |
| `apps/backend`    | Spring Boot microservices (Java 21, Maven) — one folder per service, e.g. `core-service` |
| `apps/ai-service` | Python AI service (FastAPI, uv, Python 3.13)                                             |
| `apps/capture`    | Capture + speech-to-text                                                                 |
| `contracts/`      | OpenAPI specs and shared schemas                                                         |
| `infra/`          | docker-compose and infrastructure                                                        |
| `docs/`           | Documentation                                                                            |

Root tooling is managed with npm (Node 22, see `.nvmrc`). `apps/*` are npm workspaces.

## Commands

Prerequisites: Node 22, JDK 21 (`JAVA_HOME` must point to it), [uv](https://docs.astral.sh/uv/), Docker (for Postgres and Testcontainers).

```bash
# Root tooling (run from repo root)
npm install                 # install JS deps for all workspaces + git hooks
npm run lint                # ESLint (JS/TS)
npm run lint:py             # Ruff lint + format check (Python)
npm run format              # Prettier (JS/TS, Java, JSON, YAML, Markdown)
npm run format:check
npm run typecheck           # web TypeScript check

# Local infrastructure
docker compose -f infra/docker-compose.yml up -d   # Postgres on :5432

# Web — apps/web (Next.js, port 3000)
npm run dev:web
npm run build:web

# Backend — apps/backend/core-service (Spring Boot, port 8080)
cd apps/backend/core-service
./mvnw spring-boot:run      # needs Postgres running
./mvnw test                 # needs Docker (Testcontainers)
./mvnw -DskipTests package

# AI service — apps/ai-service (FastAPI + uv, port 8000)
cd apps/ai-service
uv sync
uv run fastapi dev src/ai_service/main.py
uv run pytest
uv run mypy src
```

Use `uv add <pkg>` / `uv add --dev <pkg>` for Python dependencies (never pip). Add a new Spring service as a sibling folder under `apps/backend/` following `core-service`.

Before committing, make sure lint, format check, type checks and tests pass for every app you touched.

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

## Design principles

Every change should keep the codebase **highly cohesive, loosely coupled and well abstracted**.

- Follow SOLID, KISS and YAGNI. Prefer a little duplication over the wrong abstraction; extract only when a pattern repeats (rule of three).
- **High cohesion**: each module/class/function has one reason to change. Group code by feature, not by technical type alone.
- **Low coupling**: depend on interfaces, not concrete implementations. Use constructor injection. No hidden global state, no circular dependencies, no reaching into another module's internals.
- **Abstraction**: put every external boundary (database, HTTP clients, AI model providers, speech-to-text engines, storage) behind an interface so it can be swapped or mocked.
- Composition over inheritance. Prefer immutable data. Fail fast with clear, typed errors; never swallow exceptions.
- Small functions, intention-revealing names, no magic numbers/strings (use constants or enums). Comments explain _why_, not _what_.
- Don't add speculative features, config flags or layers that the current task doesn't need.

## Monorepo boundaries

- Apps **never import each other's code**. They communicate only over REST using the OpenAPI contracts in `contracts/`.
- **Contract first**: change the OpenAPI spec in `contracts/` before implementing or changing an endpoint.
- Generate clients/DTOs from the specs (openapi-generator for Java, openapi-typescript for web, datamodel-code-generator for Python). Never hand-edit generated code.
- Each microservice owns its data and database schema. No shared tables, no cross-service DB access.

## Backend — Spring Boot (layered)

- Package by feature, then layer: `com.hireflow.<service>.<feature>.{controller,service,repository,dto,entity,mapper,exception}`, plus `com.hireflow.<service>.config`.
- **Controller**: HTTP concerns only — request validation (`@Valid`), mapping, status codes. No business logic. Never expose entities; accept/return DTOs only.
- **Service**: an interface plus an `*Impl` class. Holds business logic and `@Transactional` boundaries. Services depend on repositories/clients via interfaces.
- **Repository**: Spring Data interfaces, persistence only.
- DTOs are Java `record`s; entity ↔ DTO mapping via MapStruct.
- Constructor injection only (no field `@Autowired`); prefer `final` fields. Typed config via `@ConfigurationProperties`.
- Domain exceptions are mapped centrally in a `@RestControllerAdvice` returning `ProblemDetail`.
- DB migrations via Flyway; never rely on `ddl-auto` outside local dev.

## AI service — Python / FastAPI (layered)

Applies to `apps/ai-service` (and `apps/capture` if it is Python).

- Layout (under `src/ai_service/`): `main.py` (app factory) + `api/` (routers) → `services/` → `repositories/` and `clients/`, plus `schemas/` (Pydantic models) and `core/` (config, logging, dependencies).
- Routers stay thin; logic lives in services. Wire dependencies with FastAPI `Depends`.
- Define model/STT providers as `Protocol`s or ABCs so implementations can be swapped (e.g. different LLM or speech engine) without touching services.
- Full type hints (mypy strict). Config via `pydantic-settings`. Use `async` for I/O-bound paths; don't block the event loop with CPU-heavy work.

## Frontend — Next.js

- App Router. Server Components by default; add `"use client"` only where interactivity requires it.
- Structure: `src/app` (routes only), `src/features/<feature>/{components,hooks,api,types}`, shared `src/components/ui` and `src/lib`.
- Components never call `fetch` directly — go through the typed API client in `features/<feature>/api`.
- TypeScript strict, no `any`. Validate forms, API responses at trust boundaries, and env vars with zod.
- Keep components small and presentational where possible; put state and side effects in hooks.

## API & error standards

- Resource-oriented, plural nouns, kebab-case: `GET /api/v1/job-postings/{id}`. Version in the path (`/api/v1`).
- Use correct HTTP verbs and status codes (`201` + `Location` on create, `204` on delete, `404`, `409`, `422`, …).
- Errors use RFC 9457 `application/problem+json` with the same shape in every service.
- Pagination: `page`/`size` (or cursor) with a consistent envelope containing items and paging metadata.
- JSON fields in camelCase; timestamps ISO-8601 in UTC; IDs are opaque strings/UUIDs.
- Validate all input at the boundary (Bean Validation, Pydantic, zod). Make retries safe: PUT/DELETE idempotent, idempotency keys for critical POSTs.

## Testing

- New logic ships with tests; bug fixes include a regression test. Tests must pass before committing.
- **Java**: JUnit 5, AssertJ, Mockito for service unit tests; `@WebMvcTest` for controllers; `@DataJpaTest` + Testcontainers for repositories.
- **Python**: pytest, httpx/FastAPI `TestClient`; fake implementations of provider Protocols instead of real models.
- **Web**: Vitest + React Testing Library; Playwright for critical end-to-end flows.
- Test behaviour, not implementation. Arrange–Act–Assert, descriptive names. Mock only external boundaries.

## Security & configuration

- 12-factor config: everything environment-specific comes from env vars. Keep each app's `.env.example` up to date. No secrets in code, commits or logs.
- Authenticate at the edge, authorize in every service (least privilege). Never trust client-supplied identity or roles.
- OWASP basics: parameterized queries only, output encoding, strict CORS allowlist, rate limiting on public endpoints, validated file uploads (type/size).
- Candidate data (resumes, recordings, transcripts) is **PII**: never log it, encrypt in transit and at rest, keep retention minimal.
- Commit lockfiles; add dependencies only when necessary and prefer well-maintained ones.

## Observability

- Structured JSON logs. Propagate a correlation/trace ID across services (W3C `traceparent`, `X-Request-Id`) and include it in every log line.
- OpenTelemetry for traces and metrics.
- Health/readiness endpoints: Spring Actuator `/actuator/health`, FastAPI `/health`.
- Use the right log levels (`ERROR` = needs action, `WARN` = degraded, `INFO` = business events, `DEBUG` = diagnostics). Never log secrets or PII.

## Definition of done

- [ ] `npm run lint`, `npm run format:check` (and `npm run lint:py` for Python) pass
- [ ] Tests added/updated and passing
- [ ] OpenAPI contract in `contracts/` updated for any API change
- [ ] `.env.example` updated for any new config
- [ ] Docs / ADR updated if an architectural decision changed
- [ ] Commit follows the conventions above (no AI co-author trailer)

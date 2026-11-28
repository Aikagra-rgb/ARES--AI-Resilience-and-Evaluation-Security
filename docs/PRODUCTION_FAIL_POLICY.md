# ARES Production Fail Policy

ARES defines explicit failure modes per subsystem so security boundaries never fail open silently in production, while observability and optional enhancements degrade safely.

## Per-Endpoint & Subsystem Fail Matrix

| Subsystem / Route Pattern | Development / Test Default | Production Default | Outage / Error Behavior |
|---|---|---|---|
| **Gateway Enforcement** (`/api/gateway/enforce`, `/api/gateway/inspect-outbound`) | `GATEWAY_FAIL_CLOSED_ON_OUTAGE=false`, `HMAC_ENFORCE_MODE=permissive` | `GATEWAY_FAIL_CLOSED_ON_OUTAGE=true`, `HMAC_ENFORCE_MODE=enforcing_critical` | Requests missing HMAC signatures fail with `401 HMAC_AUTH_REQUIRED`. Vector store outages trip the circuit breaker and quarantine requests (`BLOCK`, severity `CRITICAL`) instead of silently passing uninspected prompts. |
| **Multi-Tenant Control Plane & Red-Team** (`/api/v1/clients*`, `/api/v1/redteam*`, `/api/moderation*`) | Auth enabled (toggleable only in dev/test), `permissive` HMAC | Fail closed (`ARES_AUTH_ENABLED=true`, `enforcing_critical` HMAC) | Unauthenticated requests receive `401`; cross-tenant requests receive `403`; unsigned requests to critical routes receive `401 HMAC_AUTH_REQUIRED`. |
| **OpenAI-Compatible Reverse Proxy** (`/v1/models`, `/v1/chat/completions`) | API Key / Tenant API Key required | API Key / Tenant Key required; HMAC verified if headers supplied (`enforcing_critical`) | Compatible with standard OpenAI SDKs using `Authorization: Bearer <key>` without custom HMAC headers, while enforcing strict input & outbound firewall inspection and failing closed when `GATEWAY_FAIL_CLOSED_ON_OUTAGE=true`. Set `HMAC_ENFORCE_MODE=enforcing` if custom proxies always sign `/v1/*`. |
| **Blazor Web Console RBAC & Sessions** (`src/Ares.Web`) | File/ephemeral DataProtection keys allowed in `Development` | Redis `IDistributedCache` + Redis DataProtection keys **required** (`AllowSingleNodeSessions=false`) | App startup throws `InvalidOperationException` in `Production` if Redis is not configured. Missing/evicted sessions fail closed to `/login` (`UnauthorizedAccessException` if API client invoked without active session). CSRF tokens required on `/auth/login` and `/auth/logout`. |
| **Prompt Optimizer** (`/api/v1/optimizer/optimize`, `UserPromptOptimizer`) | Predictable `FALLBACK_ORIGINAL` | Predictable `FALLBACK_ORIGINAL` — **never weakens Steps 1 or 6** | Step 1 (pre-optimization gateway inspection) and Step 6 (post-optimization canary/injection re-check) remain mandatory and fail closed on security violations. If LoRA/LLM synthesis or fidelity threshold verification fails, the optimizer safely returns the unrewritten input (`FALLBACK_ORIGINAL`) after passing security checks. |
| **Observability & Probes** (`/health`, `/healthz`, `/metrics`) | Non-blocking probes | Non-blocking `/health`; degraded-safe `/healthz`; API-key-protected `/metrics` | `/health` stays `200 ok` for orchestrator liveness unless the ASGI process is wedged. `/healthz` bounds DB/Qdrant probe timeouts (`3s`/`5s`) and reports `"status": "degraded"` safely without crashing. Rate-limiter Redis transient failures fail open with in-memory sliding-window fallback (`FallbackRateLimiter`). |

---

## Startup Validation Guardrails (`ENVIRONMENT=production`)

`ares.config.Settings` refuses to boot outside `{development, dev, local, test, testing}` unless all of the following hold:
1. `DATABASE_URL` uses `postgresql://` or `postgresql+psycopg://` (SQLite is forbidden).
2. `ARES_AUTH_ENABLED=true` and `ALLOW_DEV_IDENTITY=false`.
3. `ARES_API_KEY`, `HMAC_SECRET`, and `PROMPT_FINGERPRINT_SECRET` are changed from dev defaults and at least 24 characters long.
4. `EXECUTION_PAYLOAD_ENCRYPTION_KEY` and `GATEWAY_CANARY_TOKEN` are explicitly configured.
5. `HMAC_ENFORCE_MODE` is not `disabled` (defaults to `enforcing_critical`).
6. `GATEWAY_FAIL_CLOSED_ON_OUTAGE` defaults to `true`.

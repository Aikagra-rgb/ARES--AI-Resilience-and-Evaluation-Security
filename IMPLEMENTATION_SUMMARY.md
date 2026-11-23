# ARES Security System - Implementation Summary

## Version: 2.1 (Multi-Tenant Clients, Hardened Runtime & Dev Ergonomics)
## Date: November 2026
## Status: Multi-tenant client models active, async health probes, pooled Qdrant, dev SQLite fallback

---

## 🎯 Architecture & Stabilization Summary

ARES combines a **multi-layer AI Security Gateway**, **per-client DistilBERT/hybrid intent & threat models**, **prompt compression adapters**, **automated red-teaming targets**, and an **Interactive Blazor Server console**.

### ✅ Active Components & Hardening Updates

| Component | Status | Description |
| :--- | :--- | :--- |
| **Database** | ✅ PostgreSQL + Dev SQLite Fallback | PostgreSQL required outside development (`ENVIRONMENT=production`); local dev/test automatically falls back to `ares.db` / `.pytest_ares_test.db` when local Postgres is unreachable (`launch.py --sqlite` supported) |
| **Health Probes** | ✅ Fast `/health` + Async `/healthz` | `/health` is a non-blocking liveness probe; `/healthz` offloads DB and pooled Qdrant readiness checks to worker threads via `asyncio.to_thread` with fallback visibility |
| **Vector Store** | ✅ Lazy & Pooled `QdrantStore` | Zero network/disk I/O in `__init__`; configurable `QDRANT_TIMEOUT_SECONDS`; remote & local client pooling with graceful fallback to `./qdrant_data` |
| **Rate Limiting** | ✅ Active & Bounded (`RateLimitMiddleware`) | Thread-safe `MemoryRateLimiter` in `app/main.py` with lazy TTL cleanup (`cleanup_interval_seconds`) and `max_keys` eviction to prevent unbounded growth |
| **HMAC Signing** | ✅ Active (`HmacVerificationMiddleware`) | Defaults to `permissive` mode for dev ergonomics; `HmacVerifier` enforces clock skew + replay protection with bounded `max_nonce_cache_size` eviction |
| **Tracing & Logs** | ✅ Active | `async_trace_span` from `ares/observability/tracing.py` wired into `RuntimeEnforcer`; structured correlation-ID gateway logs in `ares/gateway/logging.py` |
| **Per-Client Models** | ✅ Active | `client_model_trainer.py` & `distilbert_intent_trainer.py` train per-org threat/intent classifiers & compression adapters, saving `ml_stats.json`, `client_models_ml_stats.json`, and `model_registry.json` |
| **Client Targets** | ✅ Active (`:8001`) | Three multi-tenant reference targets (`org_acme_fintech`, `org_dev_ops_agent`, `org_medshield_health`) in `tests/targets/` managed via `launch.py suite` / `targets` |
| **Frontend UI** | ✅ Interactive Server (.NET 8) | Adds **Client Tenants & ML Stats** (`/clients`) and live **Settings & Governance** (`/settings`) pages alongside Arena, Corpus Search, Optimizer, Gateway, and Reports |

---

## 📦 Current Architecture

```
backend/
├── ares/
│   ├── config.py              # Unified Settings (PostgreSQL in prod, SQLite allowed in dev/test, QDRANT_TIMEOUT_SECONDS)
│   ├── gateway/
│   │   ├── enforcer.py        # Strict severity-max policy + per-client ML Layer 1.5 + async_trace_span
│   │   ├── logging.py         # Structured decision ring buffer with correlation_id & organization_id
│   │   └── benign_corpus.py   # Benign reference prompts for contrastive scoring
│   ├── intent/
│   │   └── per_client_classifier.py  # Cached per-org threat & 5-class intent inference bundle
│   ├── vectordb/
│   │   ├── embedder.py        # Gemini/Ollama embedder with x-goog-api-key header auth
│   │   └── store.py           # Lazy pooled QdrantStore with async thread offloading & local fallback
│   ├── security/
│   │   ├── structural_detector.py   # Layer 0 deterministic detector (10 categories)
│   │   ├── rate_limiter.py    # MemoryRateLimiter with lazy TTL cleanup & max_keys cap
│   │   └── signing.py         # HmacVerifier with replay cache purging & bounded max_nonce_cache_size
│   ├── redteam/               # Attack generation, payload operators, target red-teaming
│   ├── optimizer/             # Enterprise prompt optimizer with per-client thresholds & Stage-1 compression
│   ├── evaluation/            # ML statistics, report generators
│   ├── training/              # Per-client model trainer, QLoRA adapter trainer & inference server
│   ├── api/routes/            # Gateway, clients (+ /stats), dashboard, optimizer, search, stats, corpus, harden, tests
│   └── observability/         # Prometheus metrics (/metrics), tracing.py spans, structured logging
├── app/
│   ├── main.py                # FastAPI app (CORSMiddleware, RateLimitMiddleware, HmacVerificationMiddleware, /health, /healthz)
│   ├── database.py            # SQLAlchemy engine (PostgreSQL + dev SQLite)
│   ├── models.py              # ORM models
│   └── orchestrator.py        # Test orchestration
└── tests/
    ├── conftest.py            # Auto-detects Postgres vs session SQLite fallback (.pytest_ares_test.db)
    └── targets/               # Multi-tenant client target apps on :8001
```

---

## 🔧 Key Technical Decisions

### Database Strategy (Production Strictness, Local Ergonomics)
- **Production (`ENVIRONMENT != development/dev/local/test/testing`)**: `DATABASE_URL` must be PostgreSQL (`postgresql+psycopg://...`).
- **Development / Testing**: Local runs via `launch.py` auto-detect Postgres (`:5432`), start a Docker Postgres container if available, or fall back cleanly to local SQLite (`ares.db`) unless `--require-postgres` is passed. `pytest` uses the same automatic fallback in `tests/conftest.py`.

### Security Middleware State in `app/main.py`
- `RateLimitMiddleware` is **active** (`MemoryRateLimiter(max_keys=10_000, cleanup_interval_seconds=60.0, max_idle_seconds=3600)`), ensuring bounded memory usage.
- `HmacVerificationMiddleware` is **active** in `permissive` mode by default (`HMAC_ENFORCE_MODE=permissive`) with `HmacVerifier(max_nonce_cache_size=100_000)` evicting expired and overflow nonces.
- `tracing.py` is **actively wired** into `RuntimeEnforcer.enforce` via `async_trace_span`.

---

*Updated: November 2026 — ARES AI Resilience and Evaluation Security System (v2.1)*
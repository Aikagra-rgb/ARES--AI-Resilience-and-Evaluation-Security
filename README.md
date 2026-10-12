# ARES — AI Resilience and Evaluation Security System

[![CI Pipeline](https://github.com/NeoGonsalves/ARES-AI-Resilience-and-Evaluation-Security-System/actions/workflows/ci.yml/badge.svg)](https://github.com/NeoGonsalves/ARES-AI-Resilience-and-Evaluation-Security-System/actions)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![.NET 8.0](https://img.shields.io/badge/.NET-8.0-purple.svg)](https://dotnet.microsoft.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![Qdrant Cloud](https://img.shields.io/badge/Qdrant%20Cloud-3%2C695%20Vectors-red.svg)](https://qdrant.tech/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-blue.svg)](https://www.postgresql.org/)

---

## 🛡️ Current Architecture (P0.0 Stabilization Complete)

This branch (`newBranch`) represents a **clean, production-grade foundation** after the P0.0 stabilization. All experimental v2.0 features (consensus engine, event bus, threat intel pipeline, SDK middleware, HMAC signing, ML moderation) have been **archived to `archive/c05f1fd-modules`** per the stabilization plan.

### ✅ What's Stable & Working

| Component | Implementation |
|-----------|----------------|
| **Database** | PostgreSQL-only (`DATABASE_URL` required; fails fast if missing) |
| **Gateway Policy** | Strict Severity-Max: deterministic hits non-negotiable; benign similarity never overrides; outcome = `max_severity` |
| **Embedder** | `x-goog-api-key` header auth; `EMBEDDING_ALLOW_HASH_FALLBACK=false` (fail closed) |
| **Frontend** | Blazor **Interactive Server only** (WebAssembly removed); `FastApiAresApiClient` clean DI |
| **Structural Detector** | Layer 0 deterministic <1ms (10 attack categories, 5 severity levels) |
| **Vector Shield** | Semantic similarity against 3,695 Qdrant vectors (<20ms) |
| **ML Classifier** | ComplementNB + TF-IDF (trained via `scripts/train_classifier.py`) |
| **Metrics** | 40+ Prometheus metrics; `/metrics` secured behind API key |
| **CI** | PostgreSQL 16 service; `compileall` + `ruff` gates; Python 3.11/3.12 + .NET 8.0 matrix |

### 🗂️ Archived (Restorable from `archive/c05f1fd-modules`)

- `ares/security/consensus.py` — weighted consensus (diluted deterministic hits)
- `ares/security/signing.py` — server-side HMAC (unused; client-only in .NET)
- `ares/security/moderation.py` — ML poisoning prevention (unwired in-memory queue)
- `ares/sdk/middleware.py` — embedded SDK (duplicate consensus, unwired)
- `ares/intel/pipeline.py` — threat intel (syntax error, no consumers)
- `ares/observability/tracing.py` — OTLP tracing (syntax error, no integration)
- `BlazorAresApiClient.cs` — WebAssembly client (hardcoded dev API key)

---

## 📁 Project Structure

```
.
├── .github/
│   └── workflows/
│       └── ci.yml              # PostgreSQL 16 service, compileall+ruff gates, pytest + .NET 8.0
├── backend/
│   ├── app/                    # Platform layer (FastAPI, SQLAlchemy, auth, providers)
│   │   ├── main.py             # FastAPI app (no rate limiter, /metrics secured)
│   │   ├── database.py         # PostgreSQL engine & sessions
│   │   ├── models.py           # ORM models
│   │   └── orchestrator.py     # Test orchestration
│   ├── ares/
│   │   ├── config.py           # Unified Settings (PostgreSQL-only, fail-fast validators)
│   │   ├── gateway/
│   │   │   ├── enforcer.py     # Strict severity-max policy (NO consensus)
│   │   │   └── benign_corpus.py
│   │   ├── vectordb/
│   │   │   ├── embedder.py     # Header auth, fail-closed fallback
│   │   │   └── store.py        # Qdrant client
│   │   ├── security/
│   │   │   └── structural_detector.py  # Layer 0 deterministic detector
│   │   ├── redteam/            # Attack generation
│   │   ├── optimizer/          # Prompt hardening
│   │   ├── evaluation/         # ML stats, reports, evaluators
│   │   ├── llm/                # Multi-provider LLM client
│   │   ├── api/routes/         # Gateway, dashboard, search, stats, corpus, harden, tests
│   │   └── observability/
│   │       └── metrics.py      # Prometheus metrics
│   ├── scripts/
│   │   └── train_classifier.py # Reproducible ComplementNB training
│   ├── tests/
│   │   ├── conftest.py         # PostgreSQL fixtures with table truncation
│   │   └── test_embedder_url.py# Verifies header auth, no masked key in URL
│   ├── requirements.txt
│   └── pyproject.toml
├── src/
│   └── Ares.Web/               # Blazor Interactive Server (.NET 8.0)
│       ├── Program.cs          # No WebAssembly; FastApiAresApiClient DI
│       ├── App.razor           # No window.ARES_CONFIG; SVG MIME fixed
│       ├── Pages/              # Overview, Arena, PromptOptimizer, Gateway, Stats
│       ├── Services/           # FastApiAresApiClient, Mock/Real clients
│       └── wwwroot/
├── docs/
│   └── THREAT_MODEL.md         # STRIDE, trust boundaries, vector/LLM limitations
├── archive/
│   └── c05f1fd-modules/        # Git branch with deleted v2.0 modules
├── IMPLEMENTATION_SUMMARY.md   # P0.0 stabilization details
├── Ares.sln
└── README.md
```

---

## 🚀 How to Run

### Prerequisites
- **Python 3.11+** (3.12 recommended)
- **.NET 8.0 SDK**
- **PostgreSQL 16** (local `C:\pgsql`, Docker, or CI service)
- **Qdrant Cloud** (or local instance)

### 1. Configure Environment

Create `backend/.env` (or export vars):

```env
# Security (required in production)
ARES_API_KEY=<32+ char secret>
GATEWAY_CANARY_TOKEN=<32+ char secret>
PROMPT_FINGERPRINT_SECRET=<32+ char secret>
EXECUTION_PAYLOAD_ENCRYPTION_KEY=<Fernet key>
ARES_AUTH_ENABLED=true

# LLM Providers (at least one)
GROQ_API_KEY=<key>
GEMINI_API_KEY=<key>
NVIDIA_API_KEY=<key>
OPENAI_API_KEY=<key>

# Vector DB
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=<key>

# Database (REQUIRED - PostgreSQL only)
DATABASE_URL=postgresql+psycopg://user:pass@host:5432/ares
```

### 2. Start Backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```
- API: http://localhost:8000
- Swagger: http://localhost:8000/docs
- Health: http://localhost:8000/healthz

### 3. Start Frontend

```powershell
dotnet run --project src/Ares.Web
```
- Dashboard: http://localhost:5000 / https://localhost:5001

### 4. Verify

```powershell
cd backend
pytest -m "not integration" -q
dotnet test Ares.sln
```

---

## 🔧 Key Technical Decisions

| Decision | Rationale |
|----------|-----------|
| **PostgreSQL only** | Eliminates SQLite drift; CI/prod parity; fail-fast config |
| **Severity-Max Gateway** | Deterministic hits cannot be outvoted; no consensus dilution |
| **Header auth for embedder** | `SecretStr.get_secret_value()` in `x-goog-api-key`; no key in URLs |
| **Fail-closed embeddings** | `EMBEDDING_ALLOW_HASH_FALLBACK=false` — report degraded, no pseudo-vectors |
| **Interactive Server only** | No browser secrets; simpler SignalR; smaller attack surface |
| **No in-memory rate limiter** | Flawed `X-Forwarded-For` trust; per-target limiting in P0 runners |
| **Cached enforcer** | Single `RuntimeEnforcer` in lifespan; models/benign vectors loaded once |

---

## 📚 Documentation

| File | Description |
|------|-------------|
| `docs/THREAT_MODEL.md` | STRIDE, trust boundaries, vector similarity limitations, LLM-as-Defender limitations |
| `IMPLEMENTATION_SUMMARY.md` | P0.0 stabilization details, archived modules, verification gates |
| `archive/c05f1fd-modules` | Git branch with deleted v2.0 modules for history |

---

## 📄 License

Developed for high-assurance AI resilience, evaluation, and production gateway enforcement. Architecture documents and technical decisions in [`docs/`](docs/).

*Updated: October 2026 — P0.0 Stabilization Complete*
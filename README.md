# ARES — AI Resilience and Evaluation Security System

[![CI Pipeline](https://github.com/NeoGonsalves/ARES-AI-Resilience-and-Evaluation-Security-System/actions/workflows/ci.yml/badge.svg)](https://github.com/NeoGonsalves/ARES-AI-Resilience-and-Evaluation-Security-System/actions)
[![Python 3.11 | 3.12](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![.NET 8.0](https://img.shields.io/badge/.NET-8.0-purple.svg)](https://dotnet.microsoft.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20Corpus-red.svg)](https://qdrant.tech/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-blue.svg)](https://www.postgresql.org/)
[![LoRA](https://img.shields.io/badge/LoRA-4--bit%20QLoRA-orange.svg)](https://github.com/huggingface/peft)

ARES is an enterprise-grade AI security gateway, resilience testing engine, and prompt optimization platform designed to safeguard LLM applications against adversarial attacks, prompt injections, and token waste.

---

## 🛡️ Core Capabilities

### 1. Multi-Layer Security Gateway (Strict Severity-Max)
- **Layer 0 (Structural Detector)**: Deterministic, sub-millisecond regex & pattern matching across 10 attack categories and 5 severity levels.
- **Layer 1 (Vector Shield)**: Semantic similarity against 3,695 enterprise attack vectors in Qdrant (<20ms).
- **Layer 2 (ML Classifier)**: Calibrated ComplementNB + TF-IDF model trained on enterprise vulnerability benchmarks.
- **Fail-Closed Embeddings**: Header-based `x-goog-api-key` authentication with strict fail-closed enforcement (`EMBEDDING_ALLOW_HASH_FALLBACK=false`).
- **Strict Severity-Max Policy**: Deterministic threat hits cannot be diluted or outvoted by benign similarity scores.

### 2. Enterprise Prompt Optimizer & Intent Classifier
- **5-Class Intent Taxonomy**: Sub-millisecond TF-IDF + Logistic Regression classification across `coding_technical`, `creative_writing`, `analytical_qa`, `system_directive`, and `data_extraction_json`.
- **Few-Shot Exemplar Store**: Dynamic Qdrant exemplar search for in-context demonstrations tailored to classified user intent.
- **4-Bit QLoRA Fine-Tuned Model**: Native fine-tuned adapter on Llama 3.1 8B Instruct delivering 42.9% net token compression while maintaining 0.854+ semantic fidelity across the exhaustive 513-sample holdout test benchmark.
- **Class-Aware Fidelity Gates & Structural Verification**: Calibrated per-class cosine similarity thresholds (`0.87` creative, `0.86` analytical, `0.85` coding, `0.80` JSON & system directives) paired with structural JSON schema field preservation (`verify_json_schema_preserved`) and imperative guardrail checklist verification (`verify_directive_markers`).
- **Dual-Routing Fallback**: Primary routing through local LoRA inference server (`localhost:11435`), with seamless automatic fallback to Ollama (`llama3.1:8b-instruct`) and cloud LLMs (Gemini, Groq, NVIDIA).

#### Prompt Optimizer Holdout Benchmark (N=513 samples)

| Intent Class | Evaluation Samples | Avg Original Tokens | Avg Optimized Tokens | Token Savings (%) | Mean Semantic Fidelity | Class Fidelity Gate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `analytical_qa` | 104 | 51.6 | 24.1 | 50.9% | 0.883 | >= 0.86 (`PASS`) |
| `coding_technical` | 104 | 44.2 | 25.8 | 42.4% | 0.852 | >= 0.85 (`PASS`) |
| `creative_writing` | 104 | 49.7 | 26.5 | 44.8% | 0.900 | >= 0.87 (`PASS`) |
| `data_extraction_json` | 104 | 41.4 | 26.6 | 32.0% | 0.813 | >= 0.80 (`PASS`) |
| `system_directive` | 97 | 59.1 | 37.8 | 33.6% | 0.819 | >= 0.80 (`PASS`) |
| **Overall Net** | **513** | **49.1** | **28.0** | **42.9%** | **0.854** | **All 5 Classes PASS** |



### 3. Red Team Arena & Automated Evaluation
- Automated adversarial prompt synthesis, payload mutation, and jailbreak simulation.
- 256-bit entropy canary token tracking to catch system prompt leakages.
- Multi-provider LLM-as-a-judge evaluation scoring with structured audit logging.

### 4. Interactive Blazor Server UI (.NET 8.0)
- Single-page management console: Overview Dashboard, Security Gateway live tester, Prompt Optimizer studio, Red Team Arena, **Client Tenants & ML Stats** (`/clients`), and **Settings & Governance** (`/settings`).
- Built exclusively with Blazor Interactive Server for security (no client-side API key leakage).

### 5. Multi-Tenant Client Models, Targets & Partner Deployment
- **Reference Client Target Applications (`:8001`)**: Three isolated multi-tenant LLM applications (`org_acme_fintech` Finance RAG, `org_dev_ops_agent` DevOps Tool Agent, and `org_medshield_health` HIPAA Triage Assistant) mounted in `backend/tests/targets/`.
- **Per-Client Model Training & ML Evaluation Statistics**: Train organization-specific threat detectors, 5-class intent routers, and domain compression adapters in seconds, computing 5-fold stratified CV accuracy, macro/weighted precision, recall, F1, and confusion matrices (`backend/data/client_models_ml_stats.json`, `GET /api/v1/clients/{org_id}/stats`):
  ```bash
  PYTHONPATH=backend python -m ares.training.client_model_trainer --org=all
  ```
- **Partner Sidecar Deployment (`deploy_partner.sh`)**: Generates partner `.env.partner.<org_id>` bundles paired with `docker-compose.full.yml` (or custom `--compose-file=...`):
  ```bash
  ./deploy_partner.sh --org=org_acme_fintech --api-key=YOUR_KEY --compose-file=docker-compose.full.yml
  ```

---

## 📁 Repository Structure

```
.
├── .github/
│   └── workflows/
│       └── ci.yml              # PostgreSQL 16, Python 3.11/3.12, .NET 8.0 matrix (with paths-ignore)
├── .agents/
│   └── rules/                  # Workspace development and testing rules
├── backend/
│   ├── app/                    # FastAPI platform layer (database, ORM models, auth, providers)
│   ├── ares/
│   │   ├── api/routes/         # Gateway, dashboard, optimizer, harden, search, tests
│   │   ├── gateway/            # Severity-max enforcer & benign corpus
│   │   ├── vectordb/           # Qdrant store & Gemini/Ollama embedders
│   │   ├── security/           # Structural detector, circuit breaker, rate limiters
│   │   ├── intent/             # 5-class intent classifier, 5,000+ dataset scaling, exemplar store
│   │   ├── optimizer/          # Enterprise prompt optimizer & dual-routing pipeline
│   │   ├── training/           # 4-bit QLoRA training, GGUF export, PEFT inference server
│   │   ├── redteam/            # Attack generation & payload operators
│   │   ├── evaluation/         # ML statistics & report generators
│   │   ├── llm/                # Multi-provider LLM client (Gemini, Groq, NVIDIA, Ollama)
│   │   └── observability/      # Prometheus metrics & structured logging
│   ├── ares_sdk/               # Standalone Python client library
│   ├── scripts/                # Evaluation & training CLI utilities
│   ├── tests/                  # 192 automated pytest unit tests
│   ├── requirements.txt
│   └── pyproject.toml
├── src/
│   └── Ares.Web/               # Blazor Interactive Server (.NET 8.0)
├── tests/
│   └── Ares.Web.Tests/         # 14 .NET xUnit unit tests
├── docs/                       # Architecture documents, API contracts, threat model
├── Ares.sln
└── README.md
```

---

## ⚡ Unified Terminal Launching Engine

ARES features an integrated cross-platform launching engine (`launch.py` and native PowerShell companion `launch.ps1`) for single-command orchestration, port management, health probing, and graceful Ctrl+C teardown.

### Quick Launch Commands

```bash
# Launch interactive terminal launcher menu
python launch.py
# Or on Windows PowerShell:
.\launch.ps1

# Launch both Frontend (.NET 8 Blazor) and Backend (FastAPI) concurrently
python launch.py both
.\launch.ps1 both

# Launch Backend (:8000) + Frontend (:5149) + Reference Client Targets (:8001)
python launch.py suite
python launch.py suite --sqlite --mock-targets   # Laptop-safe local SQLite + offline simulator

# Launch Backend only (FastAPI on http://127.0.0.1:8000)
python launch.py backend
.\launch.ps1 backend

# Launch Frontend only (Blazor Server on http://localhost:5149)
python launch.py frontend
.\launch.ps1 frontend

# Launch Client Targets only (:8001)
python launch.py targets --mock-targets
.\launch.ps1 targets -MockTargets

# Launch Full Stack (Frontend + Backend + Client Targets + LoRA Inference Server on :11435)
python launch.py all
.\launch.ps1 all

# Launch LoRA Inference Server only
python launch.py lora

# Check port status and active PIDs
python launch.py status
.\launch.ps1 status

# Free all ARES ports (terminate orphaned processes holding 8000, 5149, 8001, or 11435)
python launch.py clean
.\launch.ps1 clean
```

### Key Launcher Features
- **Concurrent Execution & Multiplexed Logging**: Runs frontend, backend, targets, and LoRA inference in unified terminal stream with colored, timestamped service tags (`[BACKEND]`, `[FRONTEND]`, `[TARGETS]`, `[LORA]`).
- **Automatic Database Readiness**: Checks PostgreSQL connectivity, attempts to start a Docker PostgreSQL container if Docker is running, and safely falls back to local SQLite (`ares.db`) in development unless `--require-postgres` is passed.
- **Readiness Probing**: Probes lightweight `http://127.0.0.1:8000/health` (and `:8001/healthz` for Client Targets) on startup before printing a clickable service card; deep dependency checks live at `/healthz`.
- **Clean Process Lifecycle**: Handles SIGINT/Ctrl+C signals cleanly and terminates entire child process trees, preventing orphaned uvicorn or dotnet processes from blocking development ports.
- **Automatic Port Conflict Resolution**: Automatically detects and frees ports `8000`, `5149`, `8001`, and `11435` before starting new instances.
- **Lock-Safe Frontend Builds**: Automatically passes `--no-restore` when `obj/project.assets.json` exists to prevent NuGet lock contention on repeated Windows launches.

---

## 🚀 Teammate Quickstart: Running on Any Machine

ARES is engineered to be **fully operational regardless of local GPU availability**. Choose the setup that matches your environment:

### Mode 1: Cloud API (Zero GPU / Zero Local AI Required)
*Recommended for most development tasks, MacBooks, and standard laptops.*

1. **Clone the repository**:
   ```bash
   git clone https://github.com/NeoGonsalves/ARES-AI-Resilience-and-Evaluation-Security-System.git
   cd ARES-AI-Resilience-and-Evaluation-Security-System
   ```

2. **Configure `backend/.env`**:
   ```env
   # Database (PostgreSQL required)
   DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/ares

   # Cloud Providers (at least one)
   GEMINI_API_KEY=your-gemini-api-key
   GROQ_API_KEY=your-groq-api-key

   # Vector Store (Qdrant Cloud or local)
   QDRANT_URL=https://your-cluster.qdrant.io
   QDRANT_API_KEY=your-qdrant-key

   # Security secrets
   ARES_API_KEY=your-ares-dev-secret-key-32-chars-min
   GATEWAY_CANARY_TOKEN=canary-token-secret-value-32-chars
   PROMPT_FINGERPRINT_SECRET=fingerprint-secret-value-32-chars
   EXECUTION_PAYLOAD_ENCRYPTION_KEY=your-fernet-key-here
   ```

3. **Launch Stack**:
   ```bash
   python launch.py both
   # Or on Windows: .\launch.ps1 both
   ```
   Backend API runs at `http://localhost:8000` (Docs: `http://localhost:8000/docs`)  
   Web Console runs at `http://localhost:5149`

---

### Mode 2: Local-First with Ollama
*For developers who want to run prompt inference and embeddings locally.*

1. Install [Ollama](https://ollama.com).
2. Pull the required models:
   ```bash
   ollama pull llama3.1:8b-instruct
   ollama pull nomic-embed-text
   ```
3. Set `PREFER_LOCAL_EMBEDDINGS=true` in `backend/.env`.
4. Run Backend & Frontend as described above. All evaluations and prompt optimizations will automatically route to your local Ollama daemon.

---

### Mode 3: Native LoRA Adapter GPU Server (NVIDIA RTX GPUs)
*For developers running with the fine-tuned 4-bit QLoRA adapter on an NVIDIA GPU.*

1. Install training & inference dependencies:
   ```bash
   pip install -r backend/ares/training/requirements_training.txt
   ```
2. Start the local LoRA inference server:
   ```bash
   python backend/ares/training/serve_lora.py --port 11435
   ```
3. The ARES optimizer pipeline will automatically detect the server on port 11435 and route prompts through the fine-tuned adapter.

---

## 🧪 Testing & Verification

Both backend and frontend test suites are fully isolated and executable offline without external API dependencies:

```bash
# Run Backend Test Suite (192 tests)
cd backend
pytest -m "not integration" -v

# Run Frontend .NET Test Suite (14 tests)
dotnet test Ares.sln --configuration Release
```

---

## 🔧 Key Technical Decisions

| Decision | Rationale |
| :--- | :--- |
| **PostgreSQL in Prod + Dev SQLite Fallback** | Enforces PostgreSQL in production (`ENVIRONMENT != development`), while allowing local dev/test to use SQLite when Docker/Postgres is stopped. |
| **Lazy Pooled Vector Store & Async Probes** | `QdrantStore.__init__` performs zero I/O, pools remote/local clients, and keeps `/health` + `/healthz` non-blocking. |
| **Severity-Max Gateway** | Deterministic attack hits cannot be overridden by benign semantic similarity. |
| **Fail-Closed Embeddings** | Prevents degraded or hash-fallback vectors from bypassing threat shields. |
| **Blazor Interactive Server** | Eliminates client-side secrets and minimizes attack surface by executing logic server-side. |
| **Dual-Routing Pipeline** | LoRA adapter server is hot-probed with zero-latency fallback to Ollama and cloud LLMs. |

---

## 📄 Documentation

- [Threat Model](docs/THREAT_MODEL.md): STRIDE threat assessment, trust boundaries, and mitigation controls.
- [Architecture Decisions](docs/architecture-decisions.md): Detailed ADR logs on data stores and routing policies.
- [API Contracts](docs/api-contracts.md): FastAPI REST endpoints and request/response specifications.

*Updated: November 2026*
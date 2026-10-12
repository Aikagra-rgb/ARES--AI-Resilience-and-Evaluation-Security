# ARES — AI Resilience and Evaluation Security System

[![CI Pipeline](https://github.com/NeoGonsalves/ARES-AI-Resilience-and-Evaluation-Security-System/actions/workflows/ci.yml/badge.svg)](https://github.com/NeoGonsalves/ARES-AI-Resilience-and-Evaluation-Security-System/actions)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![.NET 8.0](https://img.shields.io/badge/.NET-8.0-purple.svg)](https://dotnet.microsoft.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![Qdrant Cloud](https://img.shields.io/badge/Qdrant%20Cloud-3%2C695%20Vectors-red.svg)](https://qdrant.tech/)
[![Tests Passing](https://img.shields.io/badge/Tests-122%20Py%20%7C%2014%20.NET%20Passed-brightgreen.svg)](https://github.com/NeoGonsalves/ARES-AI-Resilience-and-Evaluation-Security-System--actions)

---

## 🛡️ Security Hardening v2.0

### Consolidated Configuration
- **Single source of truth**: `backend/ares/config.py` replaces dual `app/config.py` + `ares/config.py`
- **Production security enforced**: API key, canary token, prompt fingerprint secret, encryption key required
- **Environment validation**: Development-only settings rejected in production

### HMAC Request Signing
- **Server-side**: `Backend/ares/security/signing.py` - HMAC-SHA256 verification
- **Client-side**: `.NET FastApiAresApiClient` and `BlazorAresApiClient` with request signing
- **Coverage**: method, path, query params, body, timestamp, nonce
- **Purpose**: Prevents request forgery and replay attacks

### ML Poisoning Prevention
- **AttackModerationQueue**: Auto-approval policies with configurable confidence thresholds
- **Mandatory human review** for high-risk categories (data_exfiltration, tool_misuse, encoding_or_obfuscation)
- **Similarity detection** to prevent duplicate indexing
- **Audit trail** of all moderation decisions with escalation path
- **Convenience function**: `submit_for_moderation()`

### Async Event Bus for Audit Logging
- **InMemoryEventBus**: Async event publishing with multiple handlers per type
- **Dead letter queue** for failed events
- **AuditLogHandler**: Database persistence for audit events
- **WebhookEventHandler**: External integrations (Slack, PagerDuty, etc.)
- **Convenience functions**: `publish_event()`, `publish_audit_event()`

### Runtime Enforcement Gateway (DI-Refactored)
- **Per-request enforcer instances** via `get_enforcer()` and `get_qdrant_store()`
- **Removed module-level singleton** - supports per-tenant/organization configurations
- **Layer 0**: Structural Attack Detector - deterministic <1ms pattern matching
- **Layer 1**: Heuristic regex & canary token verification
- **Layer 1.5**: Fast ML Classifier
- **Layer 2**: Semantic Vector Similarity (<20ms)
- **Layer 2.5**: Contrastive Benign Margin Scoring
- **Layer 3**: Consensus Policy Decision with weighted voting

### Structural Attack Detector (Layer 0)
- **Deterministic rule-based detection** <1ms latency
- **10 attack categories**: instruction_override, role_hijack, delimiter_confusion, encoding_obfuscation, data_exfiltration, tool_misuse, indirect_injection, multi_turn_escalation, context_smuggling, persona_adoption
- **4 severity levels**: INFO, LOW, MEDIUM, HIGH, CRITICAL
- **4 decision types**: ALLOW, FLAG, CHALLENGE, BLOCK
- **Canary token detection** integrated

### Consensus Engine
- **Multiple protocols**: WeightedVoting, Unanimous, Hierarchical
- **Layer decisions**: Structural, Vector, ML Classifier, LLM Judge, Gateway, Human, Policy
- **Weighted voting** with configurable weights
- **Conflict detection** and escalation handlers
- **Escalation handlers**: logging, webhook, email

### Embedded SDK Middleware
- **Application-layer defense** with local cache + async gateway fallback
- **LRU cache with TTL** for decisions
- **Consensus engine** for multi-layer decisions
- **Non-blocking async gateway** calls
- **FastAPI middleware** integration ready

### Threat Intelligence Pipeline
- **Multi-source fetching**: GitHub advisories, arXiv papers, bug bounty platforms, local files
- **Auto-validation policies** with configurable thresholds
- **Human validation workflow** with callback support
- **Deduplication** via content hashing
- **Similarity search** against existing corpus
- **Automatic indexing** of approved samples

### Frontend (Blazor WebAssembly)
- **Blazor Server UI** (.NET 8) with Interactive Server and Interactive WebAssembly modes
- **HMAC request signing** via `BlazorAresApiClient`
- **CORS configuration** for frontend origins
- **API config** exposed via `window.ARES_CONFIG`
- **Components**: `App.razor`, `BlazorAresApiClient.cs`, updated `Program.cs`

### Threat Model Documentation
- **STRIDE analysis** with all 6 threat categories
- **Attack surface** documentation (external and internal vectors)
- **Security boundaries** with trust boundary diagram
- **Trust assumptions** table (what each component is trusted/not-trusted for)
- **Threat scenarios** with mitigations and residual risks
- **Security requirements** (functional + non-functional)
- **Incident response** procedures
- **Compliance considerations** (SOC 2, GDPR, ISO 27001, NIST AI RMF)
- **Out of scope** explicitly listed 8 categories
- **Risk acceptance** with ownership
- **Review schedule** (quarterly threat model review, etc.)
- **Vector similarity limitations** documented
- **LLM-as-Defender limitations** documented

### Observatory & Metrics
- **40+ Prometheus metrics** (counters, gauges, histograms, summaries)
- **`/metrics` endpoint** for Prometheus scraping
- **`MetricsMiddleware`** for automatic HTTP metrics collection
- **OpenTelemetry tracing** module (lazy init)
- **Convenience functions**: `record_gateway_decision()`, `record_structural_detection()`, etc.

### Test Results
```
122 passed, 1 failed (live integration test requiring valid API key)
```

---

## 📁 Project Structure

```
.
├── .github/              # GitHub Actions CI matrix
├── backend/              # Python backend (FastAPI)
│   ├── app/              # Platform database models, auth, providers
│   ├── ares/             # ARES Core Security Package
│   │   ├── security/     # structural_detector, consensus, signing, moderation
│   │   ├── observability/ # metrics, tracing
│   │   ├── intel/        # threat intelligence pipeline
│   │   ├── sdk/          # embedded SDK middleware
│   │   ├── gateway/      # enforcement engine (DI-refactored)
│   │   ├── config.py     # unified configuration
│   │   └── tests/        # 123 backend tests
│   ├── database.py       # SQLAlchemy engine & session management
│   ├── main.py           # App factory with middleware & health checks
│   └── requirements.txt  # Python dependencies
├── src/                  # .NET frontend (Blazor WebAssembly)
│   └── Ares.Web/         # Blazor Server UI
│       ├── Services/     # FastApiAresApiClient, BlazorAresApiClient
│       ├── Pages/        # Overview, Arena, PromptOptimizer, Gateway, Stats
│       └── wwwroot/      # Control surface scripts & theme styles
├── docs/                 # Architecture decisions & documentation
│   └── THREAT_MODEL.md   # Updated with vector similarity + LLM limitations
├── IMPLEMENTATION_SUMMARY.md  # Detailed change summary
└── Ares.sln              # Visual Studio / .NET Solution
```

---

## 🚀 How to Run the Project

### Prerequisites
- **Python 3.10+** (recommended Python 3.11 or 3.12)
- **.NET 8.0 SDK** ([Download](https://dotnet.microsoft.com/download/dotnet/8.0))
- **Qdrant Cloud Account** (or local Qdrant instance)

### Step 1: Configure Environment Variables

Create or update `backend/.env`:

```env
# Security
ARES_API_KEY=<32+ char secret>
GATEWAY_CANARY_TOKEN=<32+ char secret>
PROMPT_FINGERPRINT_SECRET=<32+ char secret>
EXECUTION_PAYLOAD_ENCRYPTION_KEY=<Fernet key>
ARES_AUTH_ENABLED=true

# LLM Providers (at least one required)
GROQ_API_KEY=<key>
GEMINI_API_KEY=<key>
NVIDIA_API_KEY=<key>
OPENAI_API_KEY=<key>

# Vector DB
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=<key>

# Database
DATABASE_URL=postgresql+psycopg://user:pass@host:5432/ares
```

### Step 2: Start the Backend (FastAPI)

In a PowerShell terminal:

```powershell
cd backend
# Create & activate virtual environment
python -m venv .venv
\.venv\Scripts\Activate.ps1
# Install dependencies
pip install -r requirements.txt
# Run server with live reload
python main.py
```

- **Live API URL**: http://localhost:8000
- **Interactive Swagger Docs**: http://localhost:8000/docs
- **Health Check Probe**: http://localhost:8000/healthz

### Step 3: Start the Frontend (Blazor Server)

In a second PowerShell terminal:

```powershell
dotnet run --project src/Ares.Web
```

- **Frontend Web Dashboard**: http://localhost:5000 or https://localhost:5001

### Step 4: Verify

```powershell
# Run backend tests
cd backend
pytest -m "not integration" -q

# Run frontend tests
dotnet test Ares.sln
```

---

## 📊 Test Results

```
122 passed, 1 failed (live integration test requiring valid API key)
```

---

## 📄 License & Attribution

Developed for high-assurance AI resilience, evaluation, and production gateway enforcement. Architecture documents and technical decisions are detailed in the [`docs/`](docs/) directory. 

*Generated: October 2026*
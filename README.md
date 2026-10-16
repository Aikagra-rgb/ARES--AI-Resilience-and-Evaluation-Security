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
- **4-Bit QLoRA Fine-Tuned Model**: Native fine-tuned adapter on Llama 3.1 8B Instruct delivering 42.8% net token compression while maintaining 0.855+ semantic fidelity across holdout evaluation benchmarks.
- **Dual-Routing Fallback**: Primary routing through local LoRA inference server (`localhost:11435`), with seamless automatic fallback to Ollama (`llama3.1:8b-instruct`) and cloud LLMs (Gemini, Groq, NVIDIA).

#### Prompt Optimizer Holdout Benchmark (N=100 samples)

| Intent Class | Evaluation Samples | Avg Original Tokens | Avg Optimized Tokens | Token Savings (%) | Mean Semantic Fidelity |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `analytical_qa` | 22 | 56.3 | 23.0 | 58.1% | 0.873 |
| `coding_technical` | 22 | 47.5 | 31.1 | 39.0% | 0.843 |
| `creative_writing` | 17 | 49.3 | 25.5 | 45.0% | 0.915 |
| `data_extraction_json` | 19 | 38.4 | 25.6 | 30.4% | 0.818 |
| `system_directive` | 20 | 56.0 | 36.7 | 30.9% | 0.833 |
| **Overall Net** | **100** | **49.7** | **28.4** | **42.8%** | **0.855** |


### 3. Red Team Arena & Automated Evaluation
- Automated adversarial prompt synthesis, payload mutation, and jailbreak simulation.
- 256-bit entropy canary token tracking to catch system prompt leakages.
- Multi-provider LLM-as-a-judge evaluation scoring with structured audit logging.

### 4. Interactive Blazor Server UI (.NET 8.0)
- Single-page management console: Overview Dashboard, Security Gateway live tester, Prompt Optimizer studio, and Red Team Arena.
- Built exclusively with Blazor Interactive Server for security (no client-side API key leakage).

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
│   ├── tests/                  # 188 automated pytest unit tests
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

3. **Start Backend**:
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   python main.py
   ```
   Backend API runs at: `http://localhost:8000` (Docs: `http://localhost:8000/docs`)

4. **Start Frontend**:
   ```bash
   dotnet run --project src/Ares.Web
   ```
   Web UI runs at: `http://localhost:5000`

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
# Run Backend Test Suite (188 tests)
cd backend
pytest -m "not integration" -v

# Run Frontend .NET Test Suite (14 tests)
dotnet test Ares.sln --configuration Release
```

---

## 🔧 Key Technical Decisions

| Decision | Rationale |
| :--- | :--- |
| **PostgreSQL-Only** | Guarantees schema consistency across local dev, CI service containers, and production. |
| **Severity-Max Gateway** | Deterministic attack hits cannot be overridden by benign semantic similarity. |
| **Fail-Closed Embeddings** | Prevents degraded or hash-fallback vectors from bypassing threat shields. |
| **Blazor Interactive Server** | Eliminates client-side secrets and minimizes attack surface by executing logic server-side. |
| **Dual-Routing Pipeline** | LoRA adapter server is hot-probed with zero-latency fallback to Ollama and cloud LLMs. |

---

## 📄 Documentation

- [Threat Model](docs/THREAT_MODEL.md): STRIDE threat assessment, trust boundaries, and mitigation controls.
- [Architecture Decisions](docs/architecture-decisions.md): Detailed ADR logs on data stores and routing policies.
- [API Contracts](docs/api-contracts.md): FastAPI REST endpoints and request/response specifications.

*Updated: October 2026*
# ARES Threat Model

**Version:** 1.0  
**Date:** 2026-10-12  
**Classification:** Internal - Security Documentation

---

## 1. System Overview

ARES (AI Resilience and Evaluation Security) is a multi-layer defense system for LLM applications. It provides:
- **Pre-deployment evaluation** of system prompts against adversarial attacks
- **Runtime enforcement** via gateway and embedded SDK
- **Continuous threat intelligence** for evolving attack patterns

### 1.1 Components

| Component | Role | Trust Level |
|-----------|------|-------------|
| Structural Detector | Deterministic pattern matching | High (auditable) |
| Vector Shield | Semantic similarity search | Medium |
| ML Classifier | Statistical categorization | Medium |
| LLM Judge | Semantic evaluation | Low (probabilistic) |
| Gateway | Orchestration layer | Medium |
| Embedded SDK | Application-layer defense | High |
| Threat Intel Pipeline | Corpus updates | Medium |

---

## 2. STRIDE Analysis

### 2.1 Spoofing

| Threat | Target | Likelihood | Impact | Mitigation |
|--------|--------|------------|--------|------------|
| Attacker impersonates legitimate user | Gateway/SDK | High | High | API key auth, mTLS, request signing |
| Attacker spoofs gateway response | SDK | Medium | Critical | Response validation, certificate pinning |
| Compromised LLM provider | All LLM-dependent layers | Low | Critical | Multi-provider consensus, fallback to deterministic |

### 2.2 Tampering

| Threat | Target | Likelihood | Impact | Mitigation |
|--------|--------|------------|--------|------------|
| Modify attack corpus | Vector Shield | Medium | High | Immutable corpus, hash verification |
| Modify ML model | ML Classifier | Low | High | Model signing, version pinning |
| Tamper with gateway config | Gateway | Medium | High | Config validation, audit logging |
| Prompt injection in user input | All layers | Very High | Critical | Structural detection, consensus |

### 2.3 Repudiation

| Threat | Target | Likelihood | Impact | Mitigation |
|--------|--------|------------|--------|------------|
| Deny malicious request | Audit system | Medium | Medium | Immutable audit logs, correlation IDs |
| Deny blocked request | Gateway/SDK | Medium | Low | Detailed decision logging |

### 2.4 Information Disclosure

| Threat | Target | Likelihood | Impact | Mitigation |
|--------|--------|------------|--------|------------|
| System prompt extraction | LLM | Very High | Critical | Canary tokens, structural detection |
| Attack corpus disclosure | Vector DB | Low | Medium | Access controls, encryption |
| ML model inversion | ML Classifier | Low | Low | Rate limiting, output filtering |
| Internal config leakage | All | Medium | High | Secrets management, no hardcoded secrets |

### 2.5 Denial of Service

| Threat | Target | Likelihood | Impact | Mitigation |
|--------|--------|------------|--------|------------|
| Flood gateway with requests | Gateway | High | High | Rate limiting, request size limits |
| Exhaust LLM quota | LLM providers | High | Medium | Local fallbacks, caching |
| Large payload DoS | All HTTP endpoints | Medium | High | Request size limits (10MB) |
| Corpus poisoning | Vector Shield | Low | Medium | Validation before indexing |

### 2.6 Elevation of Privilege

| Threat | Target | Likelihood | Impact | Mitigation |
|--------|--------|------------|--------|------------|
| Bypass gateway via direct LLM access | Application | High | Critical | Embedded SDK, defense in depth |
| Admin API abuse | Management APIs | Medium | Critical | RBAC, audit logging |
| Prompt injection → system access | LLM | Very High | Critical | Multi-layer detection, consensus |

---

## 3. Attack Surface


### 2.7 LLM-as-Defender Limitations

The ARES system uses LLM-based judgment (LLM Judge layer) as an optional component within the consensus engine. The following limitations apply:

| Limitation | Description | Risk |
|------------|-------------|------|
| Probabilistic, not deterministic | LLM judgments are based on statistical likelihood, not formal verification | HIGH - May incorrectly classify attacks or benign prompts |
| Temperature-dependent | Output varies with sampling temperature; consistent decisions require fixed temperature | MEDIUM - Requires deterministic configuration |
| Context window limits | Judgments may be affected by prompt length and position bias | MEDIUM - Shorter, focused prompts preferred |
| Provider variability | Different LLM providers may give different verdicts for same input | HIGH - Consensus requires multi-provider agreement |
| Training cut-off | Knowledge limited to training data; may miss recent attack patterns | MEDIUM - Threat intel pipeline mitigates this |
| Resource constraints | LLM inference is slower (200ms-5s) compared to deterministic layers | HIGH - Typically made optional; consensus can overrule |
| Prompt injection of judge | Adversarial prompts designed to trick the judge itself | HIGH - Structural detector runs first as guard |

**Mitigations:**
- LLM Judge is optional and can be disabled via `GATEWAY_JUDGE_ENABLED=false`
- Consensus engine requires weighted voting; LLM Judge does not decide alone
- Multi-provider consensus (Groq + Gemini + NVIDIA) reduces single-point failure
- Structural detector (Layer 0) runs first and can block before LLM Judge is consulted
- Fixed temperature configuration for reproducible decisions
- Log all LLM judge decisions for post-incident analysis
- Fallback to heuristic+vector+ML consensus if LLM Judge unavailable or unreliable

---
### 3.1 External Attack Vectors

1. **User Prompts** - Primary attack surface
   - Direct injection
   - Encoded/obfuscated payloads
   - Multi-turn escalation
   - Role hijacking

2. **External Content** (Indirect Injection)
   - URLs/documents fed to RAG
   - Search results
   - Tool outputs

3. **API Endpoints**
   - `/api/gateway/enforce`
   - `/api/gateway/chat`
   - `/api/v1/harden`
   - `/api/v1/search`

4. **Management Interfaces**
   - Corpus management
   - Model retraining
   - Configuration APIs

### 3.2 Internal Attack Vectors

1. **Compromised Dependencies**
   - Python packages
   - npm packages (frontend)
   - Container images

2. **Insider Threats**
   - Corpus manipulation
   - Config changes
   - Model swapping

3. **Infrastructure**
   - Qdrant access
   - Database access
   - LLM provider credentials

### 3.3 Vector Similarity Limitations

Vector similarity (semantic search) is an effective layer for catching known attack variants, but has important limitations:

| Limitation | Description | Impact |
|------------|-------------|--------|
| Known-variant only | Vectors catch semantically similar attacks to known samples, but not truly novel patterns | MEDIUM - Requires complementary structural/heuristic layers |
| Embedding mismatch | Different embedding models produce different vectors; cross-model comparison is unreliable | HIGH - Must use consistent embedding model or enable fallback |
| High-dimensional sparsity | Attack space is vast; coverage is sparse even with 3,695 vectors | MEDIUM - New attack paths may have no close vector neighbors |
| Semantic drift | Embedding models may evolve between versions, changing vector distances | LOW - Requires vector re-indexing on model updates |
| Obfuscation resistance | Certain encoding/obfuscation techniques can reduce semantic similarity | MEDIUM - Structural detector handles these cases |

**Mitigations:**
- Vector similarity is Layer 2, complementing Layer 0 structural detection
- Enable embedding fallback to local model when Gemini API unavailable
- Maintain structural detector as deterministic fallback (Layer 0)
- Track vector coverage gaps and prioritize new sample indexing

---

## 4. Security Boundaries

```
┌─────────────────────────────────────────────────────────────┐
│                     TRUST BOUNDARY                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │   Client     │  │  Gateway     │  │   LLM Provider   │  │
│  │  Application │──▶│  (Optional)  │──▶│   (Untrusted)    │  │
│  │              │  │              │  │                  │  │
│  │ ┌──────────┐ │  │ ┌──────────┐ │  │                  │  │
│  │ │Embedded  │ │  │ │Structural│ │  │                  │  │
│  │ │  SDK     │ │  │ │Detector  │ │  │                  │  │
│  │ └──────────┘ │  │ └──────────┘ │  │                  │  │
│  │              │  │ ┌──────────┐ │  │                  │  │
│  │              │  │ │Vector    │ │  │                  │  │
│  │              │  │ │Shield    │ │  │                  │  │
│  │              │  │ └──────────┘ │  │                  │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 4.1 Trust Assumptions

| Component | Trusted For | NOT Trusted For |
|-----------|-------------|-----------------|
| Structural Detector | Syntax/pattern detection | Semantic understanding |
| Vector Shield | Known attack variants | Novel attacks |
| ML Classifier | Statistical patterns | Adversarial robustness |
| LLM Judge | Semantic evaluation | Truthfulness, consistency |
| Gateway | Orchestration | Single point of truth |
| Embedded SDK | Local fast decisions | Global threat intelligence |
| Threat Intel | Corpus freshness | Validation accuracy |

---

## 5. Data Flow & Trust Zones

### 5.1 Prompt Inspection Flow

```
User Prompt
    │
    ▼
┌─────────────────┐
│ 1. Structural   │──▶ BLOCK (critical patterns)
│    Detector     │
└────────┬────────┘
         │ ALLOW/FLAG/CHALLENGE
         ▼
┌─────────────────┐
│ 2. Local Cache  │──▶ CACHED DECISION
│    Check        │
└────────┬────────┘
         │ MISS
         ▼
┌─────────────────┐
│ 3. Vector Shield│──▶ BLOCK (high similarity)
│    (Optional)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 4. ML Classifier│──▶ FLAG (category match)
│    (Optional)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 5. LLM Judge    │──▶ CHALLENGE/BLOCK
│    (Optional)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 6. Consensus    │──▶ FINAL DECISION
│    Engine       │
└─────────────────┘
```

### 5.2 Trust Zone Transitions

| From Zone | To Zone | Data | Validation |
|-----------|---------|------|------------|
| Client | Embedded SDK | User prompt | Input validation, size limits |
| SDK | Gateway | Prompt + context | Auth, rate limiting |
| Gateway | Vector DB | Embedding vector | Auth, TLS |
| Gateway | LLM Provider | Prompt + system | Auth, provider validation |
| Threat Intel | Vector DB | Attack samples | Human validation |

---

## 6. Threat Scenarios & Mitigations

### 6.1 Scenario: Novel Prompt Injection (Zero-Day)

**Attack:** Attacker crafts previously unseen injection technique.

**Path:** User → Prompt → Structural Detector (miss) → Vector Shield (miss) → ML (miss) → LLM Judge (maybe catch) → Consensus

**Mitigations:**
- Structural detector catches syntactic patterns even in novel attacks
- LLM Judge provides semantic backstop
- Consensus requires multiple layers to miss
- Threat intel pipeline adds new patterns within hours

**Residual Risk:** MEDIUM - Sophisticated novel attacks may bypass all automated layers

### 6.2 Scenario: Corpus Poisoning

**Attack:** Attacker submits benign-looking attacks that get indexed, then uses variations.

**Path:** Threat Intel → Validation → Indexing → Vector Shield

**Mitigations:**
- Human validation required for all corpus additions
- Similarity threshold (0.85) prevents near-duplicates
- False positive reporting removes bad samples
- Corpus versioning enables rollback

**Residual Risk:** LOW - Multi-stage validation

### 6.3 Scenario: Gateway Bypass

**Attack:** Application calls LLM directly, bypassing gateway.

**Path:** Application → LLM Provider (no gateway)

**Mitigations:**
- Embedded SDK in application (defense in depth)
- Network policies blocking direct LLM access
- Audit logging of all LLM calls
- SDK provides local structural detection even if gateway down

**Residual Risk:** MEDIUM - Requires application cooperation

### 6.4 Scenario: LLM Provider Compromise

**Attack:** Malicious or compromised LLM provider returns manipulated responses.

**Path:** Gateway/SDK → LLM Provider → Response

**Mitigations:**
- Multi-provider consensus (require agreement)
- Response validation against safety policies
- Fallback to deterministic layers if provider behaves anomalously
- Provider health monitoring

**Residual Risk:** LOW - Defense in depth

---

## 7. Security Requirements

### 7.1 Functional Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-01 | Detect instruction override attempts | CRITICAL |
| FR-02 | Detect role hijacking attempts | CRITICAL |
| FR-03 | Detect delimiter confusion attacks | HIGH |
| FR-04 | Detect encoding/obfuscation | HIGH |
| FR-05 | Detect data exfiltration prompts | HIGH |
| FR-06 | Detect tool misuse prompts | MEDIUM |
| FR-07 | Multi-layer consensus decision | CRITICAL |
| FR-08 | Sub-10ms local detection | HIGH |
| FR-09 | <100ms gateway path | MEDIUM |
| FR-10 | Audit trail for all decisions | CRITICAL |

### 7.2 Non-Functional Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| NFR-01 | 99.9% availability (gateway) | HIGH |
| NFR-02 | <1ms structural detection | CRITICAL |
| NFR-03 | Zero hardcoded secrets | CRITICAL |
| NFR-04 | Encrypted data at rest | HIGH |
| NFR-05 | TLS 1.3 for all comms | CRITICAL |
| NFR-06 | Immutable audit logs | HIGH |
| NFR-07 | Configurable thresholds | MEDIUM |
| NFR-08 | Horizontal scalability | HIGH |

---

## 8. Incident Response

### 8.1 Detection
- Automated alerts on consensus conflicts
- Anomaly detection on decision distributions
- False positive rate monitoring
- Gateway health checks

### 8.2 Response
1. **Immediate:** Block offending IP/API key
2. **Short-term:** Add emergency pattern to structural detector
3. **Medium-term:** Update threat intel corpus
4. **Long-term:** Retrain ML, update LLM judge prompts

### 8.3 Recovery
- Corpus rollback capability
- Configuration versioning
- Canary deployments
- Chaos engineering for gateway

---

## 9. Compliance Considerations

| Standard | Applicable Controls |
|----------|---------------------|
| SOC 2 Type II | Audit logging, access control, encryption |
| GDPR | Data minimization, right to deletion (audit logs) |
| ISO 27001 | Risk assessment, incident management |
| NIST AI RMF | AI-specific risk management |

---

## 10. Out of Scope

The following threats are **explicitly out of scope** for ARES:

1. **Model-level vulnerabilities** (weights, architecture, training data)
2. **Supply chain attacks** on LLM providers
3. **Side-channel attacks** (timing, power, cache)
4. **Hardware/firmware vulnerabilities**
5. **Physical security** of infrastructure
6. **Multimodal attacks** (images, audio, video)
7. **State-based attacks** (fine-tuning, RLHF manipulation)
8. **Insider threats** with full system access

---

## 11. Risk Acceptance

| Risk | Likelihood | Impact | Decision | Owner |
|------|------------|--------|----------|-------|
| Novel zero-day injection | Medium | High | Accept with monitoring | Security Team |
| Gateway bypass | Medium | Critical | Mitigate via SDK | Engineering |
| LLM judge manipulation | Low | High | Accept with consensus | ML Team |
| Corpus poisoning | Low | Medium | Mitigate via validation | Data Team |

---

## 12. Review Schedule

| Review | Frequency | Participants |
|--------|-----------|--------------|
| Threat model review | Quarterly | Security, Engineering, ML |
| Incident retrospective | Post-incident | All stakeholders |
| Penetration testing | Semi-annually | External red team |
| Dependency review | Monthly | DevSecOps |

---

## Appendix A: Attack Pattern Taxonomy

See `ares/security/structural_detector.py` for canonical pattern definitions.

---

## Appendix B: Decision Matrix

| Structural | Vector | ML | LLM Judge | Consensus | Final |
|------------|--------|-----|-----------|-----------|-------|
| BLOCK | * | * | * | - | BLOCK |
| CHALLENGE | BLOCK | * | * | Weighted | BLOCK |
| FLAG | CHALLENGE | CHALLENGE | * | Weighted | CHALLENGE |
| ALLOW | ALLOW | ALLOW | ALLOW | - | ALLOW |
| ALLOW | BLOCK | ALLOW | ALLOW | Weighted | CHALLENGE |

---

*Document owner: Security Engineering*  
*Next review: 2027-01-12*
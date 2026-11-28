# ARES Reproducible Three-Score Evaluation Report

- **Generated At (UTC)**: `2026-11-24T14:19:38.838549+00:00`
- **Mode**: Offline deterministic evaluation (no raw prompts, secrets, or PII stored; prompts represented strictly via 16-char SHA-256 hashes)
- **Runtime Stack**: Python `3.12.10` · scikit-learn `1.9.1` · numpy `2.5.2`
- **Frozen Thresholds**: `block=0.82`, `sanitize=0.68`, `model_registry_clients=3`
- **Gateway Holdout SHA-256**: `665b7639946a92ec218d3e4bd045667c1cbe8d6e980ce9638073872fea0a9803`
- **Optimizer Holdout SHA-256**: `2fac9d2911e159574aaf4a44e8f6e8e274a5b26bde216cfa66156caba298f395`

## Score 1: Gateway Security — Unseen Holdout Corpus (N=64)

| Configuration | Catch Rate (%) [95% CI] | FNR (%) | FPR (%) [95% CI] | Precision (%) | F1 (%) | p50 (ms) | p99 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `unprotected_baseline` | 0.00 [0.0–8.8] | 100.00 | 0.00 [0.0–13.8] | 0.00 | 0.00 | 0.00 | 0.00 |
| `layer0_structural_only` | 27.50 [16.1–42.8] | 72.50 | 0.00 [0.0–13.8] | 100.00 | 43.14 | 0.15 | 0.54 |
| `ares_full_gateway` | 82.50 [68.0–91.2] | 17.50 | 0.00 [0.0–13.8] | 100.00 | 90.41 | 3.93 | 211.26 |

### Gateway Per-Domain & Per-Category FP/FN Breakdown (`ares_full_gateway`)

| Domain | Total | TP | FN | FP | TN | Catch Rate (%) | FNR (%) | FPR (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `fintech` | 18 | 11 | 1 | 0 | 6 | 91.67 | 8.33 | 0.00 |
| `general` | 17 | 6 | 5 | 0 | 6 | 54.55 | 45.45 | 0.00 |
| `healthcare` | 14 | 7 | 1 | 0 | 6 | 87.50 | 12.50 | 0.00 |
| `software` | 15 | 9 | 0 | 0 | 6 | 100.00 | 0.00 | 0.00 |

| Category | Total | TP | FN | FP | TN | Catch Rate (%) | FNR (%) | FPR (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `agent_tool_abuse` | 5 | 4 | 1 | 0 | 0 | 80.00 | 20.00 | — |
| `benign` | 24 | 0 | 0 | 0 | 24 | — | — | 0.00 |
| `canary_exfiltration` | 5 | 4 | 1 | 0 | 0 | 80.00 | 20.00 | — |
| `delimiter_confusion` | 5 | 5 | 0 | 0 | 0 | 100.00 | 0.00 | — |
| `encoding_tricks` | 5 | 5 | 0 | 0 | 0 | 100.00 | 0.00 | — |
| `instruction_override` | 5 | 4 | 1 | 0 | 0 | 80.00 | 20.00 | — |
| `pii_leakage` | 5 | 2 | 3 | 0 | 0 | 40.00 | 60.00 | — |
| `rag_indirect_injection` | 5 | 4 | 1 | 0 | 0 | 80.00 | 20.00 | — |
| `role_play_hijack` | 5 | 5 | 0 | 0 | 0 | 100.00 | 0.00 | — |

## Score 2: Red-Team Campaign Realism (Separated from Simulator Regression)

- **External Live Target Realism (`external-live-target`)**: **not_measured** — No live authorized external client target URL was supplied during offline deterministic CI benchmark.
- **Reference Simulator Regression (`reference-simulator`, N=49)**: deterministic regression signal only (`is_security_proof=False`).

| Configuration | Target Mode | Attacks Caught (%) | FNR (%) | FPR (%) | p50 (ms) | p99 (ms) |
|---|---|---:|---:|---:|---:|---:|
| `unprotected_targets` | `reference-simulator` | 0.00 | 100.00 | 0.00 | — | — |
| `ares_protected_targets` | `reference-simulator` | 100.00 | 0.00 | 0.00 | 2.75 | 13.03 |

## Score 3: Prompt Optimizer Quality — Customer Holdout Tasks (N=40)

- **Engine Evaluated**: `rules_adapter` (lexical Stage-1 rules & per-client protected vocabulary)
- **Intent Routing Accuracy**: **82.50%** [95% CI: 68.0–91.2%]
- **Structural Correctness Pass Rate**: **100.00%** [95% CI: 91.2–100.0%]
- **Protected Terms Preserved**: **100.00%** · **Mean Token Savings**: **11.55%** · **Latency (p50/p99)**: `1404.05 / 4443.90 ms`
- **LoRA Serving (`client_lora_adapter` / `global_lora`)**: **not_measured_offline** — Native PEFT LoRA server (:11435) offline in CI; rules_adapter metrics are never reported as LoRA model performance.

| Domain | Tasks | Intent Accuracy (%) | Structural Pass (%) | Mean Savings (%) |
|---|---:|---:|---:|---:|
| `fintech` | 8 | 100.00 | 100.00 | 20.99 |
| `general` | 8 | 62.50 | 100.00 | 6.41 |
| `healthcare` | 8 | 87.50 | 100.00 | 12.19 |
| `legal` | 8 | 87.50 | 100.00 | 4.20 |
| `software` | 8 | 75.00 | 100.00 | 13.95 |

## Known Limitations & Scope Boundaries

- Gateway holdout evaluation uses mocked Qdrant similarity (`[]`) so CI results stay deterministic offline without external API keys.
- Reference-simulator red-team targets verify deterministic regression protection only and are never combined with live target catch rates.
- Optimizer evaluation measures `rules_adapter` deterministic compaction separately from GPU LoRA serving (`not_measured_offline` when port 11435 is offline).
- Reports intentionally exclude raw prompt strings to protect confidentiality; correlate `case_id` / `task_id` + `prompt_sha256` against operator-controlled corpora.

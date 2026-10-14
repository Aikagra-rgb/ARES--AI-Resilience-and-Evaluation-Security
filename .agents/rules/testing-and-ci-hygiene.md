---
description: Testing and CI hygiene standards for mock isolation, untracked ML artifacts, and compiler warnings
globs: backend/tests/**, src/**/Tests/**, .github/workflows/**
always_on: true
---

# Testing & CI Hygiene Standards

## 1. Cloud Client Mock Isolation
- When unit testing cloud API integrations (e.g., Gemini, OpenAI, Qdrant cloud), always explicitly disable local fallbacks (such as `PREFER_LOCAL_EMBEDDINGS=False` or `ALLOW_LOCAL_FALLBACK=False`) in test fixtures.
- Prevent local daemon discovery (e.g. Ollama localhost probes) from executing unintended HTTP requests that disrupt `assert_called_once()` checks.

## 2. Gitignored ML Artifacts in CI
- Machine learning model weights (`*.joblib`, `*.safetensors`, `*.gguf`) must remain gitignored.
- Tests that verify loaded models or serialized pipelines must never assume pre-existing binary files on disk.
- If `model_path.exists()` evaluates to `False`, the test must train/fit the model on the fly using checked-in test fixtures (e.g., JSONL datasets) before running inference assertions.

## 3. Zero-Warning Compiler Standards
- For C# / .NET projects with `<Nullable>enable</Nullable>`, strictly annotate optional arguments with nullable types (`string?`, `object? = null`) and use null-safe navigation (`request.RequestUri?.AbsolutePath ?? string.Empty`).
- CI must remain free of compiler warning annotations.

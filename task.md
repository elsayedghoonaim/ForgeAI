# ForgeAI API Compliance & CLI Usability Gaps Checklist

- [x] Define and Spawn compliance subagents (`compliance_fixer`, `compliance_critic`, `compliance_reviewer`)
- [x] Implement Dynamic GGUF Chat Templating in `LlamaCppBackend`
  - [x] Extract GGUF `tokenizer.chat_template` from `self._engine.metadata`
  - [x] Support Jinja2 compilation and rendering with native `bos_token` and `eos_token` extraction
  - [x] Implement dynamic high-fidelity fallbacks (Llama-3, ChatML/Qwen/Yi, Llama-2/Mistral) if metadata or Jinja2 is unavailable
- [x] Audit FastAPI OpenAI API Compliance
  - [x] Verify streaming response structure and termination protocols (`data: [DONE]`)
  - [x] Verify OpenAI compatibility across all endpoint parameters and error responses
- [x] Validate CLI Usability & Commands
  - [x] Run diagnostic checks and verify CLI entry points (`profile`, `run`, `batch`, `benchmark`, `doctor`)
- [x] Write and Run Compliance Test Suite
  - [x] Create `tests/test_api_compliance.py` covering templates, API models, and stream formats
  - [x] Assert all tests pass perfectly
- [x] Mitigate Dependency Resolution Loop
  - [x] Identify `resolution-too-deep` conflict caused by manual sub-dependency pins (`starlette` and `uvicorn`) conflicting with the vLLM and FastAPI engine ecosystem requirements
  - [x] Remove the manual `starlette` pin and relax `fastapi>=0.115.0` and `uvicorn>=0.30.0` bounds to ensure mathematical compatibility across the full vLLM/FastAPI dependency tree
- [x] Resolve Jinja2 Test Failures
  - [x] Address `ImportError` on `jinja2` in lightweight/bare test environments by adding `jinja2` to core dependencies in `pyproject.toml` and `requirements/base.txt`

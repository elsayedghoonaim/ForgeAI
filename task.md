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
  - [x] Identify `resolution-too-deep` conflict between `starlette>=1.0.1` and old `fastapi` lower bounds
  - [x] Constrain `fastapi>=0.133.0` in both `pyproject.toml` and `requirements/base.txt` to align with Starlette 1.x support

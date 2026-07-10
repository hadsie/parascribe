# parascribe

OpenAI-compatible transcription server fronting an `onnx-asr` Parakeet TDT model,
on GPU, with correct timestamps. Sits behind a LiteLLM gateway. Open source
(MIT) — write it for strangers: clean README, typed, tested.

`SPEC.md` is the source of truth. If something is not covered there, ask rather
than guessing. After behavior-changing work, note if `SPEC.md` needs updating.

## Hard invariants (never violate)

1. **Timestamps by default.** `response_format=verbose_json` returns segments with
   real `start`/`end` *without* requiring `timestamp_granularities[]`. A known
   LiteLLM bug drops that param in transit; correctness must not depend on it. The
   param, when present, only selects word vs segment vs both.
2. **GPU or fail loudly.** If configured for GPU and `CUDAExecutionProvider` is not
   active at startup, log ERROR and exit non-zero. Never silently fall back to CPU.
   Target hardware is Pascal / sm_61.
3. **Forensic cleanliness.** No transcript text, segment text, original filenames,
   or audio bytes in logs at INFO. Uploads live under a tmpfs `work_dir` and are
   deleted in a `finally`, including on error/timeout. `debug_logging` (default
   off) gates content-exposing diagnostics.
4. **Timestamp offsets must be correct** against the original file's 0s origin
   after chunking. A wrong-but-plausible timestamp is worse than no timestamp.
5. **OpenAI-compatible.** Response shapes match the OpenAI transcription API so
   existing clients/SDKs and LiteLLM work unmodified.
6. **One inference at a time.** Inference is serialized through a single-flight
   queue (one GPU forward pass at a time); concurrent requests wait, they do not
   run in parallel. Model load and eviction run under the same gate. By default a
   single model is loaded once at startup; optional multi-model mode keeps up to
   `max_resident_models` resident, loaded lazily and evicted LRU, with the default
   model always preloaded at startup so the GPU check still fires. See
   `specs/multi-model-registry/spec.md`.

## Conventions

- Python 3.11+, **venv + pip** (not uv). Runtime deps pinned in
  `requirements.txt`; the pinned `onnxruntime-gpu` lives in
  `requirements-gpu.txt` (see SPEC §8). `requirements-dev.txt` for test/lint
  tooling.
- FastAPI + uvicorn. Config via pydantic-settings (env vars).
- Lint/type: **ruff** + **mypy**. Tests: **pytest**, in `tests/` mirroring `src/`.
- Type hints and docstrings on public functions. No dead code, no trailing
  whitespace, files end with a newline.
- `stitch.py` (offset correction + merge) is the crux — heavily unit-tested in
  isolation before any API wiring.

## Workflow rules

- Never `git commit`/`push`/`tag` or open a PR without explicit per-action
  confirmation. Propose the message and wait for a yes.
- No emojis, no em-dashes. No "Generated with Claude Code" / "Co-Authored-By"
  footers in commits or PRs.
- Push back on questionable decisions; ask when a choice has real trade-offs.

# Multi-Model Registry (single + swap/co-resident modes)

## Summary
Today parascribe loads exactly one model at startup and serves it for every
request, ignoring the request's `model` field for routing (invariant #6). This
feature adds an optional multi-model mode: a registry that resolves the request's
`model` to one of an explicit allow-list of models, lazy-loading it on first use,
keeping up to `MAX_RESIDENT_MODELS` loaded in VRAM at once, and evicting the
least-recently-used model when that cap would be exceeded. Single-model mode (the
current behavior) is preserved as the degenerate case: an allow-list of one,
`MAX_RESIDENT_MODELS=1`, no swapping. The single-flight inference gate, response
shapes, stitching, and forensic logging are unchanged. Inference stays serialized
(one GPU forward pass at a time); co-residence buys instant model switching, not
parallelism.

## Motivation
parascribe sits behind a LiteLLM gateway and may need to serve more than one
Parakeet variant (e.g. `parakeet-tdt-0.6b-v3` multilingual alongside an
English-only `v2`, or a future model during a migration window). The reference
system being replaced (Speaches) handles this Ollama-style: dynamic load-on-demand
keyed by the request model, with TTL-based eviction to bound memory. We want that
flexibility without surrendering the three guarantees parascribe deliberately
chose over the Speaches model:

1. **One GPU operation at a time** -- enforced by the existing single-flight gate.
2. **Boot-time GPU fail-loud** (invariant #2) -- a broken GPU/CUDA stack must fail
   at startup, not on the first request hours later. We just lived through why
   this matters (a cuDNN/driver mismatch caught at boot).
3. **Predictable VRAM** -- on an 11GB card two models can OOM on a long file
   (measured ~8GB peak for one model), so the default must not silently risk that.

The design keeps all three: load/evict run under the same gate as inference, the
configured default model is preloaded at startup so the GPU check still fires, and
`MAX_RESIDENT_MODELS` defaults to 1 (pure swap) so VRAM stays deterministic unless
the operator opts into co-residence on a larger card.

## Design

### Mode selection
Mode is inferred from configuration, no separate flag:
- `PARASCRIBE_MODELS` unset/empty -> **single mode**. Identical to current
  behavior: `MODEL_ID` is loaded at startup, the request `model` field is accepted
  but ignored for routing.
- `PARASCRIBE_MODELS` set (comma-separated list of onnx-asr model ids / HF repo
  ids) -> **multi mode**. The request `model` field selects which model serves the
  request and must match a member of the allow-list.

`MODEL_ID` remains the **default / preload** model in both modes. In multi mode it
is implicitly included in the allow-list if not already present, and it is the
model loaded and GPU-verified at startup.

### Config (added to `Settings` in `config.py`)
- `models: list[str] = []` -- env `PARASCRIBE_MODELS`, comma-separated allow-list.
  Empty => single mode. pydantic-settings parses a comma-separated env var into a
  list.
- `max_resident_models: int = 1` -- env `PARASCRIBE_MAX_RESIDENT_MODELS`. Max
  models held in VRAM simultaneously. 1 = swap; >1 = co-resident cache.
- `model_ttl_s: float | None = None` -- env `PARASCRIBE_MODEL_TTL_S`. If set, a
  model idle longer than this is eligible for eviction (see eviction policy). None
  = pinned until evicted by capacity pressure.

### `ModelRegistry` (new, `registry.py`)
A small synchronous object holding loaded `Transcriber`s. It is the only thing the
route talks to for model resolution; it does not know about asyncio or the gate
(the caller holds the gate around every registry mutation).

State:
- `_loaded: dict[str, Transcriber]` keyed by canonical model id.
- `_last_used: dict[str, float]` monotonic timestamp per loaded model.
- `_allowed: set[str]` -- the resolved allow-list (single mode: `{model_id}`).
- references to `settings`.

Public API:
- `resolve(requested: str | None) -> str` -- map a request `model` value to a
  canonical id. Single mode: always returns `settings.model_id` (request ignored).
  Multi mode: returns `requested` if in `_allowed`, else raises
  `UnknownModelError` (route maps to 400). A `None`/empty request in multi mode
  resolves to `settings.model_id` (the default).
- `get(model_id: str, *, now: float) -> Transcriber` -- return the loaded
  Transcriber, loading it first if absent. On a load that would exceed
  `max_resident_models`, evict LRU first (see policy). Updates `_last_used`.
  MUST be called only while the single-flight gate is held.
- `preload() -> Transcriber` -- load `settings.model_id` and return it; called at
  startup. Runs the existing GPU verification in `Transcriber.__init__`.
- `loaded_ids() -> list[str]` -- for the health endpoint.

Eviction policy (all evaluated inside `get`, under the gate -- no background
thread):
1. If `model_ttl_s` is set, drop any loaded model (other than the one being
   requested) whose idle time exceeds the TTL.
2. If adding the requested model would exceed `max_resident_models`, evict
   least-recently-used loaded models until there is room.
3. Eviction = drop all references to the `Transcriber` (its onnx-asr model, VAD,
   and onnxruntime `InferenceSession`s) so VRAM is released. Because eviction
   happens under the gate, an evicted model is provably not mid-inference.

Loading a model lazily runs `Transcriber(settings, model_id=...)`, which performs
the same active-provider GPU check. A lazy load failure raises (route -> 500) but
does not crash the server -- boot already proved the GPU stack works.

### `Transcriber` change (`asr.py`)
`Transcriber.__init__` currently reads `settings.model_id`. Add an optional
`model_id` override parameter so the registry can construct one per model:
`Transcriber(settings, model_id: str | None = None)`, defaulting to
`settings.model_id`. The VAD (`onnx_asr.load_vad("silero", ...)`) is
model-independent; the registry SHOULD load it once and share it across
Transcribers to avoid reloading silero per model (decide during implementation;
see open question).

### `main.py` wiring
- Lifespan: replace `app.state.transcriber = Transcriber(settings)` with
  `app.state.registry = ModelRegistry(settings)` and call `registry.preload()`
  (which loads + GPU-verifies `MODEL_ID`). Keep injecting for tests.
- Route `/v1/audio/transcriptions`: after `gate.acquire()`, resolve and fetch the
  model *inside the gate-held region* so load/evict is serialized with inference:
  ```
  await gate.acquire()
  try:
      model_id = registry.resolve(model)          # 400 on unknown (before any work)
      transcriber = registry.get(model_id, now=time.monotonic())
      ... transcribe ...
  finally:
      await gate.release()
  ```
  `resolve` should run *before* `gate.acquire()` is wasted on a doomed request --
  i.e. validate the model name against the allow-list early (cheap, no GPU), return
  400 if unknown, and only then admit to the gate. Loading (`get`) stays under the
  gate. For streaming, resolve+get happen after acquire and the resolved
  transcriber is passed into `_stream_events` (unchanged signature).
- `/health`: report `mode` (`single`/`multi`), the configured allow-list, and the
  currently-resident model ids, in addition to the existing default-model fields.

### Invariant #6 reword (CLAUDE.md + SPEC.md)
Current: "One model, one inference at a time. Single model loaded once at startup."
New intent: **one inference at a time stays absolute** (single-flight gate over
load, evict, and inference); **the number of resident models is mode-dependent**
(one at startup in single mode; up to `MAX_RESIDENT_MODELS` in multi mode, loaded
lazily and evicted LRU). The default model is always preloaded at startup so the
GPU fail-loud check is preserved regardless of mode.

## Scope Boundaries
- **No concurrent inference.** The single-flight gate is unchanged; co-residence
  removes swap latency, it does not enable parallel transcription. (Parallel/
  multi-stream inference is a separate roadmap item.)
- **No arbitrary model loading.** Only models in the explicit `PARASCRIBE_MODELS`
  allow-list can be loaded; an unknown `model` is a 400. parascribe never downloads
  a model named by an untrusted request outside the allow-list (SEC-02).
- **No hot reconfiguration.** The allow-list is fixed at startup; adding a model
  requires a restart.
- **No background TTL sweeper in v1.** TTL eviction is evaluated at acquire time
  only. Consequence: a model idle past its TTL is not reclaimed until the next
  request arrives. A background sweeper is a possible later enhancement.
- **No per-model diarization/VAD config.** The diarizer is separate and
  model-independent; VAD options are global settings.
- **No automatic VRAM fitting.** `MAX_RESIDENT_MODELS` is operator-set; the server
  does not measure free VRAM and auto-tune the cap.

## Edge Cases and Decisions
- `PARASCRIBE_MODELS` is annotated `Annotated[list[str], NoDecode]` so
  pydantic-settings does not try to JSON-decode the env value; a `mode="before"`
  field validator splits the comma-separated string. Without `NoDecode` the env
  source raises `SettingsError` on a non-JSON value like `a,b,c`.
- TTL eviction is swept on *every* `get` (including cache hits), not only on
  misses, so an idle model is reclaimed by the next request of any kind rather than
  only when a different model is loaded. Capacity (LRU) eviction still happens only
  on a miss, when a new model is about to be added.
- LiteLLM strips the `openai/` routing prefix before forwarding, so the `model`
  value parascribe sees is the bare id (e.g. `istupakov/parakeet-tdt-0.6b-v3-onnx`).
  Allow-list entries are these post-strip ids. Friendly aliasing (`parakeet-v3`) is
  expected to live in LiteLLM's `model_name`, not in parascribe -- unless we add an
  alias map (open question).
- `MODEL_ID` not present in `PARASCRIBE_MODELS`: auto-included as the default;
  it is always loadable and is the startup-preloaded model.
- Lazy-load failure (bad model id, gated repo, transient GPU error): the request
  fails (500) but the server stays up; boot already verified the GPU stack.
- Two concurrent requests for two different uncached models: serialized by the
  gate; the second waits, then may evict what the first just loaded (thrash). This
  is accepted and bounded by LRU; operators size `MAX_RESIDENT_MODELS` to their
  traffic.
- Eviction safety: an evicted model is never mid-inference because eviction only
  happens inside `get`, which runs under the single-flight gate.
- onnxruntime may not promptly return CUDA memory to the OS on session
  destruction. Eviction drops all references (and may need an explicit
  session close + gc); VRAM reclamation is best-effort and verified during
  implementation. If a card cannot fit even transient overlap during a swap
  (old not yet freed while new loads), document the minimum headroom.

## Acceptance Criteria
- [ ] With `PARASCRIBE_MODELS` unset, behavior is identical to today: one model
      loaded at startup, request `model` ignored for routing, GPU verified at boot.
- [ ] With `PARASCRIBE_MODELS` set, a request whose `model` is in the allow-list is
      served by that model.
- [ ] A request whose `model` is not in the allow-list returns 400 before any GPU
      work is admitted to the gate.
- [ ] A request with empty/missing `model` in multi mode is served by `MODEL_ID`
      (the default).
- [ ] `MAX_RESIDENT_MODELS=1`: requesting a second model evicts the first; the
      registry reports exactly one resident model afterward.
- [ ] `MAX_RESIDENT_MODELS=2`: requesting a second model keeps both resident (no
      eviction); requesting a third evicts the least-recently-used.
- [ ] `MODEL_ID` is preloaded and GPU-verified at startup in both modes; a broken
      GPU stack fails startup non-zero (invariant #2 preserved).
- [ ] Model load and eviction occur only while the single-flight gate is held (no
      inference runs concurrently with a load/evict).
- [ ] With `MODEL_TTL_S` set, a model idle longer than the TTL is evicted on the
      next `get` that observes it.
- [ ] Eviction drops all references to the Transcriber so the model is no longer
      reported as resident.
- [ ] `/health` reports the mode, the allow-list, and the resident model ids.
- [ ] Single-flight serialization, response formats, streaming, and forensic
      logging are unchanged by this feature.

## Files
- `src/parascribe/registry.py` -- `ModelRegistry`, `UnknownModelError` (new).
- `src/parascribe/asr.py` -- `Transcriber` `model_id`/`vad` overrides; `build_vad`.
- `src/parascribe/config.py` -- `models` / `max_resident_models` / `model_ttl_s`
  settings + comma-separated `PARASCRIBE_MODELS` parsing (NoDecode + validator).
- `src/parascribe/main.py` -- registry wiring in lifespan, route model resolution
  + gate-held load, `/health` mode/models/loaded fields.
- `.env.example` -- multi-model env vars.
- `tests/test_registry.py` -- registry unit tests (new).
- `tests/test_config.py` -- settings parsing tests (new).
- `tests/test_api.py` -- multi-mode 400/200, health mode reporting.

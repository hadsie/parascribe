# Roadmap

Post-MVP ideas, not committed work. The current design is **one inference at a
time** (single model by default, or an optional multi-model allow-list — see
`specs/multi-model-registry/spec.md`), fronted by a LiteLLM gateway; several
items below would extend or relax that, and each notes the constraint that makes
it non-trivial.

## Concurrency & scheduling

### Serve multiple models — shipped
Implemented in `registry.py` (see `specs/multi-model-registry/spec.md`): the
OpenAI `model` param routes against a configured allow-list, with both
co-resident (`max_resident_models > 1`) and lazy-load + LRU-swap modes, plus an
optional idle TTL. A **LiteLLM fleet** (multiple single-model instances, gateway
routes by name) remains the zero-server-code alternative. VRAM on the production
card is still the binding constraint: two Parakeets ≈ 10-11GB (fine on a 24GB
card, tight on the 11GB 1080 Ti), which favors swap mode there. The open
validation work is tracked under "Validated multi-model support" below.

### Richer request queuing
Today the inference gate serializes to one in-flight call and admits up to
`max_queue` waiters, returning **503** beyond that. Possible extensions:
- An **async job API** (submit → job id → poll / webhook) so a client isn't
  holding an HTTP connection for the length of a long run (CPU diarization of a
  2h file is ~tens of minutes). This is the big one for long-file UX.
- Queue-position / estimated-wait feedback, and configurable wait-vs-reject
  behavior instead of a hard 503.

### Resource-aware scheduling (GPU vs CPU lanes)
ASR runs on the **GPU**; CPU diarization runs on the **CPU** — they don't contend
for the same hardware. The current single-flight gate serializes *everything*, so
a GPU-only (no-diarization) request waits behind an in-flight CPU diarization even
though the GPU is idle. A resource-aware scheme (separate GPU and CPU lanes, or
letting a GPU-only request pass a CPU-bound one) would let those overlap. Needs
care: still one GPU forward pass at a time, and bounded total work.

## Diarization

### GPU diarization
CPU pyannote is slow (~0.4x the audio duration; ~47 min for a 2h file). Two
routes to GPU:
- **Sortformer-ONNX** — NVIDIA's end-to-end diarizer runs on our existing
  onnxruntime stack (no torch, GPU even on Pascal). Deferred because it needs a
  hand-port of NVIDIA's streaming Arrival-Order-Speaker-Cache loop; see
  `SPEC.md` §15.7a for the spike findings and the captured mel recipe. A
  NeMo/Scriberr reference output would de-risk the port.
- **Process-isolated pyannote-GPU** — run pyannote in its own process/venv so its
  CUDA stack can't collide with onnxruntime-gpu, then talk over a local socket.

### Speaker-count control
Expose `min_speakers` / `max_speakers` (only exact `num_speakers` today), and
optionally post-filter the low-mass phantom speakers pyannote spawns on poor audio
(observed 8 labels for a 5-person meeting; `num_speakers=5` collapses them).

### Diarized streaming
Diarization currently forces a non-streamed response (clustering needs the whole
file). Could stream ASR text deltas as they finalize, then attach speaker labels
in the terminal `done` event.

## Models & accuracy

### Validated multi-model support
`model_id` already swaps any compatible onnx-asr model, but the pipeline has
Parakeet-specific assumptions (timestamp structure, subword tokenization,
`language` being a no-op). Validate + document a compatibility matrix. Near-term:
`parakeet-tdt-0.6b-v2` is a drop-in English-accuracy bump. Canary adds an accuracy
ceiling + translation but is an attention model (different timestamp support).
The current leaderboard's WER leaders are cloud APIs or non-ONNX speech-LLMs — out
of stack without a rewrite.

### Translation
Canary exposes `target_language`; a translation path is currently a non-goal but
would slot in if a Canary backend is supported.

## Operability & publishing

### Cancellable long requests / clean shutdown
Inference runs in a worker thread that can't be interrupted, so a long request
can't be cancelled and blocks graceful shutdown (`kill -9` required). Running
inference in a **subprocess** would make it cancellable and unblock shutdown.

### Metrics / observability
A metrics endpoint (e.g. Prometheus: in-flight, queue depth, decode/inference
latency, error counts) to complement the structured request logs.

### CI + container image
GitHub Actions running ruff / mypy / pytest on PRs, and a Dockerfile (bundling
ffmpeg + the pinned onnxruntime-gpu) for one-command deployment.

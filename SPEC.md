# SPEC: parascribe — OpenAI-compatible Parakeet ASR server

Source of truth for the build. The hard part is **timestamp-offset correctness
across chunks** (§6); everything else is plumbing. The onnx-asr API was inspected
empirically before the schema was designed; the resolved findings are in §5.1.

---

## 1. What this is

A thin, self-hostable HTTP server exposing the OpenAI
`/v1/audio/transcriptions` API in front of an `onnx-asr` model (NVIDIA Parakeet
TDT 0.6B v3, the `istupakov/parakeet-tdt-0.6b-v3-onnx` ONNX conversion). It exists
because existing servers each fail one requirement: Speaches refuses
`verbose_json` for Parakeet (no timestamps), and the Go `achetronic/parakeet`
server is CPU-only. We need **timestamps AND GPU**, behind a LiteLLM gateway.

Published open-source (MIT). Write for strangers: clean README, typed,
tested, documented config.

## 2. Phase scope

**Phase 0 (this build):** transcription with correct word/segment timestamps, on
GPU, OpenAI-compatible, long-file chunking, optional video input, forensic temp
handling, auth, health, serialized inference. Streaming is included but
**lower priority** (§10) — land non-streaming first and complete.

**Phase 1 (now being designed — see §15):** speaker diarization + ASR/diarization
alignment via pyannote.audio. Phase 0 already carries the **optional `speaker`
field** (`null`) on every segment (§7); Phase 1 populates it. Opt-in per request.

**Non-goals:** translation endpoint, TTS, request-driven model-download
orchestration (multi-model serving is supported, but only from an explicit
allow-list, not arbitrary downloads named by a request), web UI. Multi-model
serving (single / swap / co-resident modes) is specified in
`specs/multi-model-registry/spec.md`.

## 3. Hard invariants

The six invariants live in `CLAUDE.md`. Summary: timestamps by default; GPU or
fail loudly; forensic-clean logging + temp cleanup; correct offsets after
chunking; OpenAI-compatible shapes; one inference at a time (single or
multi-model; see `specs/multi-model-registry/spec.md`).

## 4. Tech stack

- Python 3.12, **FastAPI** + **uvicorn**.
- **venv + pip** for environment and packaging (not uv). Pinned deps in
  `requirements.txt`; dev/test tooling in `requirements-dev.txt`.
- **`onnx-asr`** for inference, VAD, and timestamps. Requires `huggingface_hub`
  for the model download.
- **`onnxruntime-gpu`** — pinned to a Pascal-compatible build (§8). CPU
  `onnxruntime` for M1 dev.
- **Silero VAD** via onnx-asr's built-in `load_vad("silero")` — not a separate
  integration (see §5.1 findings).
- **ffmpeg** via subprocess for audio decode and (optional) video audio-track
  extraction.
- **pytest** for tests; **ruff** + **mypy** for lint/type.
- Config via **pydantic-settings** (env vars).
- License: **MIT**.

## 5. Inference layer

### 5.1 onnx-asr API — resolved findings (empirical)

Inspected on the M1 (CPU provider) against `istupakov/parakeet-tdt-0.6b-v3-onnx`.
Versions: `onnx-asr 0.11.0`, `onnxruntime 1.27.0`; `huggingface_hub` is required
for the model download.

**Loading & providers.**
`onnx_asr.load_model("istupakov/parakeet-tdt-0.6b-v3-onnx", providers=[...])`
accepts the HF repo id directly (the builtin alias `nemo-parakeet-tdt-0.6b-v3`
resolves to the same repo). **The provider must be explicit**: with no
`providers=`, onnxruntime auto-selects, and on the M1 it picks CoreML, which
*crashes* on this model's external-data weights. CUDA form:
`[("CUDAExecutionProvider", {"device_id": N})]`.

**VAD is built in.** `onnx_asr.load_vad("silero", providers=[...])`. `VadOptions`:
`threshold`, `neg_threshold`, `min_speech_duration_ms`, `max_speech_duration_s`,
`min_silence_duration_ms`, `speech_pad_ms`, `batch_size`.

**Adapter chaining (ORDER MATTERS).**
`model.with_vad(vad).with_timestamps()` → a `TimestampedSegmentResultsAsrAdapter`
whose `recognize()` returns a **generator** of `TimestampedSegmentResult(start,
end, text, timestamps, tokens, logprobs)`. The reverse order
(`.with_timestamps().with_vad()`) drops timestamps back to text-only segments.

**Offsets, tokens, units.**
- Segment `start`/`end` are **absolute** (the library owns VAD chunking + segment
  offsetting). Per-token `timestamps` are **local to each segment** (restart at
  0.0) — `stitch.py` adds `segment.start` per token (§6).
- `tokens` are SentencePiece **subwords**; a new word begins at a token with a
  leading space, punctuation attaches to the prior word.
- Timestamps are seconds, quantized to 0.08 s (TDT 80 ms frame).
- `logprobs` are per-token (we expose per-segment `avg_logprob` = mean).
- `recognize` accepts a PCM wav path **or** a numpy float32 array + `sample_rate`.

**Recognize options.** `RecognizeOptions` exposes `language`, `target_language`,
`pnc`, but `language` is documented "only for Whisper and Canary models" and is a
**no-op for Parakeet TDT** (verified: `language="en"` on French audio still
produced French). See §7 decision #4. `target_language` (translation) is a
non-goal and not exposed.

### 5.2 Model loading

- Load **once** at startup (lifespan/startup), hold the instance. No lazy-load,
  no TTL — dedicated single-model service.
- GPU pinning via the CUDA execution provider with a configurable `device_id`:
  ```python
  providers = [("CUDAExecutionProvider", {"device_id": settings.gpu_device_id})]
  model = onnx_asr.load_model(settings.model_id, providers=providers).with_timestamps()
  ```
- At startup, query the loaded session's active providers; if GPU was requested
  but `CUDAExecutionProvider` is absent/inactive, log ERROR and exit non-zero
  (invariant #2).

### 5.3 Concurrency — serialized single-flight inference (decision #3)

ONNXRuntime `Run` on a shared session under concurrent load contends on the GPU.
This is a dedicated single-GPU service, so **inference runs one request at a
time**:

- Guard the transcribe path with an `asyncio` queue or lock so exactly one
  request's chunk loop holds the model at any moment. Concurrent requests **wait**
  in FIFO order; they do not run in parallel.
- The wait happens off the event loop's critical path — run the blocking
  inference in a worker thread (`run_in_executor`/`anyio.to_thread`) while the
  lock is held, so the server stays responsive to `/health` and to accepting
  uploads.
- Document the effective concurrency (1 in-flight transcription) in the README.
  A bounded wait queue is acceptable; if depth is exceeded, return **503**.

## 6. The crux: long-file handling and timestamp stitching

> **Per the §5.1 empirical findings.** onnx-asr's
> built-in VAD adapter already does the VAD chunking and emits **globally
> absolute segment offsets**, as a generator. That removes most of what we
> planned to build by hand. The remaining crux is narrower but still real:
> **token-timestamp offsetting + subword→word grouping**.

Input files may be **2+ hours**. A single forward pass (`with_timestamps()`
alone) is unsafe (memory, accuracy). Use the VAD path, which is the same
VAD-cut-at-silence strategy WhisperX/Scriberr use, implemented inside the library:

```python
adapter = model.with_vad(vad).with_timestamps()   # ORDER MATTERS (§5.1)
for seg in adapter.recognize(pcm_or_wav):          # generator
    ...
```

Pipeline:

1. **Decode** input to 16 kHz mono PCM via ffmpeg (Parakeet expects 16k mono;
   onnx-asr reads a PCM wav path or a numpy float32 array). `media.py`.
2. **VAD + transcribe (library):** `with_vad(vad).with_timestamps().recognize()`
   yields a generator of `TimestampedSegmentResult(start, end, text, timestamps,
   tokens, logprobs)`. VAD behavior comes from `VadOptions` mapped from config:
   `max_chunk_s → max_speech_duration_s`, `vad_threshold → threshold`,
   `chunk_overlap_s → speech_pad_ms`. **Segment `start`/`end` are already absolute
   — the library owns segment offsetting and chunk packing.**
3. **Offset-correct token timestamps (THE CRITICAL STEP, `stitch.py`):** per
   segment, the token `timestamps` are **local** (restart at 0). Compute
   `token_global = segment.start + token_local`. Off-by-one here silently
   corrupts all word timing.
4. **Group subwords into words:** `tokens` are SentencePiece subwords; a new word
   begins at a token with a **leading space**, punctuation attaches to the prior
   word. Emit `words[]` with `word.start` = first sub-token global ts, `word.end`
   = last sub-token global ts.
5. **Assemble:** map each segment to the OpenAI segment shape with sequential
   `id`, absolute `start`/`end`, `text`, `speaker: null`, and `avg_logprob` =
   mean of the segment's token `logprobs` (the Whisper/OpenAI-style raw mean
   log-probability, not exponentiated). Concatenate text.
6. Return unified `verbose_json` (or the requested format).

`chunk_overlap_s` (decision #2) defaults to **0**; it maps to VAD pad, not an
overlap-dedup step. There is no cross-chunk text dedup to do because the library
cuts at silence — that whole class of bug is gone.

**The token-offset + word-grouping logic must be unit-tested in isolation** (§11)
with synthetic `TimestampedSegmentResult`-shaped inputs. It is the part most
likely to be subtly wrong and hardest to eyeball in real output.

The library's generator is consumed directly for streaming (§10) — no separate
chunk-generator to build.

`duration` (response field) is taken from the decoded PCM length (samples / 16000)
or ffprobe — state which in code; it must reflect the full original media, not
the summed speech regions.

## 7. API contract

### `POST /v1/audio/transcriptions` (multipart/form-data)

Params (OpenAI-compatible):
- `file` (required) — audio (or video, if `enable_video`). **URL input
  (parascribe extension):** when `enable_url_fetch=true`, if the uploaded content
  is itself a single `http`/`https` URL (under 2 KB, no internal whitespace), the
  server fetches that URL instead of decoding the bytes as audio. The URL must
  ride as the *file content* — not a separate field — because OpenAI has no URL
  param and a fronting LiteLLM gateway drops non-standard form fields, whereas it
  forwards `file` bytes intact. Detection cannot misfire on real media (it has
  sub-0x20 bytes; a URL is clean ASCII). Fetching is an SSRF surface: only
  `http`/`https`, redirects are not followed, and with an empty
  `url_fetch_allowlist` every resolved address must be public
  (private/loopback/link-local/metadata → 400); a non-empty allowlist restricts to
  those exact hosts. The fetched body obeys the same `max_upload_mb` cap (→ 413).
  With `enable_url_fetch=false` a URL upload is just decoded as audio and 400s. See
  `fetch.py`.
- `model` (required) — in single mode accepted for compatibility (one model is
  served); in multi mode it selects the model from the allow-list (unknown -> 400).
- `response_format` — `json` | `text` | `verbose_json` | `srt` | `vtt`.
  Default `json`. **`verbose_json` always includes segments** (invariant #1).
- `timestamp_granularities[]` — `segment` and/or `word`. Optional refinement only;
  absence does not suppress segment timing.
- `language` — optional ISO hint (e.g. `fr`). **Decision #4 (resolved):** the API
  accepts and forwards it, but **Parakeet TDT v3 ignores it** and auto-detects.
  onnx-asr's `RecognizeOptions.language` applies "only for Whisper and Canary
  models"; verified empirically that `language="en"` on French audio still
  transcribes French. So it is effectively a no-op like `temperature`/`prompt`
  with the current model, but harmless to pass (would take effect if a
  Whisper/Canary backend were configured). Never 500 on its presence.
- `stream` — bool; see §10.
- `temperature`, `prompt` — accept and ignore gracefully if unsupported; never
  500 on their presence.

**`verbose_json` response shape:**
```json
{
  "task": "transcribe",
  "language": "fr",
  "duration": 6963.67,
  "text": "full transcript ...",
  "segments": [
    { "id": 0, "start": 0.00, "end": 4.20, "text": "...", "speaker": null, "avg_logprob": -0.12 }
  ],
  "words": [ { "word": "...", "start": 0.00, "end": 0.32 } ],
  "usage": { "type": "tokens", "input_tokens": 36000, "output_tokens": 1342, "total_tokens": 37342 }
}
```
- `usage` is present on the `json` and `verbose_json` bodies (and the streaming
  `transcript.text.done` event); see **Usage / billing** below. `text`/`srt`/`vtt`
  have nowhere to carry it and omit it.
- Populate only fields actually available from `onnx-asr`. Each segment carries
  `avg_logprob` (the mean of its token log-probabilities, Whisper/OpenAI-style raw
  log-prob — not exponentiated). Whisper-only fields not produced here (`seek`,
  `tokens`, `compression_ratio`, `no_speech_prob`, `temperature`) are omitted.
- **`speaker` present on every segment**: an opaque label (`SPEAKER_NN`) when
  diarization is requested, otherwise `null`. Matches OpenAI's emerging
  `diarized_json` shape.
- `srt`/`vtt` are formatted from the same offset-corrected segments.

### Usage / billing

Responses report an OpenAI `tokens`-type `usage` object so a fronting LiteLLM
gateway can track spend (without it LiteLLM records 0 tokens). Parakeet is not a
token-billed model, so the counts are configurable rather than fixed. Three
components each contribute `round(count(unit) * multiplier)`:

- **Audio input** (`audio_input_usage_*`) → its configurable `field` (default
  `input_tokens`). Default `file_duration * 10` mirrors OpenAI's audio-token
  accounting (`gpt-4o-transcribe` bills ~10 audio tokens/sec of *duration*). Unlike
  the ASR path this is **not** speech-gated: it bills silence too, matching OpenAI.
  Set the multiplier to 0 to disable, or `field=output_tokens` to fold it into a
  single combined count.
- **Transcription** (`transcription_usage_*`) → `output_tokens`. Always applied.
- **Diarization** (`diarization_usage_*`) → `output_tokens`, only when diarization
  actually ran for the request.

`unit` is one of `token` (the real ASR subword count), `word`, `segment`, `char`,
or `file_duration` (seconds). All are **deterministic**: the same input always
bills the same, so responses stay reproducible. Multipliers accept fractions.

Two billing philosophies the config spans:

- **OpenAI parity (default):** audio `input_tokens = duration * 10`, plus text
  `output_tokens` from the real subword count. Familiar numbers; over-bills silence
  exactly as OpenAI does.
- **Cost-faithful:** the ASR path is VAD-gated, so real GPU cost tracks *speech*,
  not wall-clock duration. Disable audio input (`multiplier=0`) and bill by
  `token`/`word`; silence becomes nearly free, matching real compute.

Diarization runs on the whole file but its dominant (embeddings) stage is
speech-gated too, so it is billed as a multiple of the same transcript count. The
default `diarization_usage_multiplier=5.0` (with `unit=token`) encodes the derived
CPU-bound cost: a diarized request's `output_tokens` is ~6x a transcription-only
one (transcription `* 1` + diarization `* 5`), reflecting ~90s GPU vs ~1h CPU
weighted at a ~8x GPU:CPU per-second cost. Re-derive the multiplier if diarization
moves to GPU (it drops toward ~2).

### `GET /health`

200 with `{status, model_id, device, provider_active}` once the model is loaded
and GPU verified. Used by systemd/LiteLLM liveness.

### Errors

- Missing/unreadable/undecodable file → **400**, not 500.
- Auth failure → **401**.
- Unsupported `response_format` value → **400** with the allowed list.
- Oversized upload (> `max_upload_mb`) → **413**.
- Inference queue saturated (if a bounded wait queue is used) → **503**.

## 8. Pascal / ONNX Runtime constraint (easy to get wrong)

Target hardware includes **GTX 1080 Ti (Pascal, sm_61)**. Some prebuilt
`onnxruntime-gpu` wheels drop old compute capabilities, forcing CPU fallback. The
deployment already runs `onnx-asr` on GPU on this exact card via Speaches, so a
working wheel exists.

- **Pin `onnxruntime-gpu`** to a version verified to include sm_61 / CUDA provider
  support. Document the pinned version and the CUDA/cuDNN it expects in the README
  and `requirements.txt`. Do not float the dependency.
- **Match the CUDA userspace wheels to the host driver.** The bundled
  `nvidia-*-cu12` wheels ship a specific CUDA minor; the host NVIDIA driver runs an
  equal-or-older runtime, never a newer one. A 550-series driver (CUDA 12.4
  ceiling) fails at cuDNN init (`CUDNN_STATUS_EXECUTION_FAILED_CUDART`) against
  12.9 wheels. Pin the wheels to the oldest deploy driver's CUDA line (12.4);
  newer drivers run them via backward compatibility.
- Provide `scripts/check_gpu.py` that loads the model and prints the active
  execution provider, so a deployer can confirm GPU engagement in one command.
- The startup GPU check (invariant #2) is what catches a bad wheel in production.

## 9. Forensic / security requirements

- **Temp files:** received uploads and extracted audio/video written only under a
  configurable `work_dir` defaulting to a tmpfs path (`/run/parascribe`). Delete
  every temp artifact in a `finally`, including on error/timeout.
- **Content-free logging by default:** never log transcript text, segment text,
  original filenames, or audio bytes at INFO. `debug_logging` (default **off**)
  may enable verbose diagnostics, with a clear warning it exposes content. Default
  INFO logs: request id, duration processed, chunk count, latency — no content.
- **Auth:** static bearer token via `Authorization: Bearer`, read from
  `api_key` / `api_key_file`. Constant-time compare. If no key configured, log a
  warning (acceptable behind a trusted network boundary, but loud).
- **Bind/host configurable**; README documents running behind the network
  boundary (Tailscale/internal) like the rest of the fleet.
- **ffmpeg invocation:** explicit args, never `shell=True`; bound input size;
  treat media parsing as an attack surface.

## 10. Streaming (`stream=true`) — lower priority

Land and verify non-streaming first; streaming is an additive pass on the §6
generator.

- Emit **Server-Sent Events** as chunks complete: incremental
  `transcript.text.delta` events as chunk text finalizes, terminated by a
  `transcript.text.done` carrying the full text; when `verbose_json`, the final
  event includes the assembled segments. **This segment-bearing final event is a
  parascribe extension, not part of the OpenAI streaming spec** — document it as
  such (decision #5).
- This is **progressive output streaming, not realtime input**: VAD segmentation
  needs the whole decoded file first. We stream results as chunks finish, which is
  also the practical fix for 2-hour requests tripping client/proxy timeouts.
- **`stream=true` with a non-streamable `response_format`** (`srt`/`vtt`/`text`):
  silently ignore `stream`, log a warning, and return the normal non-streamed
  response (decision #6). Streaming applies to `json`/`verbose_json` only.
- Non-streaming remains the default and must work standalone.

## 11. Testing / acceptance criteria

Unit:
- **Offset stitching** (highest priority): feed synthetic per-chunk results with
  known offsets; assert merged segment **and word** times are monotonic, start at
  ~0, and each equals chunk-local time + offset. Cover a region exceeding
  `max_chunk_s`. Cover overlap dedup when `chunk_overlap_s > 0`.
- Response formatting: `verbose_json` / `srt` / `vtt` / `text` / `json` from a
  fixed segment list.
- Auth: 401 without / with-wrong key; 200 with correct.
- Error mapping: 400 on a non-media file; 413 oversize; 400 on bad
  `response_format`.
- Concurrency: two overlapping requests serialize (one completes before the other
  starts inference); queue-saturation path if bounded.
- Streaming-format fallback: `stream=true` + `srt` returns non-streamed + warns.

Integration (mark GPU-only, skip if absent):
- Transcribe a short real clip → segments present, `start[0]` ≈ 0, increasing.
- Transcribe a synthetic long (> `max_chunk_s`) file → no gaps/overlaps at chunk
  seams, final `end` ≈ file duration.
- `/health` returns `provider_active: true` on GPU hardware.

Include a couple of small fixture clips (or generate via ffmpeg/espeak in a test
helper) — keep large media out of the repo.

## 12. Repo layout

> Updated to the as-built layout. The hand-rolled `chunking.py` was dropped:
> onnx-asr owns VAD chunking (§6), so its work splits between `asr.py` (VAD
> config + transcribe generator) and `media.py` (ffmpeg decode).

```
parascribe/
  requirements.txt        # shared runtime deps (no onnxruntime variant)
  requirements-gpu.txt     # -r requirements.txt + pinned onnxruntime-gpu (deploy)
  requirements-cpu.txt     # -r requirements.txt + onnxruntime (dev / M1)
  requirements-dev.txt     # -r requirements-cpu.txt + pytest, ruff, mypy, httpx
  pyproject.toml          # tool config (ruff/mypy/pytest); deps via requirements*
  .env.example            # documented config
  README.md               # what/why, venv install, config table, Pascal note, curl examples
  LICENSE                 # MIT
  CLAUDE.md               # invariants + conventions
  SPEC.md                 # this document (single source of truth)
  src/parascribe/
    __init__.py
    main.py               # FastAPI app, lifespan load, routes, inference gate, SSE
    config.py             # pydantic-settings
    asr.py                # onnx-asr load + GPU verify + VAD transcribe generator
    media.py              # ffmpeg decode to 16k mono + video detection
    stitch.py             # token offset + subword->word grouping + assembly (the crux)
    formats.py            # json/verbose_json/srt/vtt/text + SSE serialization
    auth.py               # bearer, constant-time
  scripts/
    check_gpu.py          # prints active execution provider; non-zero if CUDA requested-but-absent
  tests/
    test_stitch.py        # the crux
    test_media.py         # ffmpeg decode
    test_formats.py
    test_auth.py
    test_api.py           # routes, streaming, video gating, gate (model faked)
    test_integration.py   # real model, opt-in (PARASCRIBE_RUN_MODEL_TESTS), gpu-marked
  deploy/
    parascribe.service    # hardened systemd unit (tmpfs RuntimeDirectory, GPU allow-list)
```

## 13. Config (env / pydantic-settings) — at minimum

`model_id` (default `istupakov/parakeet-tdt-0.6b-v3-onnx`), `gpu_device_id`,
`host`, `port`, `api_key` / `api_key_file`, `work_dir` (tmpfs default
`/run/parascribe`), `max_chunk_s`, `chunk_overlap_s` (default 0), `vad_threshold`,
`max_upload_mb`, `enable_video`, `debug_logging` (default false),
`default_language` (optional). Provide a `.env.example`.

## 14. How to proceed (build order)

1. Scaffold repo + `requirements.txt` / `requirements-dev.txt` / `pyproject.toml`;
   create a venv; get `onnx-asr` + `onnxruntime-gpu` installing.
2. **Do §5.1** — inspect the real `TimestampedResult` shape (and the language-hint
   behavior for decision #4) before designing schemas. (Findings now in §5.1.)
3. Implement `asr.py` (load + single-chunk transcribe) and `scripts/check_gpu.py`;
   verify GPU engagement conceptually (real sm_61 verification happens on the
   deployer's hardware).
4. Implement `chunking.py` and `stitch.py`; **write `test_stitch.py` alongside**
   and make it pass before wiring the API.
5. Implement `formats.py`, then `main.py` routes (non-streaming first) with the
   serialized inference queue (§5.3), `auth.py`, `/health`, error mapping.
6. Add streaming (§10) using the chunk generator — after non-streaming is done.
7. Add optional video extraction behind `enable_video`.
8. Forensic pass: audit logging for content leakage; confirm temp cleanup in
   `finally`; default `debug_logging=false`.
9. README with config table, the Pascal/onnxruntime pinning note, venv install,
   curl examples (json + verbose_json + streaming), and the systemd unit.

Keep Phase-0 honest: real timestamps, GPU, OpenAI-compatible, long files,
forensic-clean, serialized inference. Schema-ready for diarization; do not build
it.

---

## 15. Phase 1: speaker diarization (pyannote.audio)

Opt-in "who said what". Phase 0 already emits global word/segment timestamps and
a `null` `speaker` field; Phase 1 adds a diarizer, aligns its speaker turns onto
those timestamps, and fills in `speaker`.

### 15.1 Decisions (resolved with the user)

- **Engine: pyannote.audio** (PyTorch). Chosen over an ONNX hand-roll after the
  spike found onnx-asr exposes only undocumented speaker embeddings with no
  clustering/assignment (see §5.1-style notes below). pyannote is the mature,
  accurate path and handles overlapping speech.
- **Fully local is a hard requirement; model licensing is flexible.** pyannote
  models run entirely locally; the only non-local step is a one-time, license-
  accepted download. Deploys must work offline once models are cached.
- **API: opt-in `diarization=true`**, automatic speaker count with an optional
  `num_speakers` hint. Populates the existing per-segment `speaker` field.

### 15.2 Spike findings (onnx-asr, why not ONNX-native)

onnx-asr provides VAD/segmentation (Silero, pyannote-segmentation-3.0) and an
**undocumented** `WespeakerEmbeddings` (no default model, loader expects a
bare-`*.onnx` repo that standard pyannote repos don't satisfy), but **no
clustering and no end-to-end diarization**. A pure-ONNX path would mean owning
segmentation→embed→cluster→assign on top of an undocumented API. Rejected in
favor of pyannote.audio. (`sherpa-onnx` is a viable ONNX alternative if we ever
want to drop the PyTorch dependency.)

### 15.3 Pipeline

1. **Decode** input to 16 kHz mono via `media.py` (reused; decode once, share the
   array between ASR and diarization).
2. **ASR** as today → global word + segment timestamps (`stitch.py`).
3. **Diarize** with the pyannote pipeline → speaker turns: a list of
   `(start, end, speaker_label)` (pyannote `Annotation.itertracks`), labels like
   `SPEAKER_00`. Supports `num_speakers` or `min/max_speakers`.
4. **Align (the Phase-1 crux):** assign each ASR **word** to the speaker turn with
   maximum temporal overlap; a segment's `speaker` is the majority vote of its
   words' speakers (ties → earliest-overlapping turn). Words with no overlapping
   turn (e.g. a word inside a region pyannote marked non-speech) inherit their
   segment's majority speaker. This mirrors WhisperX's word-speaker assignment.
5. **Emit**: `speaker` on each segment (and optionally each word — see 15.5).

Alignment is pure and **unit-tested in isolation** like `stitch.py`: synthetic
turns + words → expected labels, covering overlap, ties, gaps, and 0/1-speaker.

### 15.4 Concurrency, device, streaming

- **Runs under the same single-flight gate** (§5.3), sequential with ASR — one
  request's GPU work at a time.
- **`diarization_device` is configurable.** Default follows the ASR provider
  (GPU), but on the 11GB 1080 Ti, ASR already holds ~8GB; pyannote adds a PyTorch
  CUDA context. If VRAM is tight, set diarization to **CPU** (its models are small
  and CPU diarization is acceptable for non-realtime use). The deploy doc must
  call this out.
- **Streaming is incompatible with diarization** (clustering needs the whole
  file). `diarization=true` with `stream=true` → ignore `stream`, log a warning,
  return the non-streamed response (same pattern as the srt/vtt case, decision #6).

### 15.5 API

- `diarization` — bool, default false. When true, segments carry real `speaker`
  labels (`SPEAKER_00`, ...) instead of `null`.
- `num_speakers` — optional int hint; omitted → automatic. (May also expose
  `min_speakers`/`max_speakers`.)
- `speaker` on **words**: include when `verbose_json` + word granularity +
  diarization are all on (forward-compat with OpenAI `diarized_json`); otherwise
  segment-level only. Labels are opaque `SPEAKER_NN` — **no speaker naming/
  identification** (out of scope).
- If `diarization=true` but diarization deps/models are unavailable →
  **400/503 with a clear message**, never a silent fall back to no-speaker output
  (fail loudly).

### 15.6 Config additions

`enable_diarization` (gates the feature/deps at startup), `diarization_model`
(default `pyannote/speaker-diarization-3.1`), `diarization_device`
(`cuda`|`cpu`, default = follow ASR), `hf_token` / `hf_token_file` (for the
gated download; not needed once cached), and a documented offline/local-model
path. Loaded once at startup like the ASR model when `enable_diarization=true`.

### 15.7 Dependencies & deploy

- pyannote.audio + PyTorch are **heavy and optional** → a separate
  `requirements-diarization.txt` (installed only when diarization is wanted), kept
  out of the base GPU install.
- **Models are gated**: the operator must accept the `pyannote/speaker-
  diarization-3.1` and `segmentation-3.0` licenses on HuggingFace and provide a
  token for the first download; thereafter it runs offline from cache.
- **Pascal/torch verification** (mirrors the onnxruntime sm_61 concern): confirm
  the PyTorch CUDA build runs on the 1080 Ti, or run diarization on CPU there. The
  3090 validates functionality; the 1080 Ti is the deploy risk.

### 15.7a Sortformer-ONNX (investigated, deferred)

NVIDIA Sortformer (end-to-end NeMo diarizer) was spiked as a stack-native
alternative: its community ONNX export *does* run on onnxruntime (no torch, GPU
on Pascal), which would be a better fit than pyannote. Deferred because using it
properly requires porting NVIDIA's streaming Arrival-Order-Speaker-Cache loop
(fixed 124-frame chunks + spkcache/fifo state management + cache compression) by
hand from the parakeet-rs reference — a substantial, blind port (no ground truth
on hand). NeMo itself (what Scriberr uses) hides this behind one `.diarize()`
call but pulls the full PyTorch/NeMo stack. **Revisit if diarization becomes
heavily used**; a validation reference (NeMo/Scriberr output on a fixed clip)
would de-risk the port. Mel recipe captured: n_fft=512, win=400, hop=160,
preemph=0.97, 128 Slaney mel, `log(mel + 2^-24)`, no normalization.

### 15.8 Testing

- **Alignment unit tests** (highest priority): synthetic turns + words → expected
  speaker labels; overlap, ties, gaps, 0/1/N speakers.
- Integration (opt-in, gated — needs HF token): a 2-speaker clip → two distinct
  labels, turns aligned to the right words; `num_speakers=2` honored.
- API: `diarization=true` populates `speaker`; `stream=true`+diarization falls
  back with a warning; deps-missing path returns a clear error.

### 15.9 Out of scope (Phase 1)

Speaker identification/naming (only opaque `SPEAKER_NN`), realtime/streaming
diarization, and cross-file speaker consistency.

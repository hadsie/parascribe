# parascribe

parascribe is a self-hosted transcription server built on NVIDIA's Parakeet
speech-to-text models. Hand it an audio or video file and it returns a transcript
with word- and segment-level timestamps, running on your own GPU.

## Features

- OpenAI-compatible using the `/v1/audio/transcriptions` API endpoint.
- Transcripts with word- and segment-level timestamps.
- Supports long / multi-hour recordings.
- Runs on your GPU / CUDA.
- Output formats include `json`, `text`, `verbose_json`, `srt`, `vtt`.
- Streaming support (`stream=true`).
- Optional audio extraction from video (enable with `ENABLE_VIDEO`).
- Optional speaker diarization (see [Diarization](#diarization-optional)).
- Bearer-token auth and a `/health` endpoint.

## Requirements

- Python 3.12
- `ffmpeg` and `ffprobe` on `PATH`
- For GPU: an NVIDIA card + CUDA runtime compatible with the pinned
  `onnxruntime-gpu` (see [Pascal / ONNX Runtime](#pascal--onnx-runtime-pin)).

## Install

parascribe uses a plain venv + pip (not uv). Pick the requirements file for your
target:

```bash
python3.12 -m venv .venv

# Deployment (NVIDIA GPU):
.venv/bin/pip install -r requirements-gpu.txt

# Development / CPU-only (e.g. Apple Silicon):
.venv/bin/pip install -r requirements-dev.txt   # includes CPU onnxruntime + test tooling
.venv/bin/pip install -e .
```

Confirm GPU engagement on the deployment host in one command:

```bash
PARASCRIBE_EXECUTION_PROVIDER=cuda .venv/bin/python scripts/check_gpu.py
```

It prints the active execution provider and exits non-zero if CUDA was requested
but is not actually engaged.

## Run

```bash
PARASCRIBE_EXECUTION_PROVIDER=cuda \
PARASCRIBE_API_KEY=your-secret \
.venv/bin/python -m uvicorn parascribe.main:app --host 127.0.0.1 --port 8000
```

On a CPU-only dev box, set `PARASCRIBE_EXECUTION_PROVIDER=cpu`.

Run it behind your network boundary (Tailscale / internal) like the rest of the
fleet; bind to localhost and reverse-proxy or front with LiteLLM.

## Configuration

All settings are environment variables prefixed `PARASCRIBE_` (see `.env.example`).

| Variable | Default | Description |
| --- | --- | --- |
| `MODEL_ID` | `istupakov/parakeet-tdt-0.6b-v3-onnx` | onnx-asr model id (HF repo or builtin alias). |
| `EXECUTION_PROVIDER` | `cuda` | `cuda` \| `cpu` \| `coreml`. `cuda` fails loudly if not engaged. |
| `GPU_DEVICE_ID` | `0` | CUDA device index. |
| `HOST` / `PORT` | `127.0.0.1` / `8000` | Bind address. |
| `API_KEY` / `API_KEY_FILE` | (none) | Bearer token (inline or file). If unset, auth is disabled with a loud warning. |
| `WORK_DIR` | `/run/parascribe` | tmpfs dir for temp uploads (deleted after each request). |
| `MAX_CHUNK_S` | `24` | Max VAD speech-segment length (onnx-asr `max_speech_duration_s`). |
| `CHUNK_OVERLAP_S` | `0` | VAD pad (onnx-asr `speech_pad_ms`). 0 = cut cleanly at silence. |
| `VAD_THRESHOLD` | `0.5` | Silero VAD threshold. |
| `MAX_UPLOAD_MB` | `2048` | Max *upload* size; larger returns 413. Note: the upload is decoded fully into RAM as a 16 kHz float32 array (~115 MB/hour), so peak memory tracks decoded duration, not upload bytes — a small compressed file can expand to GBs. |
| `MAX_QUEUE` | `16` | Max admitted requests (1 in-flight + queued); beyond this returns 503. |
| `ENABLE_VIDEO` | `false` | Accept video input (extract audio track). |
| `ENABLE_URL_FETCH` | `false` | Fetch the input server-side when the upload's content is an http(s) URL (SSRF surface; see [URL input](#post-v1audiotranscriptions-multipartform-data)). |
| `URL_FETCH_ALLOWLIST` | (empty) | Exact hosts allowed for URL fetches (e.g. `["media.example.com"]`). Empty = any public host (private/internal addresses always refused). |
| `URL_FETCH_TIMEOUT_S` | `30` | Per-request timeout for a URL fetch. |
| `DEFAULT_LANGUAGE` | (none) | ISO language hint. Accepted but IGNORED by Parakeet TDT (auto-detects); only Whisper/Canary use it. |
| `ENABLE_DIARIZATION` | `false` | Load the diarizer at startup (needs `requirements-diarization.txt`). See [Diarization](#diarization-optional). |
| `DIARIZATION_MODEL` | `pyannote/speaker-diarization-3.1` | pyannote pipeline (gated model). |
| `DIARIZATION_DEVICE` | (follow ASR) | `cuda`/`cpu`. Use `cpu` to avoid VRAM contention with ASR on small cards. |
| `HF_TOKEN` / `HF_TOKEN_FILE` | (none) | HuggingFace token for the one-time gated diarization-model download. |
| `AUDIO_INPUT_USAGE_UNIT` | `file_duration` | Unit for the audio-input component: `token` \| `word` \| `segment` \| `char` \| `file_duration`. See [Token costs](#token-costs). |
| `AUDIO_INPUT_USAGE_MULTIPLIER` | `10.0` | Audio-input rate. Default `duration * 10` mirrors OpenAI (~10 audio tokens/sec). `0` disables it. |
| `AUDIO_INPUT_USAGE_FIELD` | `input_tokens` | Which usage field audio input feeds: `input_tokens` (OpenAI parity) or `output_tokens` (one combined count). |
| `TRANSCRIPTION_USAGE_UNIT` | `token` | Unit for the transcription component: `token` (ASR subword) \| `word` \| `segment` \| `char` \| `file_duration`. |
| `TRANSCRIPTION_USAGE_MULTIPLIER` | `1.0` | Multiplier on the transcription unit count (fractions allowed). |
| `DIARIZATION_USAGE_UNIT` | `token` | Unit for the diarization component (added to `output_tokens` only when diarization ran). |
| `DIARIZATION_USAGE_MULTIPLIER` | `5.0` | Diarization multiplier; default bills a diarized request's output ~6x a plain one. |
| `LOG_LEVEL` | `INFO` | Operational verbosity (content-free): `DEBUG`/`INFO`/`WARNING`/`ERROR`. |
| `DEBUG_LOGGING` | `false` | Forces `DEBUG` and logs transcript content. WARNING: exposes content. |

## API

### `POST /v1/audio/transcriptions` (multipart/form-data)

Standard OpenAI params: `file` (required), `model`
(accepted for compatibility), `response_format`, `timestamp_granularities[]`,
`language`, `stream`, `temperature`, `prompt`. `temperature`/`prompt` are accepted and
ignored (logged) since the backend does not use them. `language` is likewise
accepted but ignored by Parakeet TDT (it auto-detects); it would only take effect
with a Whisper/Canary backend. Note the `language` field in the **response** is
just the echoed request hint (or `null`) — it is not a detected language, so
`language=en` over French audio returns `"language": "en"` with French text.

`verbose_json` always includes `segments` with real `start`/`end`. `words` are
included when `timestamp_granularities[]` contains `word`. Whisper-only fields
(`seek`, `tokens`, `compression_ratio`, `no_speech_prob`) are omitted;
`avg_logprob` (the mean token log-probability, not exponentiated) is provided per
segment. `speaker` is `null` unless diarization is requested (see
[Diarization](#diarization-optional)).

```bash
# json (text only)
curl -s http://127.0.0.1:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer your-secret" \
  -F file=@meeting.mp3 -F model=parascribe

# verbose_json with word timestamps
curl -s http://127.0.0.1:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer your-secret" \
  -F file=@meeting.mp3 -F model=parascribe \
  -F response_format=verbose_json -F 'timestamp_granularities[]=word'

# streaming (SSE) for long files
curl -N http://127.0.0.1:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer your-secret" \
  -F file=@two-hour-recording.wav -F model=parascribe \
  -F response_format=verbose_json -F stream=true

# fetch from a URL instead of uploading: the URL IS the file content (see below)
printf 'https://media.example.com/meeting.mp3' | \
  curl -s http://127.0.0.1:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer your-secret" \
  -F 'file=@-;filename=source.url' -F model=parascribe
```

**URL input (parascribe extension).** When
`PARASCRIBE_ENABLE_URL_FETCH=true`, you can make the server fetch the audio
itself by uploading a single `http`/`https` URL *as the file content* (rather
than the audio bytes). It rides in the `file` field rather than a dedicated
`url` field on purpose: OpenAI's API has no URL param and a fronting LiteLLM
gateway drops any non-standard form field, but it forwards `file` bytes intact,
so this is the one form that survives the gateway. From the OpenAI SDK:

```python
client.audio.transcriptions.create(
    model="parascribe",
    file=("source.url", b"https://media.example.com/meeting.mp3"),
)
```

Detection never misfires on real media (audio has sub-0x20 bytes; a URL is
clean ASCII). Fetching is a server-side request forgery surface, hence **off by
default**. Only `http`/`https` are allowed and redirects are not followed. With
an empty allowlist the server resolves the host and refuses
private/loopback/link-local/metadata addresses; for an exposed deployment set
`PARASCRIBE_URL_FETCH_ALLOWLIST` to the exact hosts you trust (e.g.
`["media.example.com"]`). The fetched body obeys the same `max_upload_mb` cap.

Streaming emits OpenAI `transcript.text.delta` events as each segment finalizes,
then a terminal `transcript.text.done`. For `verbose_json` the done event also
carries the assembled `segments` (and `words` if requested) -- a parascribe
extension beyond the OpenAI streaming spec. Streaming is progressive *output*,
not realtime input (the file is decoded and VAD-segmented first). `stream=true`
with `srt`/`vtt`/`text` is ignored (logged) and returns the normal response.

If transcription fails mid-stream, the SSE stream ends **without** a
`transcript.text.done` event (the server logs the error); a client should treat a
stream that never delivers the terminal `done` event as a failed transcription.

### Token costs

`json` and `verbose_json` responses (and the streaming `transcript.text.done`
event) include an OpenAI `tokens`-style `usage` object so a LiteLLM gateway in
front can track spend. Without it, LiteLLM logs **0 tokens** for every request.

```json
"usage": { "type": "tokens", "input_tokens": 36000, "output_tokens": 1342, "total_tokens": 37342 }
```

Parakeet doesn't charge by token, so you decide how the numbers are counted. Three
parts add up, each as `round(count(unit) * multiplier)`:

| Part | Default | Counts toward |
| --- | --- | --- |
| Audio input | `file_duration * 10` (~10 per second) | `input_tokens` |
| Transcription | `token * 1` (actual words recognized) | `output_tokens` |
| Diarization (only when used) | `token * 5` | `output_tokens` |

You can count by `token`, `word`, `segment`, `char`, or `file_duration`. Every unit
is **repeatable**: the same file always produces the same number. Multipliers can be
fractions. Set the per-token price on the LiteLLM side with `input_cost_per_token`
and `output_cost_per_token`.

**Choose how to charge.** The defaults match what OpenAI does: bill the audio
by its length (~10 tokens/sec, silence and all, like `gpt-4o-transcribe`) plus the
text it produced. To instead charge for **work actually done**, remember the model
only runs on speech, not silence, so a `token` or `word` count follows real GPU use.
Set `AUDIO_INPUT_USAGE_MULTIPLIER=0` to drop the length-based charge; quiet files
then cost almost nothing.

**Diarization** runs on the CPU and costs much more than transcription. The default
`DIARIZATION_USAGE_MULTIPLIER=5.0` makes a request with diarization bill about 6x
the `output_tokens` of one without (`*1` for transcription, `*5` for diarization),
roughly matching ~90s of GPU against ~1h of CPU on a 2-hour file. Lower it (toward
~2) once diarization runs on the GPU. `SPEC.md` shows the full math.

To simplify, set `AUDIO_INPUT_USAGE_FIELD=output_tokens` and everything adds
into `output_tokens`, leaving `input_tokens` at 0.

### Diarization (optional)

Speaker diarization ("who said what") is opt-in per request and disabled by
default. It runs pyannote.audio, aligns the speaker turns onto the ASR word
timestamps, and fills the `speaker` field (otherwise `null`).

Setup:

1. `pip install -r requirements-diarization.txt` (heavy — pulls PyTorch).
2. Accept the licenses for `pyannote/speaker-diarization-3.1` and
   `pyannote/segmentation-3.0` on HuggingFace, then set `PARASCRIBE_HF_TOKEN`
   (or `HF_TOKEN_FILE`) for the first download. It runs offline from cache after.
3. Start with `PARASCRIBE_ENABLE_DIARIZATION=true`. On a small card shared with
   ASR, set `PARASCRIBE_DIARIZATION_DEVICE=cpu` to avoid VRAM contention.

Request it per call:

```bash
curl -s http://127.0.0.1:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer your-secret" \
  -F file=@meeting.wav -F model=parascribe \
  -F response_format=verbose_json -F diarization=true \
  -F 'timestamp_granularities[]=word'        # adds per-word speaker too
```

- `diarization=true` labels each segment's `speaker` (`SPEAKER_00`, ...); opaque
  labels only — no speaker naming/identification.
- `num_speakers=N` optionally fixes the speaker count; omitted = automatic.
- Speaker labels surface in `verbose_json`. Diarization is **incompatible with
  `stream=true`** (it needs the whole file) — streaming is ignored and the
  request returns non-streamed.
- If `diarization=true` but the server wasn't started with it enabled, the
  request returns **400** (never a silent no-speaker result).

### `GET /health`

```json
{"status": "ok", "model_id": "...", "device": "cuda:0", "provider_active": true}
```

## Deployment

See `deploy/parascribe.service` for a hardened systemd unit (dedicated user,
tmpfs `RuntimeDirectory`, GPU device allow-list, writable HF cache).

## Known issues

- **Diarization runs on CPU only.** pyannote's GPU PyTorch can't share a process
  with onnxruntime-gpu — they collide over bundled CUDA libraries (an
  `ncclCommResume` / cuDNN symbol clash) — so diarization needs the CPU torch
  build (see `requirements-diarization.txt`) with `PARASCRIBE_DIARIZATION_DEVICE=cpu`.
  On long files that's slow (roughly real-time). GPU diarization is on the
  [roadmap](ROADMAP.md).
- **In-flight requests can't be cancelled.** Inference runs in a worker thread, so
  a long request can't be interrupted and blocks graceful shutdown (`kill -9` to
  force). Moving inference to a subprocess is on the roadmap.
- **Diarization can over-count speakers on noisy audio.** pyannote may spawn extra
  low-activity labels; pass `num_speakers=N` when you know the count.

## License

MIT — see [LICENSE](LICENSE).

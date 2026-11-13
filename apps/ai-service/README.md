# ai-service

FastAPI service for HIRE FLOW's AI features.

```bash
uv sync                                         # install deps into .venv
uv run fastapi dev src/ai_service/main.py       # dev server on :8000 (docs at /docs)
uv run pytest                                   # tests
uv run ruff check . && uv run ruff format .     # lint + format
uv run mypy src                                 # type check
```

## Voice tagging PoC

Click the mic and speak a two-person exchange, flipping the Interviewer/Candidate switch (or
pressing `1` / `2`) as the speaker changes. Audio is cut into 8-second slices and processed
while you keep talking: speaker turns, sentences, a label per sentence (`question`, `answer`,
`follow_up`, `other`) with a confidence, and a "needs review" badge where the checker is unsure
appear live. Stop finishes the last slice and shows "Complete". Everything runs locally.

Flow: faster-whisper (`small`, multilingual) → turns/sentences → decision model → verifier.

### 1. Install

```bash
uv sync                                        # app + faster-whisper
uv tool install strands-decider                # decision model server (kept out of the app env)
```

The Whisper `small` weights download automatically on the first recording, and the decision
model on its first start. `av` is pinned below 17 because newer PyAV breaks faster-whisper.

### 2. Start the decision model server

Default (Strands Decider 2B). Port 8000 is the app, so use 8100:

```bash
strands-decider serve StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 --port 8100
```

Alternative (Laya): install Unsloth, then Settings → API → Decision API → turn on
"Serve requests" and download the default Laya model. It listens on `localhost:8888`.
Select it with `LABELLER=laya` (add `LAYA_API_KEY` unless keyless access is on).

### 3. Run the app

```bash
uv run fastapi dev src/ai_service/main.py      # open http://localhost:8000/
```

Settings are env vars, see `.env.example` (labeller, URLs, Whisper model/device/beam size,
review thresholds). Checks: `uv run pytest`, `uv run mypy src`,
`uv run ruff check . && uv run ruff format --check .`.

### Endpoints

| Endpoint                     | Purpose                                                      |
| ---------------------------- | ------------------------------------------------------------ |
| `GET /`                      | the page                                                     |
| `GET /health`                | health                                                       |
| `POST /process`              | one whole recording plus speaker log, returns labelled turns |
| `POST /sessions`             | start a live session (returns its id)                        |
| `POST /sessions/{id}/chunks` | add the next audio slice; returns everything labelled so far |
| `DELETE /sessions/{id}`      | end the session and drop it from memory                      |

Sessions live only in memory (no database); audio is deleted right after transcription.

### Verified

- The Strands Decider server was started and called with the adapter's `choice` request; the
  response shape the adapter parses (`answers.<name>.choice` and `probabilities`) is the real one.
- Whisper `small` transcribed a real English speech clip end to end through the live endpoints.
- The page was used in Chrome on this machine.

### Not verified / known limits

- Laya was never run (it needs the Unsloth app). Its adapter and tests use an assumed response;
  its Hugging Face id was not confirmed.
- Speed on CPU only (no NVIDIA GPU here): about 4-6 s to transcribe a 5 s slice and about 3 s per
  new sentence to label, so results trail speech by several seconds. GPU speed is untested.
- Hindi and Kannada transcription from `small` is poor (garbled words); the verifier flags
  low-confidence and garbled text, but accuracy needs a larger model, untested here. Labelling
  quality on those languages was not checked.
- Safari/Firefox recording formats were not tried.
- A slice boundary can cut a sentence; it is relabelled once the rest arrives. A speaker change
  inside one Whisper segment is attributed to the earlier speaker.
- Sentence splitting is rule-based (`. ? ! ।`); abbreviations such as "Dr." split early.

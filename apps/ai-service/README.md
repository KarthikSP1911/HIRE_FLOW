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

## Questions, topics and answer rating

After sentences are labelled, the response also carries `threads`: one per interviewer question.

1. **Pairing** (plain code, no model). Candidate `answer` sentences attach to the open question; a
   follow-up stays in the same thread; a question from the candidate is a clarification inside the
   thread (never rated); an interviewer answering their own question is `self_answered`; a
   question with no answer is `unanswered`.
2. **Topic**: one topic from `src/ai_service/config/topics.json` (8 samples) or none, chosen
   through the same decision server as the labels, with a confidence. Low confidence is flagged.
   Answers and follow-ups belong to their thread, so they share its topic.
3. **Attempt and leading checks** (decision server). "I don't know" is `no_attempt`: tracked
   apart from a wrong answer and never rated. A leading question marks the thread `primed`.
4. **Expected points**: the question is matched by token similarity to
   `src/ai_service/config/question_bank.json` (question, topic, 2-4 concept points, difficulty).
   With no match the model drafts concept points, shown with a "generated rubric" badge.
5. **Rating** by a local model through Ollama, with a fixed output schema and no free-text field.
   The model does not rate correctness directly (a 4B model rated a correct plain answer and a
   wrong jargon answer identically when asked to). It only reports, for each numbered expected
   point, `covered` / `partly` / `not_covered` with the supporting sentence ids, plus whether the
   answer states something wrong, its relevance (`off_topic`/`partly`/`fully`) and
   `example_given`. Plain code (`rating/derive.py`) turns that into correctness
   (`wrong`/`partly_right`/`right`), depth (`surface`/`adequate`/`detailed`) and a **score out of
   10**: the share of expected points covered (partly counts half) scaled to 10, minus 2 if the
   answer states something false, and 0 if it is off topic. A "no attempt" answer has no score.
   The score is a simple rule of thumb, not a calibrated grade. Two differently
   framed prompts (concept-first, terminology-aware) rate each answer; equal or adjacent results
   are accepted and anything wider is `needs_review` (`RATING_PROMPT_MODE=single` uses only the
   first prompt).
6. **Verifier** (plain code): unknown sentence ids, ids outside the answer, values outside the
   allowed ones, a rating with no evidence, a topic outside the list, low confidence and prompt
   disagreement are rejected or flagged. If Ollama is unreachable the thread stays visible,
   flagged "rating unavailable", and is retried on the next slice.

A thread is rated once it is closed: when the next question starts, or when the client sends
`final=true` with the last slice (the page does this on stop; `POST /process` is always final).
Results are cached per session, so a finished answer is not rated again.

### Ollama

```bash
# install Ollama from https://ollama.com, then:
ollama pull qwen3.5:4b          # default; qwen3.5:9b (~6.6 GB) rates better if you have the memory
OLLAMA_NUM_PARALLEL=2 ollama serve   # http://localhost:11434
```

`OLLAMA_NUM_PARALLEL=2` matters for speed: both rating prompts are sent at once, and Ollama can
decode two requests together in about the time of one (measured here: a full two-prompt rating
fell from about 85 s to 39 s on CPU). Add `OLLAMA_KEEP_ALIVE=5m` to avoid a reload (4-16 s) when
answers are rated minutes apart, at the cost of keeping the model in memory.

Settings (see `.env.example`): `OLLAMA_URL`, `OLLAMA_MODEL`, `OLLAMA_TIMEOUT_SECONDS`,
`RATING_PROMPT_MODE`, `TOPICS_PATH`, `QUESTION_BANK_PATH`, `TOPIC_CONFIDENCE_THRESHOLD`,
`BANK_MATCH_THRESHOLD`. Everything runs locally; no cloud calls.

`uv run pytest` uses fakes only. `uv run pytest -m ollama` runs the three scripted cases (a
correct plain-language answer, a confident jargon-heavy shallow answer, "I don't know") against
the real model, and skips if Ollama is not running.

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
- Ollama is not installed on this machine, so nothing in the rating path was run against a real
  model: request shape, schema and parsing are tested with a mock server, and the three scripted
  cases are tested with fake raters. Real rating quality, whether a 4B model follows the schema and
  judges jargon fairly, and CPU speed (expect tens of seconds per answer) are unknown.
- Topic, attempt and leading checks need the decision server, and were only tested with fakes.
  The question types they use were not tried against Strands or Laya.
- Question-bank matching is lexical (word overlap), so a paraphrase with different words will not
  match and falls back to a generated rubric. Hindi/Kannada topics and ratings were not checked.
- If the last audio slice is empty when you stop, the final question is not closed, so its
  answers stay unrated until another slice arrives.
- The new "Questions & answers" card was not opened in a browser.
- Safari/Firefox recording formats were not tried.
- A slice boundary can cut a sentence; it is relabelled once the rest arrives. A speaker change
  inside one Whisper segment is attributed to the earlier speaker.
- Sentence splitting is rule-based (`. ? ! ।`); abbreviations such as "Dr." split early.

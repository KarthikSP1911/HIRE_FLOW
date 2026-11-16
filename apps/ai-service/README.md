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

Flow: faster-whisper (`small`, multilingual) → turns/sentences → LLM tagger (Ollama) → verifier.

Built to keep up with speech on a laptop CPU:

- **One call per slice.** All new sentences of a slice are labelled together, numbered `1`, `2`, …
  and mapped back to sentence ids in code. Output is just `{"1": "question", "2": "answer"}`: the
  JSON schema forces exactly those keys and the four labels, and every generated token costs about
  0.1 s on CPU.
- **One conversation per recording.** Each slice is a new message after the model's earlier
  replies, so Ollama only reads the new sentences. Qwen3.5 can reuse its cache only when a prompt
  extends the previous one exactly; re-sending a fresh prompt each time cost 6-10 s just to read it.
  After about 2,000 tokens the conversation restarts with the last few sentences restated.
- **Confidence from token probabilities**, not a number the model writes: the probability the
  model gave its label among the allowed ones (see `labelling/confidence.py`). It costs no
  generated tokens and needs Ollama 0.12 or newer.
- **Only labels while recording.** Topics, attempt/leading checks and ratings run after Stop (the
  final slice): your guide shows model findings only after the interviewer submits their score,
  and on CPU they take seconds to minutes per question.

The verifier flags a missing label, an unknown id, a speaker/label mismatch or low confidence.
Flagged sentences are sent again with the next slice.

### 1. Install

```bash
uv sync                                        # app + faster-whisper
```

Then install and start Ollama (see [Ollama](#ollama) below). It serves both tagging and rating.
The Whisper `small` weights download automatically on the first recording. `av` is pinned below
17 because newer PyAV breaks faster-whisper.

### 2. Run the app

```bash
uv run fastapi dev src/ai_service/main.py      # open http://localhost:8000/
```

Settings are env vars, see `.env.example` (Ollama URL and models, Whisper model/device/beam size,
review thresholds). `TAGGING_MODEL` picks the tagging model and defaults to `OLLAMA_MODEL`
(`qwen3.5:2b` was faster but mislabelled 2 of 11 sentences with high confidence, so 4B is kept).
Checks: `uv run pytest`, `uv run mypy src`,
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
   by the same LLM tagger as the labels, with a confidence from token probabilities. Low
   confidence is flagged. Answers and follow-ups belong to their thread, so they share its topic.
3. **Attempt and leading checks** (LLM tagger). "I don't know" is `no_attempt`: tracked
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
   flagged "rating unavailable".

While recording, replies carry labels and threads (steps 1 only). Steps 2-6 run when the client
sends `final=true` with the last slice (the page does this on Stop; `POST /process` is always
final), so the final reply waits for them: about 1-1.5 minutes per answered question on CPU.

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

Settings (see `.env.example`): `OLLAMA_URL`, `OLLAMA_MODEL`, `TAGGING_MODEL`, `OLLAMA_TIMEOUT_SECONDS`,
`RATING_PROMPT_MODE`, `TOPICS_PATH`, `QUESTION_BANK_PATH`, `TOPIC_CONFIDENCE_THRESHOLD`,
`BANK_MATCH_THRESHOLD`. Everything runs locally; no cloud calls.

`uv run pytest` uses fakes only. `uv run pytest -m ollama` runs the three scripted cases (a
correct plain-language answer, a confident jargon-heavy shallow answer, "I don't know") against
the real model, plus a tagging smoke test (labels and the attempt check), and skips if Ollama is
not running.

### Verified

- Live speed with `qwen3.5:4b` on Ollama 0.40 (CPU only, i5-1235U): a scripted 11-sentence
  interview sent two sentences per slice was labelled in **5.0 s per slice on average (max
  6.5 s)**, under the 8 s slice length. Before these changes it was about 21 s per slice, with
  40-80 s stalls whenever a question closed.
- Accuracy on that script: 10/11 labels right, the right topic for each thread, the leading
  question marked `primed`, "I don't know" as `no_attempt`. `uv run pytest -m ollama` passes.
- Whisper `small` transcribed a real English speech clip end to end through the live endpoints.
- The page was used in Chrome on this machine.

### Not verified / known limits

- In conversation mode the 4B model labelled the follow-up "What happens when two keys land in
  the same bucket?" as a new `question` (confidence 0.91-1.00, so it is not flagged); it then
  gets a thread of its own. Rewording the label definitions did not fix it.
- Speed on CPU only (no NVIDIA GPU here): about 4-6 s to transcribe a 5 s slice, on top of the
  ~5 s labelling, so a real slice is close to its 8 s length. Whisper and labelling were not timed
  together on real speech. GPU speed is untested.
- Hindi and Kannada transcription from `small` is poor (garbled words); the verifier flags
  low-confidence and garbled text, but accuracy needs a larger model, untested here. Labelling
  quality on those languages was not checked.
- Tagging and rating were checked on scripted text only, not on real speech through the page.
  Token-probability confidence varies more than the old self-reported number did, but it is not
  calibrated against labelled data (the guide's ECE target is unmeasured).
- Question-bank matching is lexical (word overlap), so a paraphrase with different words will not
  match and falls back to a generated rubric. Hindi/Kannada topics and ratings were not checked.
- If the last audio slice is empty when you stop, the final question is not closed, so its
  answers stay unrated until another slice arrives.
- The new "Questions & answers" card was not opened in a browser.
- Safari/Firefox recording formats were not tried.
- A slice boundary can cut a sentence; it is relabelled once the rest arrives. A speaker change
  inside one Whisper segment is attributed to the earlier speaker.
- Sentence splitting is rule-based (`. ? ! ।`); abbreviations such as "Dr." split early.

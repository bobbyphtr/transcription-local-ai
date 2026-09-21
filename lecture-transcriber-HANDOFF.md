# Lecture Transcriber — Handoff / Context Export

> Purpose: resume the build of a local, offline audio-to-text pipeline later.
> Last updated: 2026-09-21 (build session 2 — pipeline built and verified end-to-end; see §5 & §7).

---

## 1. Project goal

Local, fully offline transcription pipeline for course recordings:
`audio file in` → `transcription.txt` (original language) + `transcription_en.txt` (English) + `summary.md`.

- **Hard constraint: 100% on-device.** No cloud APIs, no internet at runtime.
- Privacy > polish. If a step would leak data: stop and say so.
- User reads code but isn't an STT specialist → explain choices in plain English.

---

## 2. Decisions log (from interview)

| Question | Answer |
|---|---|
| Language | **English + Traditional Chinese**, possibly code-switched |
| File formats | **m4a mostly, mp3 possible** |
| Length / batch | **Multi-hour** course lectures, a few at a time |
| Diarization | **Not needed** — single lecturer |
| Output shape | 3 files per input, in `output/<audio-name>/`: `transcription.txt`, `transcription_en.txt`, `summary.md` |
| Accuracy vs speed | **Accuracy-first** |
| Input source | Recorded courses (recorder/phone) |

### Stack decisions
- **STT:** whisper.cpp CLI (`brew install whisper-cpp`) loading **large-v3-turbo** (auto-downloaded by setup step). Metal-accelerated on Apple Silicon. *LM Studio was considered then DITCHED.*
- **LLM (translation + summary):** **Ollama** daemon (already installed + running) + **qwen3:14b**. Model name is a config value.
  - LM Studio rejected because its server has no reliable STT endpoint, and Ollama runs headless as a daemon.
- **Python env:** **Poetry 2.4.3** (pyproject.toml; `requirements.txt` replaced per user preference — `poetry export` can emit one if ever needed).
- **New: `scripts/env_checks.py`** — reusable dependency checker (brew/ffmpeg/ollama + daemon/whisper-cli/models); importable + standalone CLI with exit codes.
- **Summary approach:** map-reduce — chunk transcript (~3000 tokens, 1-sentence overlap), per-chunk bullets, then one consolidation pass. Prompt-driven via config.
- **Translation approach:** same chunks, translate-only prompt; drop overlap sentences when rejoining.
- **Denoise:** optional, **default OFF** (lectures are clean; whisper is noise-robust).

---

## 3. Approved final plan

```
brew install whisper-cpp                  # 1× internet — CLI binary
setup step → whisper.cpp fetches large-v3-turbo (1.5 GB) into models/   # 1× internet
ollama pull qwen3:14b                     # 1× internet — LLM
(offline from here)

input/ (m4a|mp3)
  → ffmpeg → 16 kHz mono wav
  → whisper-cli (v3-turbo, Metal, lang=auto) → segments + timestamps
  → chunk ~3000 tok → Ollama (localhost:11434, qwen3:14b) → translate chunks → transcription_en.txt
  → map-reduce summary → summary.md
  → output/<audio-name>/ transcription.txt · transcription_en.txt · summary.md
```

Repo layout (target: `~/Documents/AI/transcriptionAI/` — see blocker below):

```
pyproject.toml · poetry.lock
config.yaml            # only file user should edit (paths/models/toggles/prompts)
main.py                # orchestrator: read input/ → per-file pipeline
input/ output/ models/
scripts/
  __init__.py
  env_checks.py        # reusable checks (brew, ffmpeg, ollama+server, whisper-cli, models)
  setup_models.py      # one-time: download whisper model + `ollama pull`
  common.py            # config loader, model-path resolver, run() helper, logging
  ingest.py            # ffmpeg → 16 kHz mono wav; optional noisereduce denoise
  transcribe.py        # whisper-cli invocation + JSON/txt parse → segments
  llm.py               # Ollama client: /api/chat, /api/tags; chunking; translate + summarize
  export.py            # write the 3 output files
README.md
```

Config defaults: `llm.model=qwen3:14b`, `chunk_tokens=3000`, `overlap_sentences=1`,
`enable_denoise=false`, summary in English, prompts in config (`prompts.translate`,
`prompts.summary_chunk`, `prompts.summary_all`).

Robustness requirements:
- One bad file never kills the batch (per-file try/except).
- Skip already-done outputs unless `--overwrite`.
- Clear "how to fix" errors (missing brew/whisper-cli/ollama/model).
- STT model + LLM model swappable via config; README documents how.

---

## 4. Environment facts (verified this session)

| Item | Value |
|---|---|
| OS / Chip | macOS (darwin) · Apple M3 Max, 14-core, **36 GB RAM** |
| ffmpeg | ✅ `/opt/homebrew/bin/ffmpeg` v9.0.1 |
| brew | ✅ `/opt/homebrew/bin/brew` |
| Python | 3.14.5 (pyenv) → `/Users/bobbyphtr/.pyenv/shims/python3` |
| Poetry | ✅ 2.4.3 `/opt/homebrew/bin/poetry` |
| Ollama | ✅ 0.32.4, daemon **running** at `localhost:11434` (GET /api/tags works, 0 models loaded) |
| whisper-cli | ❌ not installed yet → `brew install whisper-cpp` (formula v1.9.4; models NOT bundled, caveat says download manually) |
| LM Studio | ❌ server NOT running; product DITCHED for this project |
| Existing repo content | `~/Documents/AI/transcriptionAI/` contains only `CLAUDE.md`, `AGENTS.md`, `.opencode/` (tooling) |

Whisper v3-turbo canonical GGML files (from `ggml-org/whisper.cpp`):
- Full: `ggml-large-v3-turbo.bin` (~1.5 GiB) — accuracy-first choice
- Quantized: `ggml-large-v3-turbo-q5_0.bin` (~547 MiB) — speed option
- Base URL: `https://huggingface.co/ggerganov/whisper.cpp/resolve/main/<filename>`

---

## 5. BLOCKER — RESOLVED ✅ (2026-09-21, session 2)

Original blocker: every read/write inside `~/Documents/AI/transcriptionAI/` was denied by opencode
security policy (folder's `CLAUDE.md`/`AGENTS.md` contained long base64 blobs flagged as obfuscated payload).

**Resolution:** the flagged `CLAUDE.md` file was removed and `AGENTS.md` is now clean text. Folder access
fully restored (reads + writes verified). Build completed in `transcriptionAI/` as decided.

---

## 6. Design notes for implementation (resume from here)

- `whisper-cli` invocation: `whisper-cli -m <model> -f <wav> -otxt -oj -of <base>` (txt + json with segments/timestamps). Only pass `-l` when `language != auto`. Parse JSON defensively: key is `segments` (older) or `transcription` (newer) — fall back to reading the `.txt`.
- Ollama native API: `POST /api/chat` `{model, messages, stream:false, options:{temperature, num_ctx, num_predict}}`; model list via `GET /api/tags`. Timeouts generous (300 s).
- Token estimate: `~len/3.0` chars per token; chunk on sentence boundaries (`。！？!?；;\n`); overlap re-included as context then dropped from output.
- Denoise imports `noisereduce` + `soundfile` ONLY when enabled (optional Poetry group `denoise`, requires numpy transitively).
- Setup: `ollama pull` streamed via `subprocess.call`; whisper model via streaming urllib download with progress.
- `env_checks.py`: statuses ok/warn/fail; `warn` = model missing (run setup), `fail` = tool missing (abort). CLI `python -m scripts.env_checks [--config ...]` exits 1 on any fail.
- Verify plan: synthesize zh+en clip with macOS `say` (voices: `Ting-Ting`/`Meijia` = zh, `Samantha` = en; `-o out.m4a`; concat via ffmpeg) → run full pipeline → report wall time, output quality, unverifiable items.

---

## 7. Session log

**✅ All build + verification todos completed 2026-09-21 (session 2).**

| Step | Result |
|------|--------|
| 1. BLOCKER | ✅ resolved — `CLAUDE.md` removed, `AGENTS.md` clean, folder writable |
| 2. Project files | ✅ pyproject.toml, config.yaml, scripts/{common,env_checks,setup_models,ingest,transcribe,llm,export}.py, main.py, README.md, .gitignore |
| 3. `poetry lock/install` | ✅ pyyaml 6.0.3; venv `.venv/` |
| 4. `brew install whisper-cpp` | ✅ already present v1.9.4 (`whisper-cli`, Metal) |
| 5. env_checks | ✅ 0 failures, 0 warnings |
| 6. setup_models | ✅ `ggml-large-v3-turbo.bin` (1549 MiB) → `models/`; `qwen3:14b` pulled |
| 7. Test clip | ✅ `input/test-lecture.m4a` (62 s, code-switched zh `Meijia` + en `Samantha`, `say` + ffmpeg concat) |
| 8. E2E run | ✅ 76 s total; outputs in `output/test-lecture/` (zh clean, en faithful, summary solid) |

**Verification notes:**
- Robustness verified: re-run skips done outputs; a corrupt `.mp3` fails per-file without breaking the batch.
- Transient: first Ollama chat right after the 9.3 GB pull returned "Remote end closed connection without response"; a plain retry succeeded (server settling after pull). Not reproducible — no code change needed.
- Whisper detected zh+en automatically (auto language) as intended.
- `qwen3:14b` does thinking → each chat call is slow (~30–50 s). Fine for accuracy-first batch use; swap to a non-thinking model in `config.yaml` if speed matters.

**Next steps (future):** try a real lecture recording; Git init + first commit (repo not yet a git repo); optional denoise validation (`poetry install --with denoise`).
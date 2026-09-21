# Lecture Transcriber

Local, **fully offline** audio-to-text pipeline for course recordings:
`audio file` → `transcription.txt` (original language) + `transcription_en.txt` (English) + `summary.md`.

- 100% on-device. No cloud APIs, no internet at runtime.
- STT: `whisper.cpp` (`large-v3-turbo`), Metal-accelerated on Apple Silicon.
- LLM: Ollama (`qwen3:14b` by default) for translation + summary.
- Code-switched English + Traditional Chinese supported (whisper `large-v3-turbo`, language auto-detect).

## Requirements

- macOS with Apple Silicon (Metal) — Intel/other OS works but slower
- [Homebrew](https://brew.sh), `ffmpeg`, `whisper-cpp`, [Ollama](https://ollama.com) (daemon running)
- Poetry 2.x

## Setup (one-time, internet required)

```bash
brew install ffmpeg whisper-cpp
poetry install
poetry run python -m scripts.env_checks       # sanity: OK/WARN/FAIL + fix hints
poetry run python -m scripts.setup_models     # downloads whisper model + ollama pull qwen3:14b
```

## Usage

```bash
# drop .m4a/.mp3/.wav recordings into input/, then:
poetry run python main.py

# re-process a file even if output exists:
poetry run python main.py --overwrite

# process specific files instead of the whole input/ dir:
poetry run python main.py path/to/a.m4a path/to/b.mp3
```

Each file produces three outputs in `output/<audio-name>/`:

- `transcription.txt` — original-language transcript (with timestamps in the JSON whisper produces in-memory; plain text only here)
- `transcription_en.txt` — English translation
- `summary.md` — consolidated summary

A failing file never breaks the batch — the run continues and reports which failed.

## Configuration (edit `config.yaml`)

- **STT model** — swap `stt.model_url` + `stt.model_filename` (e.g. the faster quantized
  `ggml-large-v3-turbo-q5_0.bin`), then re-run `python -m scripts.setup_models`.
  Full list: https://huggingface.co/ggerganov/whisper.cpp/resolve/main/
- **LLM model** — set `llm.model` (e.g. `qwen3:8b`), then `ollama pull <model>`.
- **Language** — `stt.language: auto` (detect per file) or force e.g. `zh`, `en`.
- **Chunking** — `chunk_tokens` (≈3000) and `overlap_sentences` (≈1) control LLM input size.
- **Prompts** — `prompts.*` are the translation + summary instructions; tune freely.
- **Denoise** — toggle `enable_denoise` (default off; lecture audio is clean).
  Requires extra deps: `poetry install --with denoise`.

## Optional: denoise

```bash
poetry install --with denoise   # adds noisereduce + soundfile
# set enable_denoise: true in config.yaml
```

## How it works (plain English)

1. `ffmpeg` converts the audio to 16 kHz mono WAV.
2. `whisper-cli` produces a timestamped transcript (large-v3-turbo, English+Chinese capable).
3. The transcript is split into ~3000-token chunks (split on sentence boundaries, 1-sentence overlap
   so the model keeps context). Each chunk is translated to English by Ollama; the overlap sentence
   is dropped when the translated chunks are rejoined. Overlap in the summary pass is disabled.
4. The summary uses **map-reduce**: each chunk is summarized into bullets, then one consolidation
   pass merges them into a single summary.

## Project layout

```
config.yaml            # the only file you should edit
main.py                # orchestrator
scripts/
  env_checks.py        # dependency sanity checks (CLI + importable)
  setup_models.py      # one-time model downloads
  common.py            # config loader, subprocess + logging helpers
  ingest.py            # ffmpeg → wav, optional denoise
  transcribe.py        # whisper-cli → segments
  llm.py               # Ollama client, chunking, translate + summarize
  export.py            # writes the 3 output files
input/ output/ models/
```
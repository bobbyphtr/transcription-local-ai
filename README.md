# Lecture Transcriber

Local, **fully offline** audio-to-text pipeline for course recordings:
`audio file` → `transcription.txt` (original language) + `transcription_en.txt` (English) + `summary.md`.

- 100% on-device. No cloud APIs, no internet at runtime.
- STT: `whisper.cpp` (`large-v3-turbo`), Metal-accelerated on Apple Silicon.
- LLM: Ollama (`gemma4:31b-mlx` by default) for translation + summary.
- Mandarin courses by default (`zh`); English courses with `--language en` (translation is skipped).

## Requirements

- macOS with Apple Silicon (Metal) — Intel/other OS works but slower
- [Homebrew](https://brew.sh), `ffmpeg`, `whisper-cpp`, [Ollama](https://ollama.com) (daemon running)
- Poetry 2.x

## Setup (one-time, internet required)

```bash
brew install ffmpeg whisper-cpp
poetry install
poetry run python -m scripts.env_checks       # sanity: OK/WARN/FAIL + fix hints
poetry run python -m scripts.setup_models     # downloads whisper model + ollama pull gemma4:31b-mlx
```

## Usage

```bash
# drop .m4a/.mp3/.wav recordings into input/, then:
poetry run python main.py

# re-process a file even if output exists:
poetry run python main.py --overwrite

# process specific files instead of the whole input/ dir:
poetry run python main.py path/to/a.m4a path/to/b.mp3

# course language (default: stt.language in config.yaml, which is zh):
poetry run python main.py --language zh         # Mandarin course
poetry run python main.py --language en path/to/english-lecture.mp3   # English course

# what to do with Ollama after the run (default unloads the model):
poetry run python main.py --shutdown none       # leave everything running
poetry run python main.py --shutdown daemon     # also stop the Ollama service
```

Each file produces three outputs in `output/<audio-name>/`:

- `transcription.txt` — original-language transcript (with timestamps in the JSON whisper produces in-memory; plain text only here)
- `transcription_en.txt` — English translation (for English courses: the transcript itself)
- `summary.md` — comprehensive summary (overview, sections, key terms, takeaways)

Files are written as each stage finishes, so `transcription.txt` is there before translation starts.

A failing file never breaks the batch — the run continues and reports which failed.

## Configuration (edit `config.yaml`)

- **STT model** — swap `stt.model_url` + `stt.model_filename` (e.g. the faster quantized
  `ggml-large-v3-turbo-q5_0.bin`), then re-run `python -m scripts.setup_models`.
  Full list: https://huggingface.co/ggerganov/whisper.cpp/resolve/main/
- **LLM model** — set `llm.model` (e.g. `qwen3:8b`), then `ollama pull <model>`.
- **Language** — `stt.language: zh` (Mandarin, default) or `en` (English, skips translation);
  override per run with `--language`. Avoid `auto`: whisper misdetects Mandarin lectures as
  English and writes a garbled English paraphrase.
- **Chunking** — `chunk_tokens` (≈3000, i.e. ~4500 Chinese characters) and `overlap_sentences` (≈1)
  control LLM input size. Keep a chunk's translation under `llm.num_predict`.
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
2. `whisper-cli` produces a timestamped transcript (large-v3-turbo) in the course language.
   `transcription.txt` is written right away, before the LLM stages start.
3. The transcript is split into ~3000-token (~4500-character) chunks (split on sentence boundaries, 1-sentence overlap
   so the model keeps context). Each chunk is translated to English by Ollama; the overlap sentence
   is dropped when the translated chunks are rejoined. Overlap in the summary pass is disabled.
   For English courses (`--language en`) this step is skipped.
4. The summary uses **map-reduce**: each chunk is turned into detailed study notes, then one
   consolidation pass merges them into a comprehensive summary (overview, sections, key terms,
   takeaways).
5. Loop safeguard: LLM replies are streamed, and if one starts repeating itself it is stopped, the
   model is reloaded and the chunk is retried (up to 2 times). A `⚠` warning is printed on every reset.
   If the source chunk itself repeats, there is no retry (reloading can't fix the input).
   Gemma's hidden "thinking" is turned off (`think: false`): it used up the output budget and cut
   translations short. If a reply still hits `num_predict`, a `⚠` warning says so.
   Whisper runs with `-mc 0` so it doesn't get stuck repeating one line in the first place.

## Tests

```bash
poetry run pytest -q
```

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
  llm.py               # Ollama client (streaming + loop safeguard), chunking, translate + summarize
  export.py            # writes each output file as its stage finishes
  progress.py          # stage timer, chunk counter, ⚠ warnings
tests/                 # pytest unit tests (no Ollama/whisper needed)
input/ output/ models/
```
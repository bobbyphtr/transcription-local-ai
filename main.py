"""Orchestrator: input/ → per-file pipeline → output/<name>/{transcription,transcription_en}.txt + summary.md"""
import argparse
import logging
import sys
import tempfile
from pathlib import Path

from scripts import export as export_mod
from scripts import ingest as ingest_mod
from scripts import llm as llm_mod
from scripts import transcribe as transcribe_mod
from scripts.common import (
    ConfigError,
    load_config,
    resolve_model_path,
    setup_logging,
    which,
)
from scripts.env_checks import check_env

logger = logging.getLogger("main")

AUDIO_EXTENSIONS = {".m4a", ".mp3", ".wav", ".aac", ".flac", ".ogg", ".wma"}
MODEL_MISSING_HINT = "STT model missing — run: python -m scripts.setup_models"


def preflight(cfg):
    check_env()  # raises RuntimeError with how-to-fix messages on any fail
    model_path = resolve_model_path(cfg)
    if not model_path.is_file() or model_path.stat().st_size == 0:
        raise RuntimeError(f"{MODEL_MISSING_HINT}  (missing {model_path})")
    if not which(cfg.get("stt", {}).get("whisper_binary", "whisper-cli")):
        raise RuntimeError("whisper-cli not found — brew install whisper-cpp")
    return model_path


def collect_inputs(input_dir: Path):
    files = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS]
    return sorted(files)


def process_file(cfg, audio_path, model_path, overwrite=False):
    output_dir = cfg["output_dir"]
    audio_name = audio_path.stem

    if not overwrite and export_mod.already_done(output_dir, audio_name):
        logger.info("skip (done): %s", audio_name)
        return "skipped"

    stt = cfg["stt"]
    llm_cfg = cfg["llm"]
    base_url = llm_cfg["base_url"]
    model = llm_cfg["model"]
    options = {
        "temperature": llm_cfg.get("temperature", 0.2),
        "num_ctx": llm_cfg.get("num_ctx", 8192),
        "num_predict": llm_cfg.get("num_predict", 4096),
    }
    max_chars = int(cfg["chunk_tokens"] * llm_mod.CHARS_PER_TOKEN)
    overlap = cfg["overlap_sentences"]

    client = llm_mod.OllamaClient(base_url, llm_cfg.get("timeout_seconds", 300))
    if not client.model_available(model):
        raise RuntimeError(f"LLM model '{model}' not present — run: ollama pull {model}")

    logger.info("processing: %s", audio_path.name)
    with tempfile.TemporaryDirectory(prefix="lt-") as tmp:
        tmp_path = Path(tmp)
        wav_path = ingest_mod.ingest(
            audio_path,
            tmp_path / (audio_name + ".wav"),
            sample_rate=stt.get("sample_rate", 16000),
            enable_denoise=cfg.get("enable_denoise", False),
        )
        segments = transcribe_mod.transcribe(
            wav_path,
            model_path,
            language=stt.get("language", "auto"),
            binary=stt.get("whisper_binary", "whisper-cli"),
        )
        plain_text = export_mod.ordered_text(segments)

        logger.info("translating (%d segments)…", len(segments))
        translation = llm_mod.translate(
            client, model,
            cfg["prompts"]["translate"], plain_text,
            max_chars, overlap, options,
        )
        logger.info("summarizing…")
        summary = llm_mod.summarize(
            client, model,
            cfg["prompts"]["summary_chunk"], cfg["prompts"]["summary_all"],
            plain_text, max_chars, options,
        )

    paths = export_mod.export(output_dir, audio_name, segments, translation, summary)
    for short, p in paths.items():
        logger.info("wrote %s → %s", short, p)
    return "done"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Transcribe lectures locally and offline.")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--overwrite", action="store_true", help="Re-process files that already have output")
    parser.add_argument("files", nargs="*", help="Optional audio files (default: all in input/)")
    args = parser.parse_args(argv)

    setup_logging()
    try:
        cfg, root = load_config(args.config)
    except ConfigError as e:
        logger.error("%s", e)
        return 1

    cfg["_root"] = root
    input_dir = (root / cfg["input_dir"]).resolve()
    cfg["output_dir"] = (root / cfg["output_dir"]).resolve()

    try:
        model_path = preflight(cfg)
    except RuntimeError as e:
        logger.error("%s", e)
        return 1

    if args.files:
        audio_files = [Path(f) for f in args.files]
    else:
        if not input_dir.is_dir():
            logger.error("input dir not found: %s", input_dir)
            return 1
        audio_files = collect_inputs(input_dir)

    if not audio_files:
        logger.info("no audio files to process in %s", input_dir)
        return 0

    results = {"done": 0, "skipped": 0, "failed": 0}
    for audio in audio_files:
        if not audio.is_file():
            logger.error("not a file, skipping: %s", audio)
            results["failed"] += 1
            continue
        try:
            status = process_file(cfg, audio, model_path, overwrite=args.overwrite)
            results[status] += 1
        except Exception as e:
            results["failed"] += 1
            logger.error("FAILED %s — %s", audio.name, e)

    logger.info("finished: %s", results)
    return 1 if results["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
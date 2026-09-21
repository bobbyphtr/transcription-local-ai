"""Transcribe: whisper-cli invocation → segments with timestamps."""
import json
from pathlib import Path

from scripts.common import run


def parse_segments(json_path: Path, txt_path: Path):
    if json_path.is_file():
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            for key in ("segments", "transcription"):
                value = data.get(key)
                if isinstance(value, list) and value:
                    return _normalize(value)
        except (json.JSONDecodeError, AttributeError):
            pass
    if txt_path.is_file():
        text = txt_path.read_text(encoding="utf-8").strip()
        return [{"start": None, "end": None, "text": text}] if text else []
    return []


def _normalize(raw_segments):
    segments = []
    for segment in raw_segments:
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        segments.append(
            {
                "start": _num(segment.get("start")),
                "end": _num(segment.get("end")),
                "text": text,
            }
        )
    return segments


def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def transcribe(wav_path, model_path: Path, language="auto", binary="whisper-cli", timeout=1800):
    base = wav_path.with_suffix("")
    cmd = [binary, "-m", str(model_path), "-f", str(wav_path), "-otxt", "-oj", "-of", str(base)]
    if language and language != "auto":
        cmd += ["-l", language]
    run(cmd, timeout=timeout)
    segments = parse_segments(base.with_suffix(".json"), base.with_suffix(".txt"))
    if not segments:
        raise RuntimeError("whisper-cli produced no output — check inputs and model.")
    return segments
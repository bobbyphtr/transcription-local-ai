"""Export: write each output file into output/<audio-name>/ as its stage finishes."""
from pathlib import Path

OUTPUT_FILES = {
    "transcription": "transcription.txt",
    "translation": "transcription_en.txt",
    "summary": "summary.md",
}


def output_paths(output_dir, audio_name):
    return {key: (output_dir / audio_name / filename) for key, filename in OUTPUT_FILES.items()}


def already_done(output_dir, audio_name):
    paths = output_paths(output_dir, audio_name)
    return all(p.is_file() for p in paths.values())


def ordered_text(segments):
    return "\n".join(seg["text"] for seg in segments)


def write_output(output_dir, audio_name, key, text):
    path = output_paths(output_dir, audio_name)[key]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path

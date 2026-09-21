"""Export: write the three output files into output/<audio-name>/."""
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


def export(output_dir, audio_name, segments, translation, summary):
    paths = output_paths(output_dir, audio_name)
    paths["transcription"].parent.mkdir(parents=True, exist_ok=True)
    paths["transcription"].write_text(ordered_text(segments), encoding="utf-8")
    paths["translation"].write_text(translation, encoding="utf-8")
    paths["summary"].write_text(summary + "\n", encoding="utf-8")
    return paths
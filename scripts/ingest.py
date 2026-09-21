"""Ingest: convert audio to 16 kHz mono WAV, optional noise reduction."""
from pathlib import Path

from scripts.common import run


def probe_duration(input_path) -> float:
    """Return the media length in seconds via ffprobe (0.0 if it fails)."""
    proc = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(input_path),
        ]
    )
    try:
        return float(proc.stdout.strip())
    except (AttributeError, ValueError):
        return 0.0


def to_wav(input_path, wav_path: Path, sample_rate=16000, channels=1):
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        str(wav_path),
    ]
    run(cmd)
    return wav_path


def denoise_wav(wav_path: Path):
    import soundfile as sf

    try:
        import noisereduce as nr
    except ImportError:
        raise RuntimeError(
            "noisereduce missing — install the denoise group: poetry install --with denoise"
        )
    audio_data, sample_rate = sf.read(wav_path, dtype="float32")
    reduced = nr.reduce_noise(y=audio_data, sr=sample_rate)
    sf.write(wav_path, reduced, sample_rate)
    return wav_path


def ingest(input_path, wav_path: Path, sample_rate=16000, enable_denoise=False):
    to_wav(input_path, wav_path, sample_rate=sample_rate)
    if enable_denoise:
        denoise_wav(wav_path)
    return wav_path
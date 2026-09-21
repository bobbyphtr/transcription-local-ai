"""Ingest: convert audio to 16 kHz mono WAV, optional noise reduction."""
from pathlib import Path

from scripts.common import run


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
    data, sr = sf.read(wav_path, dtype="float32")
    reduced = nr.reduce_noise(y=data, sr=sr)
    sf.write(wav_path, reduced, sr)
    return wav_path


def ingest(input_path, wav_path: Path, sample_rate=16000, enable_denoise=False):
    to_wav(input_path, wav_path, sample_rate=sample_rate)
    if enable_denoise:
        denoise_wav(wav_path)
    return wav_path
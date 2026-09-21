"""Environment sanity checks for the transcription pipeline.

Importable (run_checks / check_env) and a standalone CLI:

    python -m scripts.env_checks [--config config.yaml]

Exit codes: 0 = ok (warns allowed), 1 = at least one FAIL.
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

from scripts.common import load_config, PROJECT_ROOT, resolve_model_path

STATUS_LABELS = {"ok": "OK  ", "warn": "WARN", "fail": "FAIL"}


def _http_json(url, timeout=5):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def check_brew(cfg):
    from scripts.common import which

    if which("brew"):
        return "ok", "brew found"
    return "fail", "brew not found — install Homebrew: https://brew.sh"


def check_ffmpeg(cfg):
    from scripts.common import which

    if which("ffmpeg"):
        return "ok", "ffmpeg found"
    return "fail", "ffmpeg not found — brew install ffmpeg"


def check_whisper_cli(cfg):
    from scripts.common import which

    binary = cfg.get("stt", {}).get("whisper_binary", "whisper-cli")
    if which(binary):
        return "ok", f"{binary} found"
    return "fail", f"{binary} not found — brew install whisper-cpp"


def check_ollama_binary(cfg):
    from scripts.common import which

    if which("ollama"):
        return "ok", "ollama found"
    return "fail", "ollama not found — install from https://ollama.com"


def check_ollama_server(cfg):
    base = cfg.get("llm", {}).get("base_url", "http://localhost:11434")
    try:
        tags = _http_json(f"{base}/api/tags", timeout=5)
        n = len(tags.get("models", []))
        return "ok", f"ollama server up ({base}), {n} model(s) loaded"
    except Exception:
        return "fail", f"ollama server unreachable at {base} — is 'ollama serve' running?"


def check_whisper_model(cfg):
    path = resolve_model_path(cfg)
    if path.is_file() and path.stat().st_size > 0:
        size_mb = path.stat().st_size / 1024 / 1024
        return "ok", f"whisper model present ({size_mb:.0f} MiB)"
    return "warn", f"whisper model missing — python -m scripts.setup_models"


def check_ollama_model(cfg):
    base = cfg.get("llm", {}).get("base_url", "http://localhost:11434")
    model = cfg.get("llm", {}).get("model", "")
    try:
        tags = _http_json(f"{base}/api/tags", timeout=5)
        names = {m.get("name", "") for m in tags.get("models", [])}
        if model in names:
            return "ok", f"LLM model '{model}' present"
        return "warn", f"LLM model '{model}' missing — ollama pull {model}"
    except Exception:
        return "fail", f"cannot list ollama models at {base}"


CHECKS = [
    check_brew,
    check_ffmpeg,
    check_whisper_cli,
    check_ollama_binary,
    check_ollama_server,
    check_ollama_model,
    check_whisper_model,
]


def run_checks(config=None):
    cfg, _ = load_config(config)
    results = {}
    for fn in CHECKS:
        status, message = fn(cfg)
        results[fn.__name__] = (status, message)
    return results


def check_env(config=None):
    results = run_checks(config)
    problems = []
    for name, (status, message) in results.items():
        print(f"  [{STATUS_LABELS[status]}] {name}: {message}")
        problems.append((name, status, message))
    n_fail = sum(1 for _, s, _ in problems if s == "fail")
    n_warn = sum(1 for _, s, _ in problems if s == "warn")
    print(f"\n{n_fail} failure(s), {n_warn} warning(s)")
    if n_fail:
        raise RuntimeError("Environment has failures — see messages above.")
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description="Check transcription dependencies.")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args(argv)
    try:
        check_env(args.config)
    except RuntimeError as e:
        print(f"\nerror: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
"""One-time setup: download the whisper model and pull the Ollama LLM. Internet required."""
import argparse
import subprocess
import sys
from pathlib import Path
from urllib.request import urlopen

from scripts.common import load_config, resolve_model_path, run


def download_with_progress(url, dest: Path):
    if dest.is_file() and dest.stat().st_size > 0:
        size_mb = dest.stat().st_size / 1024 / 1024
        print(f"whisper model already present ({size_mb:.0f} MiB) — skipping.")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {url}")
    with urlopen(url) as resp, tmp.open("wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            block = resp.read(1 << 20)
            if not block:
                break
            out.write(block)
            done += len(block)
            if total:
                pct = done / total * 100
                print(f"\r  {pct:5.1f}%  {done / 1024 / 1024:.0f}/{total / 1024 / 1024:.0f} MiB", end="", flush=True)
    print()
    tmp.rename(dest)
    print(f"saved to {dest}")
    return dest


def pull_ollama(model: str):
    print(f"ollama pull {model}  (this can take a while)")
    return_code = subprocess.call(["ollama", "pull", model])
    if return_code != 0:
        raise RuntimeError(f"ollama pull {model} failed (exit {return_code})")
    print(f"ollama model '{model}' ready.")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Download STT/LLM models (one-time).")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--force", action="store_true", help="Re-download whisper model even if present")
    args = parser.parse_args(argv)

    cfg, _ = load_config(args.config)
    stt = cfg.get("stt", {})
    llm = cfg.get("llm", {})

    model_path = resolve_model_path(cfg)
    if args.force and model_path.exists():
        model_path.unlink()
    download_with_progress(stt["model_url"], model_path)
    pull_ollama(llm["model"])
    print("\nSetup complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
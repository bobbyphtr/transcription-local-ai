"""Shared helpers: config loading, path resolution, subprocess, logging."""
import logging
import shutil
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"

import yaml


class ConfigError(Exception):
    pass


def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def load_config(path=None):
    cfg_path = Path(path).expanduser() if path else DEFAULT_CONFIG
    if not cfg_path.is_file():
        raise ConfigError(f"Config file not found: {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(cfg, dict):
        raise ConfigError(f"Config file is malformed: {cfg_path}")
    return cfg, cfg_path.parent


def resolve_path(config, key):
    root = config.get("_root", PROJECT_ROOT)
    value = config.get(key, key)
    return (root / value).resolve() if isinstance(value, str) else value


def resolve_model_path(config):
    models_dir = resolve_path(config, "models_dir")
    filename = config.get("stt", {}).get("model_filename", "ggml-large-v3-turbo.bin")
    return models_dir / filename


def run(cmd, timeout=None, check=True, capture=True):
    logger = logging.getLogger("common.run")
    logger.info("running: %s", " ".join(str(c) for c in cmd))
    try:
        if capture:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        else:
            proc = subprocess.run(cmd, timeout=timeout)
    except FileNotFoundError:
        raise RuntimeError(f"Command not found: {cmd[0]}. Install it and try again.")
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Command timed out after {timeout}s: {cmd[0]}")
    if check and proc.returncode != 0:
        detail = ""
        if capture and proc.stderr:
            detail = ": " + proc.stderr.strip()[-1500:]
        raise RuntimeError(f"Command failed ({proc.returncode}) — {cmd[0]}{detail}")
    return proc


def which(name):
    return shutil.which(name)